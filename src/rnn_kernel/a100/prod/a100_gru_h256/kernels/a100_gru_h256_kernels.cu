#include <math_functions.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32

__forceinline__ __device__ float half_warp_reduce_sum(float value, unsigned int mask)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset, 16);
    }
    return value;
}

extern "C" __global__
void a100_gru_h256_pack_hidden_prev_time_major_kernel(
    const float* __restrict__ h0,
    const float* __restrict__ output,
    float* __restrict__ hidden_prev_steps,
    int batch_size,
    int seq_len)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int HIDDEN_VEC4 = HIDDEN_SIZE / 4;

    const float4* h0_vec = reinterpret_cast<const float4*>(h0);
    const float4* output_vec = reinterpret_cast<const float4*>(output);
    float4* hidden_prev_vec = reinterpret_cast<float4*>(hidden_prev_steps);
    const int total_vec = seq_len * batch_size * HIDDEN_VEC4;

    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_vec;
         linear += blockDim.x * gridDim.x) {
        const int h_vec = linear % HIDDEN_VEC4;
        const int batch_step = linear / HIDDEN_VEC4;
        const int batch_idx = batch_step % batch_size;
        const int step = batch_step / batch_size;

        // 直接写成 [time, batch, hidden]，避免 torch.cat + transpose + contiguous 的布局搬运。
        const float4 value = (step == 0)
            ? h0_vec[batch_idx * HIDDEN_VEC4 + h_vec]
            : output_vec[(batch_idx * seq_len + step - 1) * HIDDEN_VEC4 + h_vec];
        hidden_prev_vec[linear] = value;
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_shmem_gate_cache_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ partial_gates,
    float* __restrict__ hidden_state,
    float* __restrict__ output,
    float* __restrict__ gate_cache,
    int seq_len)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int H_TILES = 4;
    constexpr int H_TILE = HIDDEN_SIZE / H_TILES;
    constexpr int K_CTAS = 4;
    constexpr int K_TILE = HIDDEN_SIZE / K_CTAS;
    constexpr int CTAS_PER_BATCH = H_TILES * K_CTAS;
    constexpr int COMPACT_PARTIALS_PER_BATCH = H_TILES * (K_CTAS - 1);
    constexpr int PARTIAL_STRIDE = 3 * H_TILE;
    constexpr int GROUPS_PER_BLOCK = 16;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int h_tile_idx = cta_idx >> 2;
    const int k_cta_idx = cta_idx & (K_CTAS - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid >> 4;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
    const int partial_base_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
        + h_tile_idx * (K_CTAS - 1);
    float* partial_out_base = partial0_local;
    if (k_cta_idx != 0) {
        partial_out_base = partial_gates + (partial_base_idx + k_cta_idx - 1) * PARTIAL_STRIDE;
    }
    const float* partial1_base = partial_gates + partial_base_idx * PARTIAL_STRIDE;
    const float* partial2_base = partial1_base + PARTIAL_STRIDE;
    const float* partial3_base = partial2_base + PARTIAL_STRIDE;
    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;

    if (k_cta_idx == 0) {
        for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
            const int hid = h_begin + h_local;
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // 每个 half-warp 固定负责 4 行 hidden，按 2 行一组计算来控制寄存器占用。
#pragma unroll
        for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
            const int pair_base = pair_idx * 2 * GROUPS_PER_BLOCK;
            const int h0_local = group_idx + pair_base;
            const int h1_local = h0_local + GROUPS_PER_BLOCK;
            const int hid0 = h_begin + h0_local;
            const int hid1 = h_begin + h1_local;

            const float* weight_r0 = weight_hh + hid0 * HIDDEN_SIZE + k_begin;
            const float* weight_z0 = weight_hh + (HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
            const float* weight_n0 = weight_hh + (2 * HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
            const float* weight_r1 = weight_hh + hid1 * HIDDEN_SIZE + k_begin;
            const float* weight_z1 = weight_hh + (HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;
            const float* weight_n1 = weight_hh + (2 * HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;

            float acc_r0 = 0.0f;
            float acc_z0 = 0.0f;
            float acc_n0 = 0.0f;
            float acc_r1 = 0.0f;
            float acc_z1 = 0.0f;
            float acc_n1 = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r0 += hidden_value * weight_r0[k_local];
                acc_z0 += hidden_value * weight_z0[k_local];
                acc_n0 += hidden_value * weight_n0[k_local];
                acc_r1 += hidden_value * weight_r1[k_local];
                acc_z1 += hidden_value * weight_z1[k_local];
                acc_n1 += hidden_value * weight_n1[k_local];
            }

            acc_r0 = half_warp_reduce_sum(acc_r0, group_mask);
            acc_z0 = half_warp_reduce_sum(acc_z0, group_mask);
            acc_n0 = half_warp_reduce_sum(acc_n0, group_mask);
            acc_r1 = half_warp_reduce_sum(acc_r1, group_mask);
            acc_z1 = half_warp_reduce_sum(acc_z1, group_mask);
            acc_n1 = half_warp_reduce_sum(acc_n1, group_mask);

            if (lane == 0) {
                partial_out_base[h0_local] = acc_r0;
                partial_out_base[H_TILE + h0_local] = acc_z0;
                partial_out_base[2 * H_TILE + h0_local] = acc_n0;
                partial_out_base[h1_local] = acc_r1;
                partial_out_base[H_TILE + h1_local] = acc_z1;
                partial_out_base[2 * H_TILE + h1_local] = acc_n1;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1_base[h_local]
                    + partial2_base[h_local]
                    + partial3_base[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1_base[H_TILE + h_local]
                    + partial2_base[H_TILE + h_local]
                    + partial3_base[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1_base[2 * H_TILE + h_local]
                    + partial2_base[2 * H_TILE + h_local]
                    + partial3_base[2 * H_TILE + h_local];

                const float i_r = input_step[hid];
                const float i_z = input_step[HIDDEN_SIZE + hid];
                const float i_n = input_step[2 * HIDDEN_SIZE + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                hidden[hid] = hidden_next;
                output_step[hid] = hidden_next;
                cache_step[hid] = reset_gate;
                cache_step[HIDDEN_SIZE + hid] = update_gate;
                cache_step[2 * HIDDEN_SIZE + hid] = new_gate;
                cache_step[3 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_no_cache_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ partial_gates,
    float* __restrict__ hidden_state,
    float* __restrict__ output,
    int seq_len)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int H_TILES = 4;
    constexpr int H_TILE = HIDDEN_SIZE / H_TILES;
    constexpr int K_CTAS = 4;
    constexpr int K_TILE = HIDDEN_SIZE / K_CTAS;
    constexpr int CTAS_PER_BATCH = H_TILES * K_CTAS;
    constexpr int COMPACT_PARTIALS_PER_BATCH = H_TILES * (K_CTAS - 1);
    constexpr int PARTIAL_STRIDE = 3 * H_TILE;
    constexpr int GROUPS_PER_BLOCK = 16;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int h_tile_idx = cta_idx >> 2;
    const int k_cta_idx = cta_idx & (K_CTAS - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid >> 4;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
    const int partial_base_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
        + h_tile_idx * (K_CTAS - 1);
    float* partial_out_base = partial0_local;
    if (k_cta_idx != 0) {
        partial_out_base = partial_gates + (partial_base_idx + k_cta_idx - 1) * PARTIAL_STRIDE;
    }
    const float* partial1_base = partial_gates + partial_base_idx * PARTIAL_STRIDE;
    const float* partial2_base = partial1_base + PARTIAL_STRIDE;
    const float* partial3_base = partial2_base + PARTIAL_STRIDE;
    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;

    if (k_cta_idx == 0) {
        for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
            const int hid = h_begin + h_local;
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;

        // 推理路径不写 gate cache，只保留 recurrent projection 和输出状态。
#pragma unroll
        for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
            const int pair_base = pair_idx * 2 * GROUPS_PER_BLOCK;
            const int h0_local = group_idx + pair_base;
            const int h1_local = h0_local + GROUPS_PER_BLOCK;
            const int hid0 = h_begin + h0_local;
            const int hid1 = h_begin + h1_local;

            const float* weight_r0 = weight_hh + hid0 * HIDDEN_SIZE + k_begin;
            const float* weight_z0 = weight_hh + (HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
            const float* weight_n0 = weight_hh + (2 * HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
            const float* weight_r1 = weight_hh + hid1 * HIDDEN_SIZE + k_begin;
            const float* weight_z1 = weight_hh + (HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;
            const float* weight_n1 = weight_hh + (2 * HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;

            float acc_r0 = 0.0f;
            float acc_z0 = 0.0f;
            float acc_n0 = 0.0f;
            float acc_r1 = 0.0f;
            float acc_z1 = 0.0f;
            float acc_n1 = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r0 += hidden_value * weight_r0[k_local];
                acc_z0 += hidden_value * weight_z0[k_local];
                acc_n0 += hidden_value * weight_n0[k_local];
                acc_r1 += hidden_value * weight_r1[k_local];
                acc_z1 += hidden_value * weight_z1[k_local];
                acc_n1 += hidden_value * weight_n1[k_local];
            }

            acc_r0 = half_warp_reduce_sum(acc_r0, group_mask);
            acc_z0 = half_warp_reduce_sum(acc_z0, group_mask);
            acc_n0 = half_warp_reduce_sum(acc_n0, group_mask);
            acc_r1 = half_warp_reduce_sum(acc_r1, group_mask);
            acc_z1 = half_warp_reduce_sum(acc_z1, group_mask);
            acc_n1 = half_warp_reduce_sum(acc_n1, group_mask);

            if (lane == 0) {
                partial_out_base[h0_local] = acc_r0;
                partial_out_base[H_TILE + h0_local] = acc_z0;
                partial_out_base[2 * H_TILE + h0_local] = acc_n0;
                partial_out_base[h1_local] = acc_r1;
                partial_out_base[H_TILE + h1_local] = acc_z1;
                partial_out_base[2 * H_TILE + h1_local] = acc_n1;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1_base[h_local]
                    + partial2_base[h_local]
                    + partial3_base[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1_base[H_TILE + h_local]
                    + partial2_base[H_TILE + h_local]
                    + partial3_base[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1_base[2 * H_TILE + h_local]
                    + partial2_base[2 * H_TILE + h_local]
                    + partial3_base[2 * H_TILE + h_local];

                const float i_r = input_step[hid];
                const float i_z = input_step[HIDDEN_SIZE + hid];
                const float i_n = input_step[2 * HIDDEN_SIZE + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                hidden[hid] = hidden_next;
                output_step[hid] = hidden_next;
            }
        }
        grid.sync();
    }
}

template <int SPLIT_COUNT>
__device__ __forceinline__ float a100_gru_h256_sum_partials_from1(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 1; split < SPLIT_COUNT; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
}

template <int SPLIT_COUNT>
__device__ __forceinline__ float a100_gru_h256_sum_partials(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 0; split < SPLIT_COUNT; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split6_gate_cache_state_tiled_weight_shmem_split0_keep_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ gate_cache,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates_steps,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_state,
    int batch_size,
    int seq_len)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;
    constexpr int SPLIT_COUNT = 6;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_storage[];
    float* shared_grad_gates = shared_storage;
    float* shared_weight_tile = shared_storage + SPLIT_SIZE;
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / SPLIT_COUNT;
    const int split_idx = global_cta - batch_idx * SPLIT_COUNT;
    const int hid = threadIdx.x;
    const int split_begin = split_idx * SPLIT_SIZE;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    // split6 用约 131.6KB dynamic shared memory 缓存 recurrent weight tile。
    float split0_partial_state = 0.0f;

#pragma unroll 4
    for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
        const int gate_idx = split_begin + local_idx;
        shared_weight_tile[local_idx * HIDDEN_SIZE + hid] =
            weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    __syncthreads();

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_prev_direct = 0.0f;
        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        if (split_idx == 0) {
            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[hidden_base];
            } else {
                grad_hidden_acc = split0_partial_state
                    + a100_gru_h256_sum_partials_from1<SPLIT_COUNT>(
                        partial_sums, partial_base);
            }

            const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + hid;
            const float reset_gate = gate_cache[cache_base];
            const float update_gate = gate_cache[cache_base + HIDDEN_SIZE];
            const float new_gate = gate_cache[cache_base + 2 * HIDDEN_SIZE];
            const float recurrent_new = gate_cache[cache_base + 3 * HIDDEN_SIZE];
            const float h_prev = (step == 0)
                ? h0[hidden_base]
                : output[output_base - HIDDEN_SIZE];
            const float grad_out = grad_hidden_acc + grad_output[output_base];

            const float grad_update = grad_out * (h_prev - new_gate);
            const float grad_new = grad_out * (1.0f - update_gate);
            const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
            const float grad_reset = grad_new_pre * recurrent_new;
            const float grad_recurrent_n = grad_new_pre * reset_gate;
            const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
            const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
            grad_hidden_prev_direct = grad_out * update_gate;

            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;

            if (hid < SPLIT_SIZE) {
                shared_grad_gates[hid] = grad_reset_pre;
            }
        } else if (hid < SPLIT_SIZE) {
            const int gate_idx = split_begin + hid;
            const int gate_type = gate_idx / HIDDEN_SIZE;
            const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
            const int gate_hidden_base = batch_idx * HIDDEN_SIZE + gate_hid;
            const int gate_partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;
            const int gate_output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + gate_hid;

            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[gate_hidden_base];
            } else {
                grad_hidden_acc =
                    a100_gru_h256_sum_partials<SPLIT_COUNT>(partial_sums, gate_partial_base);
            }

            const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + gate_hid;
            const float reset_gate = gate_cache[cache_base];
            const float update_gate = gate_cache[cache_base + HIDDEN_SIZE];
            const float new_gate = gate_cache[cache_base + 2 * HIDDEN_SIZE];
            const float recurrent_new = gate_cache[cache_base + 3 * HIDDEN_SIZE];
            const float h_prev = (step == 0)
                ? h0[gate_hidden_base]
                : output[gate_output_base - HIDDEN_SIZE];
            const float grad_out = grad_hidden_acc + grad_output[gate_output_base];

            const float grad_update = grad_out * (h_prev - new_gate);
            const float grad_new = grad_out * (1.0f - update_gate);
            const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
            const float grad_reset = grad_new_pre * recurrent_new;
            const float grad_recurrent_n = grad_new_pre * reset_gate;
            const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
            const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);

            float local_grad = grad_reset_pre;
            if (gate_type == 1) {
                local_grad = grad_update_pre;
            } else if (gate_type == 2) {
                local_grad = grad_recurrent_n;
            }
            shared_grad_gates[hid] = local_grad;
        }
        __syncthreads();

        float partial = 0.0f;
#pragma unroll 4
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            partial += shared_grad_gates[local_idx]
                * shared_weight_tile[local_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
            split0_partial_state = partial;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = split0_partial_state
            + a100_gru_h256_sum_partials_from1<SPLIT_COUNT>(partial_sums, partial_base);
    }
}
