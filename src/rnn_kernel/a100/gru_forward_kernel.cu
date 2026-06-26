#include <math_functions.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

__forceinline__ __device__ float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

__forceinline__ __device__ float half_warp_reduce_sum(float value, unsigned int mask)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset, 16);
    }
    return value;
}

__forceinline__ __device__ float quarter_warp_reduce_sum(float value, unsigned int mask)
{
#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset, 8);
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

        // 直接写成 [time, batch, hidden]，避免 torch.cat + transpose + contiguous 的两次布局搬运。
        const float4 value = (step == 0)
            ? h0_vec[batch_idx * HIDDEN_VEC4 + h_vec]
            : output_vec[(batch_idx * seq_len + step - 1) * HIDDEN_VEC4 + h_vec];
        hidden_prev_vec[linear] = value;
    }
}

extern "C" __global__
void a100_gru_forward_layer_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h0,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int input_size,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden = shared;
    float* hidden_gates = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_idx = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* x_step = x + (batch_idx * seq_len + step) * input_size;

        // hidden projection 是主要成本，每个 warp 负责一个 gate/hid dot-product。
        for (int out_idx = warp_idx; out_idx < 3 * hidden_size; out_idx += warps_per_block) {
            const int gate = out_idx / hidden_size;
            const int hid = out_idx - gate * hidden_size;
            float acc = (lane == 0) ? bias_hh[gate * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += WARP_SIZE) {
                acc += hidden[k] * weight_hh[(gate * hidden_size + hid) * hidden_size + k];
            }

            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                hidden_gates[out_idx] = acc;
            }
        }
        __syncthreads();

        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            float i_r = bias_ih[hid];
            float i_z = bias_ih[hidden_size + hid];
            float i_n = bias_ih[2 * hidden_size + hid];
            for (int k = 0; k < input_size; ++k) {
                const float x_value = x_step[k];
                i_r += x_value * weight_ih[hid * input_size + k];
                i_z += x_value * weight_ih[(hidden_size + hid) * input_size + k];
                i_n += x_value * weight_ih[(2 * hidden_size + hid) * input_size + k];
            }

            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + hidden_gates[hid])));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + hidden_gates[hidden_size + hid])));
            const float new_gate = tanhf(i_n + reset_gate * hidden_gates[2 * hidden_size + hid]);
            const float hidden_prev = hidden[hid];
            const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden = shared;
    float* hidden_gates = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_idx = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;

        // input projection 已由 cuBLAS 完成，这里只做 recurrent projection。
        for (int out_idx = warp_idx; out_idx < 3 * hidden_size; out_idx += warps_per_block) {
            const int gate = out_idx / hidden_size;
            const int hid = out_idx - gate * hidden_size;
            float acc = (lane == 0) ? bias_hh[gate * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += WARP_SIZE) {
                acc += hidden[k] * weight_hh[(gate * hidden_size + hid) * hidden_size + k];
            }

            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                hidden_gates[out_idx] = acc;
            }
        }
        __syncthreads();

        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            const float i_r = input_step[hid];
            const float i_z = input_step[hidden_size + hid];
            const float i_n = input_step[2 * hidden_size + hid];
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + hidden_gates[hid])));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + hidden_gates[hidden_size + hid])));
            const float new_gate = tanhf(i_n + reset_gate * hidden_gates[2 * hidden_size + hid]);
            const float hidden_prev = hidden[hid];
            const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_half_warp_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden = shared;
    float* hidden_gates = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;

        // 一个 warp 拆成两个 half-warp，同时计算两个 gate/hid dot-product。
        for (int out_idx = group_idx; out_idx < 3 * hidden_size; out_idx += groups_per_block) {
            const int gate = out_idx / hidden_size;
            const int hid = out_idx - gate * hidden_size;
            float acc = (lane == 0) ? bias_hh[gate * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += 16) {
                acc += hidden[k] * weight_hh[(gate * hidden_size + hid) * hidden_size + k];
            }

            acc = half_warp_reduce_sum(acc, group_mask);
            if (lane == 0) {
                hidden_gates[out_idx] = acc;
            }
        }
        __syncthreads();

        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            const float i_r = input_step[hid];
            const float i_z = input_step[hidden_size + hid];
            const float i_n = input_step[2 * hidden_size + hid];
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + hidden_gates[hid])));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + hidden_gates[hidden_size + hid])));
            const float new_gate = tanhf(i_n + reset_gate * hidden_gates[2 * hidden_size + hid]);
            const float hidden_prev = hidden[hid];
            const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_quarter_warp_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden = shared;
    float* hidden_gates = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 7;
    const int group_idx = tid / 8;
    const int groups_per_block = blockDim.x / 8;
    const unsigned int group_mask = 0xffu << (warp_lane & 24);

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;

        // 一个 warp 拆成四个 quarter-warp，减少输出维循环轮数。
        for (int out_idx = group_idx; out_idx < 3 * hidden_size; out_idx += groups_per_block) {
            const int gate = out_idx / hidden_size;
            const int hid = out_idx - gate * hidden_size;
            float acc = (lane == 0) ? bias_hh[gate * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += 8) {
                acc += hidden[k] * weight_hh[(gate * hidden_size + hid) * hidden_size + k];
            }

            acc = quarter_warp_reduce_sum(acc, group_mask);
            if (lane == 0) {
                hidden_gates[out_idx] = acc;
            }
        }
        __syncthreads();

        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            const float i_r = input_step[hid];
            const float i_z = input_step[hidden_size + hid];
            const float i_n = input_step[2 * hidden_size + hid];
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + hidden_gates[hid])));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + hidden_gates[hidden_size + hid])));
            const float new_gate = tanhf(i_n + reset_gate * hidden_gates[2 * hidden_size + hid]);
            const float hidden_prev = hidden[hid];
            const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_fused_half_warp_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden = shared;
    float* next_hidden = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;

        // 一个 half-warp 负责一个 hidden index，同时计算 r/z/n 三个 recurrent gate。
        for (int hid = group_idx; hid < hidden_size; hid += groups_per_block) {
            float acc_r = (lane == 0) ? bias_hh[hid] : 0.0f;
            float acc_z = (lane == 0) ? bias_hh[hidden_size + hid] : 0.0f;
            float acc_n = (lane == 0) ? bias_hh[2 * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += 16) {
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_hh[hid * hidden_size + k];
                acc_z += hidden_value * weight_hh[(hidden_size + hid) * hidden_size + k];
                acc_n += hidden_value * weight_hh[(2 * hidden_size + hid) * hidden_size + k];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                const float i_r = input_step[hid];
                const float i_z = input_step[hidden_size + hid];
                const float i_n = input_step[2 * hidden_size + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                next_hidden[hid] = new_gate + update_gate * (hidden_prev - new_gate);
            }
        }
        __syncthreads();

        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            const float hidden_next = next_hidden[hid];
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_fused_pingpong_half_warp_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int seq_len,
    int hidden_size)
{
    extern __shared__ float shared[];
    float* hidden_even = shared;
    float* hidden_odd = shared + hidden_size;

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
        hidden_even[hid] = h0[batch_idx * hidden_size + hid];
    }
    __syncthreads();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;
        const float* read_hidden = (step & 1) ? hidden_odd : hidden_even;
        float* write_hidden = (step & 1) ? hidden_even : hidden_odd;

        // ping-pong shared buffer 避免每步再做 next_hidden -> hidden 的拷贝。
        for (int hid = group_idx; hid < hidden_size; hid += groups_per_block) {
            float acc_r = (lane == 0) ? bias_hh[hid] : 0.0f;
            float acc_z = (lane == 0) ? bias_hh[hidden_size + hid] : 0.0f;
            float acc_n = (lane == 0) ? bias_hh[2 * hidden_size + hid] : 0.0f;

            for (int k = lane; k < hidden_size; k += 16) {
                const float hidden_value = read_hidden[k];
                acc_r += hidden_value * weight_hh[hid * hidden_size + k];
                acc_z += hidden_value * weight_hh[(hidden_size + hid) * hidden_size + k];
                acc_n += hidden_value * weight_hh[(2 * hidden_size + hid) * hidden_size + k];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                const float i_r = input_step[hid];
                const float i_z = input_step[hidden_size + hid];
                const float i_n = input_step[2 * hidden_size + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = read_hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                write_hidden[hid] = hidden_next;
                output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ partial_gates,
    float* __restrict__ hidden_state,
    float* __restrict__ output,
    int seq_len,
    int hidden_size,
    int ctas_per_batch)
{
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / ctas_per_batch;
    const int cta_idx = global_cta - batch_idx * ctas_per_batch;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * hidden_size;
    float* partial_base = partial_gates + global_cta * 3 * hidden_size;

    if (cta_idx == 0) {
        for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * hidden_size + hid];
        }
    }
    grid.sync();

    const int k_begin = (hidden_size * cta_idx) / ctas_per_batch;
    const int k_end = (hidden_size * (cta_idx + 1)) / ctas_per_batch;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * hidden_size;

        // 多个 CTA 按 hidden 维切分 recurrent dot-product，先写 partial sums。
        for (int hid = group_idx; hid < hidden_size; hid += groups_per_block) {
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

            for (int k = k_begin + lane; k < k_end; k += 16) {
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_hh[hid * hidden_size + k];
                acc_z += hidden_value * weight_hh[(hidden_size + hid) * hidden_size + k];
                acc_n += hidden_value * weight_hh[(2 * hidden_size + hid) * hidden_size + k];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                partial_base[hid] = acc_r;
                partial_base[hidden_size + hid] = acc_z;
                partial_base[2 * hidden_size + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            for (int hid = tid; hid < hidden_size; hid += blockDim.x) {
                float acc_r = bias_hh[hid];
                float acc_z = bias_hh[hidden_size + hid];
                float acc_n = bias_hh[2 * hidden_size + hid];

                for (int cta = 0; cta < ctas_per_batch; ++cta) {
                    const float* partial = partial_gates
                        + (batch_idx * ctas_per_batch + cta) * 3 * hidden_size;
                    acc_r += partial[hid];
                    acc_z += partial[hidden_size + hid];
                    acc_n += partial[2 * hidden_size + hid];
                }

                const float i_r = input_step[hid];
                const float i_z = input_step[hidden_size + hid];
                const float i_n = input_step[2 * hidden_size + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                hidden[hid] = hidden_next;
                output[(batch_idx * seq_len + step) * hidden_size + hid] = hidden_next;
            }
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;

        // h256 固定 4 CTA，每个 CTA 处理 64 维 k-tile。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_hh[hid * HIDDEN_SIZE + k];
                acc_z += hidden_value * weight_hh[(HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
                acc_n += hidden_value * weight_hh[(2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                partial_base[hid] = acc_r;
                partial_base[HIDDEN_SIZE + hid] = acc_z;
                partial_base[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float* partial0 = partial_gates + (batch_idx * CTAS_PER_BATCH) * 3 * HIDDEN_SIZE;
                const float* partial1 = partial0 + 3 * HIDDEN_SIZE;
                const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
                const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

                const float acc_r = bias_hh[hid]
                    + partial0[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

                const float i_r = input_step[hid];
                const float i_z = input_step[HIDDEN_SIZE + hid];
                const float i_n = input_step[2 * HIDDEN_SIZE + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                hidden[hid] = hidden_next;
                output[(batch_idx * seq_len + step) * HIDDEN_SIZE + hid] = hidden_next;
            }
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_parallel_update_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    const int k_begin = cta_idx * K_TILE;
    const int h_begin = cta_idx * K_TILE;
    const int h_end = h_begin + K_TILE;

    for (int hid = h_begin + tid; hid < h_end; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;

        // recurrent projection 仍按 k 维切成 4 个 CTA，保持 dot-product 并行度。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_hh[hid * HIDDEN_SIZE + k];
                acc_z += hidden_value * weight_hh[(HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
                acc_n += hidden_value * weight_hh[(2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                partial_base[hid] = acc_r;
                partial_base[HIDDEN_SIZE + hid] = acc_z;
                partial_base[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        // 4 个 CTA 分摊 hidden update，避免所有 sigmoid/tanh 都集中在 CTA0。
        for (int hid = h_begin + tid; hid < h_end; hid += blockDim.x) {
            const float* partial0 = partial_gates + (batch_idx * CTAS_PER_BATCH) * 3 * HIDDEN_SIZE;
            const float* partial1 = partial0 + 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            const float acc_r = bias_hh[hid]
                + partial0[hid]
                + partial1[hid]
                + partial2[hid]
                + partial3[hid];
            const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                + partial0[HIDDEN_SIZE + hid]
                + partial1[HIDDEN_SIZE + hid]
                + partial2[HIDDEN_SIZE + hid]
                + partial3[HIDDEN_SIZE + hid];
            const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                + partial0[2 * HIDDEN_SIZE + hid]
                + partial1[2 * HIDDEN_SIZE + hid]
                + partial2[2 * HIDDEN_SIZE + hid]
                + partial3[2 * HIDDEN_SIZE + hid];

            const float i_r = input_step[hid];
            const float i_z = input_step[HIDDEN_SIZE + hid];
            const float i_n = input_step[2 * HIDDEN_SIZE + hid];
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
            const float new_gate = tanhf(i_n + reset_gate * acc_n);
            const float hidden_prev = hidden[hid];
            const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
            hidden[hid] = hidden_next;
            output[(batch_idx * seq_len + step) * HIDDEN_SIZE + hid] = hidden_next;
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_shmem_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;

        // CTA0 的 partial 留在 shared memory，减少一次全局写读往返。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

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

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_shmem_gate_cache_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // 在 forward 的已有计算路径上顺手保存 backward pointwise 所需值。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

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
void a100_gru_forward_from_gates_cooperative_h256_shmem_grad_coeff_cache_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ partial_gates,
    float* __restrict__ hidden_state,
    float* __restrict__ output,
    float* __restrict__ grad_coeff_cache,
    int seq_len)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;
    constexpr int CACHE_SIZE = 5 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = grad_coeff_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

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

                // 多存 1H 导数系数，减少 backward split 中重复的 pointwise 乘法链。
                const float one_minus_update = 1.0f - update_gate;
                const float new_pre_coeff = one_minus_update * (1.0f - new_gate * new_gate);
                cache_step[hid] = new_pre_coeff
                    * acc_n
                    * reset_gate
                    * (1.0f - reset_gate);
                cache_step[HIDDEN_SIZE + hid] = (hidden_prev - new_gate)
                    * update_gate
                    * (1.0f - update_gate);
                cache_step[2 * HIDDEN_SIZE + hid] = new_pre_coeff;
                cache_step[3 * HIDDEN_SIZE + hid] = new_pre_coeff * reset_gate;
                cache_step[4 * HIDDEN_SIZE + hid] = update_gate;
            }
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_parallel_update_gate_cache_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    const int k_begin = cta_idx * K_TILE;
    const int h_begin = cta_idx * K_TILE;
    const int h_end = h_begin + K_TILE;

    for (int hid = h_begin + tid; hid < h_end; hid += blockDim.x) {
        hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                partial_base[hid] = acc_r;
                partial_base[HIDDEN_SIZE + hid] = acc_z;
                partial_base[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        for (int hid = h_begin + tid; hid < h_end; hid += blockDim.x) {
            const float* partial0 = partial_gates
                + (batch_idx * CTAS_PER_BATCH) * 3 * HIDDEN_SIZE;
            const float* partial1 = partial0 + 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            const float acc_r = bias_hh[hid]
                + partial0[hid]
                + partial1[hid]
                + partial2[hid]
                + partial3[hid];
            const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                + partial0[HIDDEN_SIZE + hid]
                + partial1[HIDDEN_SIZE + hid]
                + partial2[HIDDEN_SIZE + hid]
                + partial3[HIDDEN_SIZE + hid];
            const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                + partial0[2 * HIDDEN_SIZE + hid]
                + partial1[2 * HIDDEN_SIZE + hid]
                + partial2[2 * HIDDEN_SIZE + hid]
                + partial3[2 * HIDDEN_SIZE + hid];

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
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_cta8_shmem_gate_cache_kernel(
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
    constexpr int CTAS_PER_BATCH = 8;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 3;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // 8 CTA 版本提高 h256 forward 的常驻 CTA 数，验证 A100 SM 利用率是否受限。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;
            const float* partial4 = partial3 + 3 * HIDDEN_SIZE;
            const float* partial5 = partial4 + 3 * HIDDEN_SIZE;
            const float* partial6 = partial5 + 3 * HIDDEN_SIZE;
            const float* partial7 = partial6 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid]
                    + partial4[hid]
                    + partial5[hid]
                    + partial6[hid]
                    + partial7[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid]
                    + partial4[HIDDEN_SIZE + hid]
                    + partial5[HIDDEN_SIZE + hid]
                    + partial6[HIDDEN_SIZE + hid]
                    + partial7[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid]
                    + partial4[2 * HIDDEN_SIZE + hid]
                    + partial5[2 * HIDDEN_SIZE + hid]
                    + partial6[2 * HIDDEN_SIZE + hid]
                    + partial7[2 * HIDDEN_SIZE + hid];

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
void a100_gru_forward_from_gates_cooperative_h256_cta6_shmem_gate_cache_kernel(
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
    constexpr int CTAS_PER_BATCH = 6;
    constexpr int K_TILE = 44;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / CTAS_PER_BATCH;
    const int cta_idx = global_cta - batch_idx * CTAS_PER_BATCH;
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;
    const int k_end = min(k_begin + K_TILE, HIDDEN_SIZE);

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // 6 CTA/batch 在 A100 上形成 96 个 resident blocks，比 4 CTA 更接近 SM 数。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                if (k < k_end) {
                    const float hidden_value = hidden[k];
                    acc_r += hidden_value * weight_r[k_local];
                    acc_z += hidden_value * weight_z[k_local];
                    acc_n += hidden_value * weight_n[k_local];
                }
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;
            const float* partial4 = partial3 + 3 * HIDDEN_SIZE;
            const float* partial5 = partial4 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid]
                    + partial4[hid]
                    + partial5[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid]
                    + partial4[HIDDEN_SIZE + hid]
                    + partial5[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid]
                    + partial4[2 * HIDDEN_SIZE + hid]
                    + partial5[2 * HIDDEN_SIZE + hid];

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
void a100_gru_forward_from_gates_cooperative_h256_htile2_shmem_gate_cache_kernel(
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
    constexpr int H_TILES = 2;
    constexpr int H_TILE = HIDDEN_SIZE / H_TILES;
    constexpr int K_CTAS = 4;
    constexpr int K_TILE = HIDDEN_SIZE / K_CTAS;
    constexpr int CTAS_PER_BATCH = H_TILES * K_CTAS;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 3;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int h_tile_idx = cta_idx >> 2;
    const int k_cta_idx = cta_idx & (K_CTAS - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * H_TILE;

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

        // hidden 维分片只分摊输出行，不增加 K 维 partial 路数。
        for (int h_local = group_idx; h_local < H_TILE; h_local += groups_per_block) {
            const int hid = h_begin + h_local;
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (k_cta_idx == 0) ? partial0_local : partial_base;
                partial_out[h_local] = acc_r;
                partial_out[H_TILE + h_local] = acc_z;
                partial_out[2 * H_TILE + h_local] = acc_n;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            const int tile_base_cta = batch_idx * CTAS_PER_BATCH + h_tile_idx * K_CTAS;
            const float* partial1 = partial_gates + (tile_base_cta + 1) * 3 * H_TILE;
            const float* partial2 = partial1 + 3 * H_TILE;
            const float* partial3 = partial2 + 3 * H_TILE;

            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1[h_local]
                    + partial2[h_local]
                    + partial3[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1[H_TILE + h_local]
                    + partial2[H_TILE + h_local]
                    + partial3[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1[2 * H_TILE + h_local]
                    + partial2[2 * H_TILE + h_local]
                    + partial3[2 * H_TILE + h_local];

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
void a100_gru_forward_from_gates_cooperative_h256_htile4_shmem_gate_cache_kernel(
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
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * H_TILE;

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

        // hidden 维进一步分片，观察更高 CTA 并行度是否能覆盖同步开销。
        for (int h_local = group_idx; h_local < H_TILE; h_local += groups_per_block) {
            const int hid = h_begin + h_local;
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (k_cta_idx == 0) ? partial0_local : partial_base;
                partial_out[h_local] = acc_r;
                partial_out[H_TILE + h_local] = acc_z;
                partial_out[2 * H_TILE + h_local] = acc_n;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            const int tile_base_cta = batch_idx * CTAS_PER_BATCH + h_tile_idx * K_CTAS;
            const float* partial1 = partial_gates + (tile_base_cta + 1) * 3 * H_TILE;
            const float* partial2 = partial1 + 3 * H_TILE;
            const float* partial3 = partial2 + 3 * H_TILE;

            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1[h_local]
                    + partial2[h_local]
                    + partial3[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1[H_TILE + h_local]
                    + partial2[H_TILE + h_local]
                    + partial3[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1[2 * H_TILE + h_local]
                    + partial2[2 * H_TILE + h_local]
                    + partial3[2 * H_TILE + h_local];

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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_shmem_gate_cache_kernel(
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
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
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

        // compact partial buffer 只保存 K1/K2/K3，去掉 K0 的全局空洞槽位。
        for (int h_local = group_idx; h_local < H_TILE; h_local += groups_per_block) {
            const int hid = h_begin + h_local;
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = partial0_local;
                if (k_cta_idx != 0) {
                    const int partial_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
                        + h_tile_idx * (K_CTAS - 1)
                        + (k_cta_idx - 1);
                    partial_out = partial_gates + partial_idx * 3 * H_TILE;
                }
                partial_out[h_local] = acc_r;
                partial_out[H_TILE + h_local] = acc_z;
                partial_out[2 * H_TILE + h_local] = acc_n;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            const int partial_base_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
                + h_tile_idx * (K_CTAS - 1);
            const float* partial1 = partial_gates + partial_base_idx * 3 * H_TILE;
            const float* partial2 = partial1 + 3 * H_TILE;
            const float* partial3 = partial2 + 3 * H_TILE;

            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1[h_local]
                    + partial2[h_local]
                    + partial3[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1[H_TILE + h_local]
                    + partial2[H_TILE + h_local]
                    + partial3[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1[2 * H_TILE + h_local]
                    + partial2[2 * H_TILE + h_local]
                    + partial3[2 * H_TILE + h_local];

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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_shmem_gate_cache_kernel(
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
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
    const int partial_base_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
        + h_tile_idx * (K_CTAS - 1);
    float* partial_global = partial0_local;
    if (k_cta_idx != 0) {
        partial_global = partial_gates + (partial_base_idx + k_cta_idx - 1) * PARTIAL_STRIDE;
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

        // hoist 分支把 compact partial 的地址基址移到时间循环外，验证地址计算开销上限。
        for (int h_local = group_idx; h_local < H_TILE; h_local += groups_per_block) {
            const int hid = h_begin + h_local;
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = partial_global;
                if (k_cta_idx == 0) {
                    partial_out = partial0_local;
                }
                partial_out[h_local] = acc_r;
                partial_out[H_TILE + h_local] = acc_z;
                partial_out[2 * H_TILE + h_local] = acc_n;
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

        // row4 分支让每个 half-warp 固定负责 4 行，但按 2 行一组计算来控制寄存器占用。
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

extern "C" __global__ __launch_bounds__(256, 3)
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_prev_cache_shmem_gate_cache_kernel(
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ partial_gates,
    float* __restrict__ hidden_state,
    float* __restrict__ output,
    float* __restrict__ gate_cache,
    float* __restrict__ hidden_prev_cache,
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
    const int batch_size = gridDim.x >> 4;
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
        float* hidden_prev_step = hidden_prev_cache + (step * batch_size + batch_idx) * HIDDEN_SIZE;

        // prev-cache 分支保持 row4 dot 路径，只额外保存 weight_hh 梯度 GEMM 需要的 h_{t-1}。
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
                hidden_prev_step[hid] = hidden_prev;
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

extern "C" __global__ __launch_bounds__(256, 3)
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_parallel_update_gate_cache_kernel(
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
    constexpr int PARTIALS_PER_BATCH = H_TILES * K_CTAS;
    constexpr int PARTIAL_STRIDE = 3 * H_TILE;
    constexpr int GROUPS_PER_BLOCK = 16;
    constexpr int H_UPDATE_TILE = H_TILE / K_CTAS;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

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
    const int partial_base_idx = batch_idx * PARTIALS_PER_BATCH + cta_idx;
    float* partial_out_base = partial_gates + partial_base_idx * PARTIAL_STRIDE;
    const float* partial0_base = partial_gates
        + (batch_idx * PARTIALS_PER_BATCH + h_tile_idx * K_CTAS) * PARTIAL_STRIDE;
    const float* partial1_base = partial0_base + PARTIAL_STRIDE;
    const float* partial2_base = partial1_base + PARTIAL_STRIDE;
    const float* partial3_base = partial2_base + PARTIAL_STRIDE;
    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;

    // 每个 k CTA 初始化自己后续负责 update 的 16 个 hidden，避免只让 k0 做初始化。
    if (tid < H_UPDATE_TILE) {
        const int h_local = k_cta_idx * H_UPDATE_TILE + tid;
        const int hid = h_begin + h_local;
        hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // recurrent dot 仍保持 row4 分配，但 k0 partial 也写入 global 供其它 k CTA 更新使用。
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

        if (tid < H_UPDATE_TILE) {
            const int h_local = k_cta_idx * H_UPDATE_TILE + tid;
            const int hid = h_begin + h_local;
            const float acc_r = bias_hh[hid]
                + partial0_base[h_local]
                + partial1_base[h_local]
                + partial2_base[h_local]
                + partial3_base[h_local];
            const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                + partial0_base[H_TILE + h_local]
                + partial1_base[H_TILE + h_local]
                + partial2_base[H_TILE + h_local]
                + partial3_base[H_TILE + h_local];
            const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                + partial0_base[2 * H_TILE + h_local]
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
        grid.sync();
    }
}

extern "C" __global__ __launch_bounds__(256, 3)
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_ldg_shmem_gate_cache_kernel(
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
            hidden[hid] = __ldg(h0 + batch_idx * HIDDEN_SIZE + hid);
        }
    }
    grid.sync();

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;
        float* cache_step = gate_cache + (batch_idx * seq_len + step) * CACHE_SIZE;

        // row4_ldg 只对本 kernel 内不会被写回的 input/weight/bias 使用只读加载。
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
                acc_r0 += hidden_value * __ldg(weight_r0 + k_local);
                acc_z0 += hidden_value * __ldg(weight_z0 + k_local);
                acc_n0 += hidden_value * __ldg(weight_n0 + k_local);
                acc_r1 += hidden_value * __ldg(weight_r1 + k_local);
                acc_z1 += hidden_value * __ldg(weight_z1 + k_local);
                acc_n1 += hidden_value * __ldg(weight_n1 + k_local);
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
                const float acc_r = __ldg(bias_hh + hid)
                    + partial0_local[h_local]
                    + partial1_base[h_local]
                    + partial2_base[h_local]
                    + partial3_base[h_local];
                const float acc_z = __ldg(bias_hh + HIDDEN_SIZE + hid)
                    + partial0_local[H_TILE + h_local]
                    + partial1_base[H_TILE + h_local]
                    + partial2_base[H_TILE + h_local]
                    + partial3_base[H_TILE + h_local];
                const float acc_n = __ldg(bias_hh + 2 * HIDDEN_SIZE + hid)
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1_base[2 * H_TILE + h_local]
                    + partial2_base[2 * H_TILE + h_local]
                    + partial3_base[2 * H_TILE + h_local];

                const float i_r = __ldg(input_step + hid);
                const float i_z = __ldg(input_step + HIDDEN_SIZE + hid);
                const float i_n = __ldg(input_step + 2 * HIDDEN_SIZE + hid);
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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row3_shmem_gate_cache_kernel(
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

        // 先用同一个 hidden 读服务 3 行，第二轮再处理剩余 1 行，测试 row4 的寄存器边界。
        const int h0_local = group_idx;
        const int h1_local = h0_local + GROUPS_PER_BLOCK;
        const int h2_local = h1_local + GROUPS_PER_BLOCK;
        const int hid0 = h_begin + h0_local;
        const int hid1 = h_begin + h1_local;
        const int hid2 = h_begin + h2_local;

        const float* weight_r0 = weight_hh + hid0 * HIDDEN_SIZE + k_begin;
        const float* weight_z0 = weight_hh + (HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
        const float* weight_n0 = weight_hh + (2 * HIDDEN_SIZE + hid0) * HIDDEN_SIZE + k_begin;
        const float* weight_r1 = weight_hh + hid1 * HIDDEN_SIZE + k_begin;
        const float* weight_z1 = weight_hh + (HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;
        const float* weight_n1 = weight_hh + (2 * HIDDEN_SIZE + hid1) * HIDDEN_SIZE + k_begin;
        const float* weight_r2 = weight_hh + hid2 * HIDDEN_SIZE + k_begin;
        const float* weight_z2 = weight_hh + (HIDDEN_SIZE + hid2) * HIDDEN_SIZE + k_begin;
        const float* weight_n2 = weight_hh + (2 * HIDDEN_SIZE + hid2) * HIDDEN_SIZE + k_begin;

        float acc_r0 = 0.0f;
        float acc_z0 = 0.0f;
        float acc_n0 = 0.0f;
        float acc_r1 = 0.0f;
        float acc_z1 = 0.0f;
        float acc_n1 = 0.0f;
        float acc_r2 = 0.0f;
        float acc_z2 = 0.0f;
        float acc_n2 = 0.0f;

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
            acc_r2 += hidden_value * weight_r2[k_local];
            acc_z2 += hidden_value * weight_z2[k_local];
            acc_n2 += hidden_value * weight_n2[k_local];
        }

        acc_r0 = half_warp_reduce_sum(acc_r0, group_mask);
        acc_z0 = half_warp_reduce_sum(acc_z0, group_mask);
        acc_n0 = half_warp_reduce_sum(acc_n0, group_mask);
        acc_r1 = half_warp_reduce_sum(acc_r1, group_mask);
        acc_z1 = half_warp_reduce_sum(acc_z1, group_mask);
        acc_n1 = half_warp_reduce_sum(acc_n1, group_mask);
        acc_r2 = half_warp_reduce_sum(acc_r2, group_mask);
        acc_z2 = half_warp_reduce_sum(acc_z2, group_mask);
        acc_n2 = half_warp_reduce_sum(acc_n2, group_mask);

        if (lane == 0) {
            partial_out_base[h0_local] = acc_r0;
            partial_out_base[H_TILE + h0_local] = acc_z0;
            partial_out_base[2 * H_TILE + h0_local] = acc_n0;
            partial_out_base[h1_local] = acc_r1;
            partial_out_base[H_TILE + h1_local] = acc_z1;
            partial_out_base[2 * H_TILE + h1_local] = acc_n1;
            partial_out_base[h2_local] = acc_r2;
            partial_out_base[H_TILE + h2_local] = acc_z2;
            partial_out_base[2 * H_TILE + h2_local] = acc_n2;
        }

        const int h3_local = h2_local + GROUPS_PER_BLOCK;
        const int hid3 = h_begin + h3_local;
        const float* weight_r3 = weight_hh + hid3 * HIDDEN_SIZE + k_begin;
        const float* weight_z3 = weight_hh + (HIDDEN_SIZE + hid3) * HIDDEN_SIZE + k_begin;
        const float* weight_n3 = weight_hh + (2 * HIDDEN_SIZE + hid3) * HIDDEN_SIZE + k_begin;

        float acc_r3 = 0.0f;
        float acc_z3 = 0.0f;
        float acc_n3 = 0.0f;

#pragma unroll
        for (int k_local = lane; k_local < K_TILE; k_local += 16) {
            const int k = k_begin + k_local;
            const float hidden_value = hidden[k];
            acc_r3 += hidden_value * weight_r3[k_local];
            acc_z3 += hidden_value * weight_z3[k_local];
            acc_n3 += hidden_value * weight_n3[k_local];
        }

        acc_r3 = half_warp_reduce_sum(acc_r3, group_mask);
        acc_z3 = half_warp_reduce_sum(acc_z3, group_mask);
        acc_n3 = half_warp_reduce_sum(acc_n3, group_mask);

        if (lane == 0) {
            partial_out_base[h3_local] = acc_r3;
            partial_out_base[H_TILE + h3_local] = acc_z3;
            partial_out_base[2 * H_TILE + h3_local] = acc_n3;
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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_qwarp_gate_cache_kernel(
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
    constexpr int GROUPS_PER_BLOCK = 32;
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
    const int lane = tid & 7;
    const int group_idx = tid >> 3;
    const unsigned int group_mask = 0xffu << (warp_lane & 24);

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

        // quarter-warp 分支用 32 个 8-lane group 覆盖 H_TILE，降低单线程寄存器压力。
#pragma unroll
        for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
            const int h_local = group_idx + pair_idx * GROUPS_PER_BLOCK;
            const int hid = h_begin + h_local;

            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;

            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 8) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = quarter_warp_reduce_sum(acc_r, group_mask);
            acc_z = quarter_warp_reduce_sum(acc_z, group_mask);
            acc_n = quarter_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                partial_out_base[h_local] = acc_r;
                partial_out_base[H_TILE + h_local] = acc_z;
                partial_out_base[2 * H_TILE + h_local] = acc_n;
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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_hidden_shmem_gate_cache_kernel(
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
    float* hidden_tile_cache = shared + PARTIAL_STRIDE;

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

        if (tid < K_TILE) {
            hidden_tile_cache[tid] = hidden[k_begin + tid];
        }
        __syncthreads();

        // 每个 block 内缓存 64 个 hidden 值，减少 row4 dot-product 中重复 hidden 读。
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
                const float hidden_value = hidden_tile_cache[k_local];
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
void a100_gru_forward_from_gates_cooperative_h256_htile4_compact_hoist_row4_weight_shmem_gate_cache_kernel(
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
    constexpr int WEIGHT_TILE_STRIDE = H_TILE * K_TILE;
    constexpr int WEIGHT_TILE_SIZE = 3 * WEIGHT_TILE_STRIDE;

    extern __shared__ float shared[];
    float* partial0_local = shared;
    float* weight_tile = shared + PARTIAL_STRIDE;

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

    // recurrent weight 在整个序列内不变，每个 CTA 只负责固定的 hidden tile 和 k tile。
    for (int offset = tid; offset < WEIGHT_TILE_SIZE; offset += blockDim.x) {
        const int gate_idx = offset / WEIGHT_TILE_STRIDE;
        const int tile_offset = offset - gate_idx * WEIGHT_TILE_STRIDE;
        const int h_local = tile_offset / K_TILE;
        const int k_local = tile_offset - h_local * K_TILE;
        const int hid = h_begin + h_local;
        const int gate_row = gate_idx * HIDDEN_SIZE + hid;
        weight_tile[offset] = weight_hh[gate_row * HIDDEN_SIZE + k_begin + k_local];
    }
    __syncthreads();

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

        // row4 分支让每个 half-warp 固定负责 4 行，但按 2 行一组计算来控制寄存器占用。
#pragma unroll
        for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
            const int pair_base = pair_idx * 2 * GROUPS_PER_BLOCK;
            const int h0_local = group_idx + pair_base;
            const int h1_local = h0_local + GROUPS_PER_BLOCK;
            const int hid0 = h_begin + h0_local;
            const int hid1 = h_begin + h1_local;

            const float* weight_r0 = weight_tile + h0_local * K_TILE;
            const float* weight_z0 = weight_tile + WEIGHT_TILE_STRIDE + h0_local * K_TILE;
            const float* weight_n0 = weight_tile + 2 * WEIGHT_TILE_STRIDE + h0_local * K_TILE;
            const float* weight_r1 = weight_tile + h1_local * K_TILE;
            const float* weight_z1 = weight_tile + WEIGHT_TILE_STRIDE + h1_local * K_TILE;
            const float* weight_n1 = weight_tile + 2 * WEIGHT_TILE_STRIDE + h1_local * K_TILE;

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
void a100_gru_forward_from_gates_cooperative_h256_htile8_compact_shmem_gate_cache_kernel(
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
    constexpr int H_TILES = 8;
    constexpr int H_TILE = HIDDEN_SIZE / H_TILES;
    constexpr int K_CTAS = 4;
    constexpr int K_TILE = HIDDEN_SIZE / K_CTAS;
    constexpr int CTAS_PER_BATCH = H_TILES * K_CTAS;
    constexpr int COMPACT_PARTIALS_PER_BATCH = H_TILES * (K_CTAS - 1);
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 5;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int h_tile_idx = cta_idx >> 2;
    const int k_cta_idx = cta_idx & (K_CTAS - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    const int h_begin = h_tile_idx * H_TILE;
    const int k_begin = k_cta_idx * K_TILE;
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

        // htile8 继续提高 hidden 维并行度，用 compact buffer 控制 partial 空洞。
        for (int h_local = group_idx; h_local < H_TILE; h_local += groups_per_block) {
            const int hid = h_begin + h_local;
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = partial0_local;
                if (k_cta_idx != 0) {
                    const int partial_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
                        + h_tile_idx * (K_CTAS - 1)
                        + (k_cta_idx - 1);
                    partial_out = partial_gates + partial_idx * 3 * H_TILE;
                }
                partial_out[h_local] = acc_r;
                partial_out[H_TILE + h_local] = acc_z;
                partial_out[2 * H_TILE + h_local] = acc_n;
            }
        }
        grid.sync();

        if (k_cta_idx == 0) {
            const int partial_base_idx = batch_idx * COMPACT_PARTIALS_PER_BATCH
                + h_tile_idx * (K_CTAS - 1);
            const float* partial1 = partial_gates + partial_base_idx * 3 * H_TILE;
            const float* partial2 = partial1 + 3 * H_TILE;
            const float* partial3 = partial2 + 3 * H_TILE;

            for (int h_local = tid; h_local < H_TILE; h_local += blockDim.x) {
                const int hid = h_begin + h_local;
                const float acc_r = bias_hh[hid]
                    + partial0_local[h_local]
                    + partial1[h_local]
                    + partial2[h_local]
                    + partial3[h_local];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[H_TILE + h_local]
                    + partial1[H_TILE + h_local]
                    + partial2[H_TILE + h_local]
                    + partial3[H_TILE + h_local];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * H_TILE + h_local]
                    + partial1[2 * H_TILE + h_local]
                    + partial2[2 * H_TILE + h_local]
                    + partial3[2 * H_TILE + h_local];

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
void a100_gru_forward_from_gates_cooperative_h256_qwarp_shmem_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;

    extern __shared__ float shared[];
    float* partial0_local = shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 7;
    const int group_idx = tid / 8;
    const int groups_per_block = blockDim.x / 8;
    const unsigned int group_mask = 0xffu << (warp_lane & 24);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;

        // quarter-warp 提高每个 block 同时覆盖的 hidden 列数。
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 8) {
                const int k = k_begin + k_local;
                const float hidden_value = hidden[k];
                acc_r += hidden_value * weight_hh[hid * HIDDEN_SIZE + k];
                acc_z += hidden_value * weight_hh[(HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
                acc_n += hidden_value * weight_hh[(2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k];
            }

            acc_r = quarter_warp_reduce_sum(acc_r, group_mask);
            acc_z = quarter_warp_reduce_sum(acc_z, group_mask);
            acc_n = quarter_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float* partial1 = partial_gates
                    + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
                const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
                const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

                const float i_r = input_step[hid];
                const float i_z = input_step[HIDDEN_SIZE + hid];
                const float i_n = input_step[2 * HIDDEN_SIZE + hid];
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r)));
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z)));
                const float new_gate = tanhf(i_n + reset_gate * acc_n);
                const float hidden_prev = hidden[hid];
                const float hidden_next = new_gate + update_gate * (hidden_prev - new_gate);
                hidden[hid] = hidden_next;
                output[(batch_idx * seq_len + step) * HIDDEN_SIZE + hid] = hidden_next;
            }
        }
        grid.sync();
    }
}

extern "C" __global__
void a100_gru_forward_from_gates_cooperative_h256_cached_shmem_kernel(
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
    constexpr int CTAS_PER_BATCH = 4;
    constexpr int K_TILE = HIDDEN_SIZE / CTAS_PER_BATCH;

    extern __shared__ float shared[];
    float* partial0_local = shared;
    float* hidden_tile = (blockIdx.x & (CTAS_PER_BATCH - 1)) == 0
        ? shared + 3 * HIDDEN_SIZE
        : shared;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int cta_idx = global_cta & (CTAS_PER_BATCH - 1);
    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane = tid & 15;
    const int group_idx = tid / 16;
    const int groups_per_block = blockDim.x / 16;
    const unsigned int group_mask = 0xffffu << (warp_lane & 16);

    float* hidden = hidden_state + batch_idx * HIDDEN_SIZE;
    float* partial_base = partial_gates + global_cta * 3 * HIDDEN_SIZE;

    if (cta_idx == 0) {
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
            hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid];
        }
    }
    grid.sync();

    const int k_begin = cta_idx * K_TILE;

    for (int step = 0; step < seq_len; ++step) {
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE;
        float* output_step = output + (batch_idx * seq_len + step) * HIDDEN_SIZE;

        // 每个 CTA 只需要 64 维 hidden tile，先缓存到 shared memory 复用。
        for (int k_local = tid; k_local < K_TILE; k_local += blockDim.x) {
            hidden_tile[k_local] = hidden[k_begin + k_local];
        }
        __syncthreads();

        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) {
            const float* weight_r = weight_hh + hid * HIDDEN_SIZE + k_begin;
            const float* weight_z = weight_hh + (HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            const float* weight_n = weight_hh + (2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k_begin;
            float acc_r = 0.0f;
            float acc_z = 0.0f;
            float acc_n = 0.0f;

#pragma unroll
            for (int k_local = lane; k_local < K_TILE; k_local += 16) {
                const float hidden_value = hidden_tile[k_local];
                acc_r += hidden_value * weight_r[k_local];
                acc_z += hidden_value * weight_z[k_local];
                acc_n += hidden_value * weight_n[k_local];
            }

            acc_r = half_warp_reduce_sum(acc_r, group_mask);
            acc_z = half_warp_reduce_sum(acc_z, group_mask);
            acc_n = half_warp_reduce_sum(acc_n, group_mask);

            if (lane == 0) {
                float* partial_out = (cta_idx == 0) ? partial0_local : partial_base;
                partial_out[hid] = acc_r;
                partial_out[HIDDEN_SIZE + hid] = acc_z;
                partial_out[2 * HIDDEN_SIZE + hid] = acc_n;
            }
        }
        grid.sync();

        if (cta_idx == 0) {
            const float* partial1 = partial_gates
                + (batch_idx * CTAS_PER_BATCH + 1) * 3 * HIDDEN_SIZE;
            const float* partial2 = partial1 + 3 * HIDDEN_SIZE;
            const float* partial3 = partial2 + 3 * HIDDEN_SIZE;

            for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) {
                const float acc_r = bias_hh[hid]
                    + partial0_local[hid]
                    + partial1[hid]
                    + partial2[hid]
                    + partial3[hid];
                const float acc_z = bias_hh[HIDDEN_SIZE + hid]
                    + partial0_local[HIDDEN_SIZE + hid]
                    + partial1[HIDDEN_SIZE + hid]
                    + partial2[HIDDEN_SIZE + hid]
                    + partial3[HIDDEN_SIZE + hid];
                const float acc_n = bias_hh[2 * HIDDEN_SIZE + hid]
                    + partial0_local[2 * HIDDEN_SIZE + hid]
                    + partial1[2 * HIDDEN_SIZE + hid]
                    + partial2[2 * HIDDEN_SIZE + hid]
                    + partial3[2 * HIDDEN_SIZE + hid];

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

extern "C" __global__
void a100_gru_h256_pointwise_backward_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ grad_hidden_prev_direct,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * HIDDEN_SIZE;
    if (idx >= total) {
        return;
    }

    const int batch_idx = idx / HIDDEN_SIZE;
    const int hid = idx - batch_idx * HIDDEN_SIZE;
    const int input_base = (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE + hid;
    const int gates_base = batch_idx * 3 * HIDDEN_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    const float i_r = input_gates[input_base];
    const float i_z = input_gates[input_base + HIDDEN_SIZE];
    const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
    const float h_r = hidden_gates[gates_base];
    const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
    const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    const float h_prev = (step == 0)
        ? h0[hidden_base]
        : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
    const float grad_out = grad_hidden_next[hidden_base];

    const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
    const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
    const float new_gate = tanhf(i_n + reset_gate * h_n);

    // gate 顺序与 PyTorch GRU 保持一致：r, z, n。
    const float grad_update = grad_out * (h_prev - new_gate);
    const float grad_new = grad_out * (1.0f - update_gate);
    const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
    const float grad_reset = grad_new_pre * h_n;
    const float grad_recurrent_n = grad_new_pre * reset_gate;
    const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
    const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);

    grad_input_gates[input_base] = grad_reset_pre;
    grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
    grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
    grad_hidden_gates[gates_base] = grad_reset_pre;
    grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
    grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    grad_hidden_prev_direct[hidden_base] = grad_out * update_gate;
}

extern "C" __global__
void a100_gru_h256_recurrent_backward_kernel(
    const float* __restrict__ grad_hidden_gates,
    const float* __restrict__ weight_hh,
    const float* __restrict__ grad_hidden_prev_direct,
    float* __restrict__ grad_hidden_prev,
    int batch_size)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;

    extern __shared__ float shared_grad_gates[];

    const int batch_idx = blockIdx.x;
    const int hid = threadIdx.x;
    if (batch_idx >= batch_size || hid >= HIDDEN_SIZE) {
        return;
    }

    const float* grad_gates_base = grad_hidden_gates + batch_idx * GATES_SIZE;
    for (int gate_idx = hid; gate_idx < GATES_SIZE; gate_idx += HIDDEN_SIZE) {
        shared_grad_gates[gate_idx] = grad_gates_base[gate_idx];
    }
    __syncthreads();

    float acc = grad_hidden_prev_direct[batch_idx * HIDDEN_SIZE + hid];

#pragma unroll 3
    for (int gate_block = 0; gate_block < GATES_SIZE; gate_block += HIDDEN_SIZE) {
#pragma unroll 4
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            const int gate_idx = gate_block + k;
            acc += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
    }

    grad_hidden_prev[batch_idx * HIDDEN_SIZE + hid] = acc;
}

extern "C" __global__
void a100_gru_h256_recurrent_backward_tiled_kernel(
    const float* __restrict__ grad_hidden_gates,
    const float* __restrict__ weight_hh,
    const float* __restrict__ grad_hidden_prev_direct,
    float* __restrict__ grad_hidden_prev,
    int batch_size)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int TILE_BATCH = 16;
    constexpr int TILE_HIDDEN = 16;
    constexpr int TILE_K = 32;

    extern __shared__ float shared[];
    float* shared_gates = shared;
    float* shared_weight = shared + TILE_BATCH * TILE_K;

    const int hidden_col = blockIdx.x * TILE_HIDDEN + threadIdx.x;
    const int batch_idx = blockIdx.y * TILE_BATCH + threadIdx.y;
    const int linear_tid = threadIdx.y * TILE_HIDDEN + threadIdx.x;

    float acc = 0.0f;
    if (batch_idx < batch_size && hidden_col < HIDDEN_SIZE) {
        acc = grad_hidden_prev_direct[batch_idx * HIDDEN_SIZE + hidden_col];
    }

    for (int k_base = 0; k_base < GATES_SIZE; k_base += TILE_K) {
        for (int offset = linear_tid; offset < TILE_BATCH * TILE_K; offset += TILE_BATCH * TILE_HIDDEN) {
            const int batch_row = offset / TILE_K;
            const int k_inner = offset - batch_row * TILE_K;
            const int load_batch = blockIdx.y * TILE_BATCH + batch_row;
            const int gate_idx = k_base + k_inner;
            shared_gates[offset] = (load_batch < batch_size)
                ? grad_hidden_gates[load_batch * GATES_SIZE + gate_idx]
                : 0.0f;
        }
        for (int offset = linear_tid; offset < TILE_K * TILE_HIDDEN; offset += TILE_BATCH * TILE_HIDDEN) {
            const int k_inner = offset / TILE_HIDDEN;
            const int hidden_inner = offset - k_inner * TILE_HIDDEN;
            const int gate_idx = k_base + k_inner;
            const int load_hidden = blockIdx.x * TILE_HIDDEN + hidden_inner;
            shared_weight[offset] = weight_hh[gate_idx * HIDDEN_SIZE + load_hidden];
        }
        __syncthreads();

        if (batch_idx < batch_size && hidden_col < HIDDEN_SIZE) {
#pragma unroll
            for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
                acc += shared_gates[threadIdx.y * TILE_K + k_inner]
                    * shared_weight[k_inner * TILE_HIDDEN + threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (batch_idx < batch_size && hidden_col < HIDDEN_SIZE) {
        grad_hidden_prev[batch_idx * HIDDEN_SIZE + hidden_col] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_recurrent_backward_split_kernel(
    const float* __restrict__ grad_hidden_gates,
    const float* __restrict__ weight_hh,
    float* __restrict__ partial_sums,
    int batch_size,
    int split_count)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int TILE_BATCH = 16;
    constexpr int TILE_HIDDEN = 16;
    constexpr int TILE_K = 32;

    extern __shared__ float shared[];
    float* shared_gates = shared;
    float* shared_weight = shared + TILE_BATCH * TILE_K;

    const int split_idx = blockIdx.z;
    const int split_size = (GATES_SIZE + split_count - 1) / split_count;
    const int split_begin = split_idx * split_size;
    const int split_end = min(split_begin + split_size, GATES_SIZE);
    const int hidden_col = blockIdx.x * TILE_HIDDEN + threadIdx.x;
    const int batch_idx = blockIdx.y * TILE_BATCH + threadIdx.y;
    const int linear_tid = threadIdx.y * TILE_HIDDEN + threadIdx.x;

    float acc = 0.0f;
    for (int k_base = split_begin; k_base < split_end; k_base += TILE_K) {
        const int k_limit = min(k_base + TILE_K, split_end);
        const int current_tile_k = k_limit - k_base;

        for (int offset = linear_tid; offset < TILE_BATCH * TILE_K; offset += TILE_BATCH * TILE_HIDDEN) {
            const int batch_row = offset / TILE_K;
            const int k_inner = offset - batch_row * TILE_K;
            const int load_batch = blockIdx.y * TILE_BATCH + batch_row;
            const int gate_idx = k_base + k_inner;
            shared_gates[offset] = (load_batch < batch_size && k_inner < current_tile_k)
                ? grad_hidden_gates[load_batch * GATES_SIZE + gate_idx]
                : 0.0f;
        }
        for (int offset = linear_tid; offset < TILE_K * TILE_HIDDEN; offset += TILE_BATCH * TILE_HIDDEN) {
            const int k_inner = offset / TILE_HIDDEN;
            const int hidden_inner = offset - k_inner * TILE_HIDDEN;
            const int gate_idx = k_base + k_inner;
            const int load_hidden = blockIdx.x * TILE_HIDDEN + hidden_inner;
            shared_weight[offset] = (k_inner < current_tile_k)
                ? weight_hh[gate_idx * HIDDEN_SIZE + load_hidden]
                : 0.0f;
        }
        __syncthreads();

        if (batch_idx < batch_size && hidden_col < HIDDEN_SIZE) {
#pragma unroll
            for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
                acc += shared_gates[threadIdx.y * TILE_K + k_inner]
                    * shared_weight[k_inner * TILE_HIDDEN + threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (batch_idx < batch_size && hidden_col < HIDDEN_SIZE) {
        partial_sums[(split_idx * batch_size + batch_idx) * HIDDEN_SIZE + hidden_col] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_recurrent_backward_split_reduce_kernel(
    const float* __restrict__ partial_sums,
    const float* __restrict__ grad_hidden_prev_direct,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int split_count)
{
    constexpr int HIDDEN_SIZE = 256;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * HIDDEN_SIZE;
    if (idx >= total) {
        return;
    }

    const int batch_idx = idx / HIDDEN_SIZE;
    const int hid = idx - batch_idx * HIDDEN_SIZE;
    float acc = grad_hidden_prev_direct[idx];
#pragma unroll
    for (int split_idx = 0; split_idx < 8; ++split_idx) {
        if (split_idx < split_count) {
            acc += partial_sums[(split_idx * batch_size + batch_idx) * HIDDEN_SIZE + hid];
        }
    }
    grad_hidden_prev[idx] = acc;
}

extern "C" __global__
void a100_gru_h256_backward_step_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;

    extern __shared__ float shared_grad_gates[];

    const int batch_idx = blockIdx.x;
    const int hid = threadIdx.x;
    if (batch_idx >= batch_size || hid >= HIDDEN_SIZE) {
        return;
    }

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    const float i_r = input_gates[input_base];
    const float i_z = input_gates[input_base + HIDDEN_SIZE];
    const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
    const float h_r = hidden_gates[gates_base];
    const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
    const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    const float h_prev = (step == 0)
        ? h0[hidden_base]
        : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
    const float grad_out = grad_hidden_next[hidden_base];

    const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
    const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
    const float new_gate = tanhf(i_n + reset_gate * h_n);

    // 先在每个 hidden lane 中完成 GRU gate 的逐元素反向。
    const float grad_update = grad_out * (h_prev - new_gate);
    const float grad_new = grad_out * (1.0f - update_gate);
    const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
    const float grad_reset = grad_new_pre * h_n;
    const float grad_recurrent_n = grad_new_pre * reset_gate;
    const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
    const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
    const float grad_hidden_prev_direct = grad_out * update_gate;

    shared_grad_gates[hid] = grad_reset_pre;
    shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
    shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    grad_input_gates[input_base] = grad_reset_pre;
    grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
    grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
    grad_hidden_gates[gates_base] = grad_reset_pre;
    grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
    grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    __syncthreads();

    float acc = grad_hidden_prev_direct;
#pragma unroll 3
    for (int gate_block = 0; gate_block < GATES_SIZE; gate_block += HIDDEN_SIZE) {
#pragma unroll 4
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            const int gate_idx = gate_block + k;
            acc += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
    }

    grad_hidden_prev[hidden_base] = acc;
}

extern "C" __global__
void a100_gru_h256_backward_step_cooperative_split_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step,
    int split_count)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / split_count;
    const int split_idx = global_cta - batch_idx * split_count;
    const int hid = threadIdx.x;

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    const float i_r = input_gates[input_base];
    const float i_z = input_gates[input_base + HIDDEN_SIZE];
    const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
    const float h_r = hidden_gates[gates_base];
    const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
    const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    const float h_prev = (step == 0)
        ? h0[hidden_base]
        : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
    const float grad_out = grad_hidden_next[hidden_base];

    const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
    const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
    const float new_gate = tanhf(i_n + reset_gate * h_n);

    // 每个 split CTA 都重算 pointwise 梯度，避免额外的跨 CTA 读依赖。
    const float grad_update = grad_out * (h_prev - new_gate);
    const float grad_new = grad_out * (1.0f - update_gate);
    const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
    const float grad_reset = grad_new_pre * h_n;
    const float grad_recurrent_n = grad_new_pre * reset_gate;
    const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
    const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
    const float grad_hidden_prev_direct = grad_out * update_gate;

    shared_grad_gates[hid] = grad_reset_pre;
    shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
    shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    if (split_idx == 0) {
        grad_input_gates[input_base] = grad_reset_pre;
        grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
        grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
        grad_hidden_gates[gates_base] = grad_reset_pre;
        grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
        grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    }
    __syncthreads();

    const int split_size = (GATES_SIZE + split_count - 1) / split_count;
    const int split_begin = split_idx * split_size;
    const int split_end = min(split_begin + split_size, GATES_SIZE);

    float partial = 0.0f;
#pragma unroll 4
    for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
        partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    partial_sums[(batch_idx * split_count + split_idx) * HIDDEN_SIZE + hid] = partial;
    grid.sync();

    if (split_idx == 0) {
        float acc = grad_hidden_prev_direct;
#pragma unroll
        for (int reduce_idx = 0; reduce_idx < 8; ++reduce_idx) {
            if (reduce_idx < split_count) {
                acc += partial_sums[(batch_idx * split_count + reduce_idx) * HIDDEN_SIZE + hid];
            }
        }
        grad_hidden_prev[hidden_base] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_backward_step_cooperative_split2_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int hid = threadIdx.x;

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    const float i_r = input_gates[input_base];
    const float i_z = input_gates[input_base + HIDDEN_SIZE];
    const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
    const float h_r = hidden_gates[gates_base];
    const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
    const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    const float h_prev = (step == 0)
        ? h0[hidden_base]
        : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
    const float grad_out = grad_hidden_next[hidden_base];

    const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
    const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
    const float new_gate = tanhf(i_n + reset_gate * h_n);

    // split2 专用版本：保留重复 pointwise，避免额外 grid sync。
    const float grad_update = grad_out * (h_prev - new_gate);
    const float grad_new = grad_out * (1.0f - update_gate);
    const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
    const float grad_reset = grad_new_pre * h_n;
    const float grad_recurrent_n = grad_new_pre * reset_gate;
    const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
    const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
    const float grad_hidden_prev_direct = grad_out * update_gate;

    shared_grad_gates[hid] = grad_reset_pre;
    shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
    shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    if (split_idx == 0) {
        grad_input_gates[input_base] = grad_reset_pre;
        grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
        grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
        grad_hidden_gates[gates_base] = grad_reset_pre;
        grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
        grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    }
    __syncthreads();

    const int split_begin = split_idx * SPLIT_SIZE;
    const int split_end = split_begin + SPLIT_SIZE;

    float partial = 0.0f;
#pragma unroll 4
    for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
        partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    partial_sums[(batch_idx * SPLIT_COUNT + split_idx) * HIDDEN_SIZE + hid] = partial;
    grid.sync();

    if (split_idx == 0) {
        const float acc = grad_hidden_prev_direct
            + partial_sums[batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid]
            + partial_sums[(batch_idx * SPLIT_COUNT + 1) * HIDDEN_SIZE + hid];
        grad_hidden_prev[hidden_base] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_backward_step_cooperative_split_cached_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step,
    int split_count)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / split_count;
    const int split_idx = global_cta - batch_idx * split_count;
    const int hid = threadIdx.x;

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    float grad_hidden_prev_direct = 0.0f;
    if (split_idx == 0) {
        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates[gates_base];
        const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
        const float grad_out = grad_hidden_next[hidden_base];

        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);

        // 只在 split0 计算一次 pointwise backward，后续 split CTA 复用全局 gate 梯度。
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        grad_hidden_prev_direct = grad_out * update_gate;

        grad_input_gates[input_base] = grad_reset_pre;
        grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
        grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
        grad_hidden_gates[gates_base] = grad_reset_pre;
        grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
        grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    }
    grid.sync();

    shared_grad_gates[hid] = grad_hidden_gates[gates_base];
    shared_grad_gates[hid + HIDDEN_SIZE] = grad_hidden_gates[gates_base + HIDDEN_SIZE];
    shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    __syncthreads();

    const int split_size = (GATES_SIZE + split_count - 1) / split_count;
    const int split_begin = split_idx * split_size;
    const int split_end = min(split_begin + split_size, GATES_SIZE);

    float partial = 0.0f;
#pragma unroll 4
    for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
        partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    partial_sums[(batch_idx * split_count + split_idx) * HIDDEN_SIZE + hid] = partial;
    grid.sync();

    if (split_idx == 0) {
        float acc = grad_hidden_prev_direct;
#pragma unroll
        for (int reduce_idx = 0; reduce_idx < 8; ++reduce_idx) {
            if (reduce_idx < split_count) {
                acc += partial_sums[(batch_idx * split_count + reduce_idx) * HIDDEN_SIZE + hid];
            }
        }
        grad_hidden_prev[hidden_base] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_backward_step_cooperative_split2_cached_local_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int hid = threadIdx.x;

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    float grad_hidden_prev_direct = 0.0f;
    if (split_idx == 0) {
        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates[gates_base];
        const float h_z = hidden_gates[gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates[gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
        const float grad_out = grad_hidden_next[hidden_base];

        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);

        // split0 保留 gate 梯度在本 CTA shared 中，避免自己再从全局读回。
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        grad_input_gates[input_base] = grad_reset_pre;
        grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
        grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
        grad_hidden_gates[gates_base] = grad_reset_pre;
        grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
        grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    }
    grid.sync();

    if (split_idx != 0) {
        shared_grad_gates[hid] = grad_hidden_gates[gates_base];
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_hidden_gates[gates_base + HIDDEN_SIZE];
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    }
    __syncthreads();

    const int split_begin = split_idx * SPLIT_SIZE;
    const int split_end = split_begin + SPLIT_SIZE;

    float partial = 0.0f;
#pragma unroll 4
    for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
        partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    partial_sums[(batch_idx * SPLIT_COUNT + split_idx) * HIDDEN_SIZE + hid] = partial;
    grid.sync();

    if (split_idx == 0) {
        const float acc = grad_hidden_prev_direct
            + partial_sums[batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid]
            + partial_sums[(batch_idx * SPLIT_COUNT + 1) * HIDDEN_SIZE + hid];
        grad_hidden_prev[hidden_base] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_backward_step_cooperative_split2_gate_cache_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ gate_cache,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ partial_sums,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int hid = threadIdx.x;

    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    float grad_hidden_prev_direct = 0.0f;
    if (split_idx == 0) {
        const float reset_gate = gate_cache[cache_base];
        const float update_gate = gate_cache[cache_base + HIDDEN_SIZE];
        const float new_gate = gate_cache[cache_base + 2 * HIDDEN_SIZE];
        const float recurrent_new = gate_cache[cache_base + 3 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
        const float grad_out = grad_hidden_next[hidden_base];

        // gate_cache 让 backward 避免重新计算 sigmoid/tanh 和读取 hidden_gates。
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * recurrent_new;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        grad_input_gates[input_base] = grad_reset_pre;
        grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
        grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
        grad_hidden_gates[gates_base] = grad_reset_pre;
        grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
        grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
    }
    grid.sync();

    if (split_idx != 0) {
        shared_grad_gates[hid] = grad_hidden_gates[gates_base];
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_hidden_gates[gates_base + HIDDEN_SIZE];
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE];
    }
    __syncthreads();

    const int split_begin = split_idx * SPLIT_SIZE;
    const int split_end = split_begin + SPLIT_SIZE;

    float partial = 0.0f;
#pragma unroll 4
    for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
        partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
    }
    partial_sums[(batch_idx * SPLIT_COUNT + split_idx) * HIDDEN_SIZE + hid] = partial;
    grid.sync();

    if (split_idx == 0) {
        const float acc = grad_hidden_prev_direct
            + partial_sums[batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid]
            + partial_sums[(batch_idx * SPLIT_COUNT + 1) * HIDDEN_SIZE + hid];
        grad_hidden_prev[hidden_base] = acc;
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split2_cached_local_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        float grad_hidden_prev_direct = 0.0f;
        if (split_idx == 0) {
            const float i_r = input_gates[input_base];
            const float i_z = input_gates[input_base + HIDDEN_SIZE];
            const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
            const float h_r = hidden_gates_steps[step_gates_base];
            const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
            const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
            const float h_prev = (step == 0)
                ? h0[hidden_base]
                : output[output_base - HIDDEN_SIZE];
            const float grad_out = grad_hidden_state[hidden_base] + grad_output[output_base];

            // persistent 版本把 time loop 放入一个 kernel，减少每步 launch 间隙。
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
            const float new_gate = tanhf(i_n + reset_gate * h_n);
            const float grad_update = grad_out * (h_prev - new_gate);
            const float grad_new = grad_out * (1.0f - update_gate);
            const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
            const float grad_reset = grad_new_pre * h_n;
            const float grad_recurrent_n = grad_new_pre * reset_gate;
            const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
            const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
            grad_hidden_prev_direct = grad_out * update_gate;

            shared_grad_gates[hid] = grad_reset_pre;
            shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
            shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        grid.sync();

        if (split_idx != 0) {
            shared_grad_gates[hid] = grad_hidden_gates_steps[step_gates_base];
            shared_grad_gates[hid + HIDDEN_SIZE] = (
                grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE]
            );
            shared_grad_gates[hid + 2 * HIDDEN_SIZE] = (
                grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE]
            );
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = split_begin + SPLIT_SIZE;

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        partial_sums[(batch_idx * SPLIT_COUNT + split_idx) * HIDDEN_SIZE + hid] = partial;
        grid.sync();

        if (split_idx == 0) {
            grad_hidden_state[hidden_base] = grad_hidden_prev_direct
                + partial_sums[batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid]
                + partial_sums[(batch_idx * SPLIT_COUNT + 1) * HIDDEN_SIZE + hid];
        }
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split2_state_parts_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // 两个 split 都重算 pointwise，换掉 cached-local 路径中的第一次 grid.sync。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = split_begin + SPLIT_SIZE;

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split2_state_local_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 2;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 1;
    const int split_idx = global_cta & 1;
    const int other_split_idx = split_idx ^ 1;
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;
    float local_state_part = 0.0f;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = local_state_part
                + partial_sums[partial_base + other_split_idx * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // 每个 split 在寄存器里保留自己的 state partial，下一步只读对侧 partial。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = split_begin + SPLIT_SIZE;

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        local_state_part = partial;
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = local_state_part;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = local_state_part + partial_sums[partial_base + HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split4_state_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 4;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 2;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // split4 继续保持每步一次 grid.sync，用更多 CTA 降低 recurrent dot-product 长度。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = split_begin + SPLIT_SIZE;

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split8_state_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 8;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 3;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                + partial_sums[partial_base + 7 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // split8 用更多 CTA 继续缩短 recurrent dot-product，验证并行度上限。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = split_begin + SPLIT_SIZE;

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_state_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = (GATES_SIZE + SPLIT_COUNT - 1) / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                + partial_sums[partial_base + 15 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // split16 验证更多 CTA 是否还能抵消重复 pointwise 和 partial 规约。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = min(split_begin + SPLIT_SIZE, GATES_SIZE);

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_kernel(
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = (GATES_SIZE + SPLIT_COUNT - 1) / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                + partial_sums[partial_base + 15 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float reset_gate = gate_cache[cache_base];
        const float update_gate = gate_cache[cache_base + HIDDEN_SIZE];
        const float new_gate = gate_cache[cache_base + 2 * HIDDEN_SIZE];
        const float recurrent_new = gate_cache[cache_base + 3 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // 复用 forward 保存的 gate activation，避免 backward 重算 hidden gates 和 exp/tanh。
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * recurrent_new;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = min(split_begin + SPLIT_SIZE, GATES_SIZE);

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_kernel(
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

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
                grad_hidden_acc = partial_sums[partial_base]
                    + partial_sums[partial_base + HIDDEN_SIZE]
                    + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 15 * HIDDEN_SIZE];
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

            // split0 负责完整写出 gate 梯度，供后续大 GEMM 计算 weight_hh 梯度。
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
            const int gate_idx = split_idx * SPLIT_SIZE + hid;
            const int gate_type = gate_idx / HIDDEN_SIZE;
            const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
            const int gate_hidden_base = batch_idx * HIDDEN_SIZE + gate_hid;
            const int gate_partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;
            const int gate_output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + gate_hid;

            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[gate_hidden_base];
            } else {
                grad_hidden_acc = partial_sums[gate_partial_base]
                    + partial_sums[gate_partial_base + HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 15 * HIDDEN_SIZE];
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

            // 其它 split 只计算本 split recurrent dot-product 需要的 48 个 gate 梯度。
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

        const int split_begin = split_idx * SPLIT_SIZE;
        float partial = 0.0f;
#pragma unroll 4
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            const int gate_idx = split_begin + local_idx;
            partial += shared_grad_gates[local_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split8_gate_cache_state_tiled_kernel(
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
    constexpr int SPLIT_COUNT = 8;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 3;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

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
                grad_hidden_acc = partial_sums[partial_base]
                    + partial_sums[partial_base + HIDDEN_SIZE]
                    + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 7 * HIDDEN_SIZE];
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

            // split0 负责完整写出 gate 梯度，供后续大 GEMM 计算 weight_hh 梯度。
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
            const int gate_idx = split_idx * SPLIT_SIZE + hid;
            const int gate_type = gate_idx / HIDDEN_SIZE;
            const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
            const int gate_hidden_base = batch_idx * HIDDEN_SIZE + gate_hid;
            const int gate_partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;
            const int gate_output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + gate_hid;

            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[gate_hidden_base];
            } else {
                grad_hidden_acc = partial_sums[gate_partial_base]
                    + partial_sums[gate_partial_base + HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 7 * HIDDEN_SIZE];
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

            // split8 每个 split 处理 96 个 gate，减少 partial 数和跨 split 规约开销。
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

        const int split_begin = split_idx * SPLIT_SIZE;
        float partial = 0.0f;
#pragma unroll 4
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            const int gate_idx = split_begin + local_idx;
            partial += shared_grad_gates[local_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_kernel(
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_storage[];
    float* shared_grad_gates = shared_storage;
    float* shared_weight_tile = shared_storage + SPLIT_SIZE;
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int split_begin = split_idx * SPLIT_SIZE;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    // weight_hh 在整条序列上不变，先把本 split 的 48x256 tile 放到 shared memory 复用。
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
                grad_hidden_acc = partial_sums[partial_base]
                    + partial_sums[partial_base + HIDDEN_SIZE]
                    + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 15 * HIDDEN_SIZE];
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

            // split0 负责完整写出 gate 梯度，供后续大 GEMM 计算 weight_hh 梯度。
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
                grad_hidden_acc = partial_sums[gate_partial_base]
                    + partial_sums[gate_partial_base + HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 15 * HIDDEN_SIZE];
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

            // 其它 split 只计算本 split recurrent dot-product 需要的 48 个 gate 梯度。
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
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

__device__ __forceinline__ float a100_gru_h256_sum_split16_partials(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 0; split < 16; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
}

__device__ __forceinline__ float a100_gru_h256_sum_split16_partials_from1(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 1; split < 16; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
}

__device__ __forceinline__ float a100_gru_h256_sum_split16_partials_except(
    const float* __restrict__ partial_sums,
    int partial_base,
    int skip_split)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 0; split < 16; ++split) {
        if (split != skip_split) {
            total += partial_sums[partial_base + split * HIDDEN_SIZE];
        }
    }
    return total;
}

__device__ __forceinline__ float a100_gru_h256_sum_split12_partials_from1(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 1; split < 12; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
}

__device__ __forceinline__ float a100_gru_h256_sum_split12_partials(
    const float* __restrict__ partial_sums,
    int partial_base)
{
    constexpr int HIDDEN_SIZE = 256;
    float total = 0.0f;
#pragma unroll
    for (int split = 0; split < 12; ++split) {
        total += partial_sums[partial_base + split * HIDDEN_SIZE];
    }
    return total;
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
void a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_kernel(
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_storage[];
    float* shared_grad_gates = shared_storage;
    float* shared_weight_tile = shared_storage + SPLIT_SIZE;
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int split_begin = split_idx * SPLIT_SIZE;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    // split0 的自身 partial 用寄存器跨 step 复用，避免它下一步再读 partial_sums[0]。
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
                    + a100_gru_h256_sum_split16_partials_from1(partial_sums, partial_base);
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
                    a100_gru_h256_sum_split16_partials(partial_sums, gate_partial_base);
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
            + a100_gru_h256_sum_split16_partials_from1(partial_sums, partial_base);
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split5_gate_cache_state_tiled_weight_shmem_split0_keep_kernel(
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
    constexpr int SPLIT_COUNT = 5;
    constexpr int SPLIT_SIZE = (GATES_SIZE + SPLIT_COUNT - 1) / SPLIT_COUNT;

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

    // split5 用约 158.3KB dynamic shared memory 继续减少 partial 路数，最后 split 有 2 个 padding 行。
    float split0_partial_state = 0.0f;

#pragma unroll 4
    for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
        const int gate_idx = split_begin + local_idx;
        float weight_value = 0.0f;
        if (gate_idx < GATES_SIZE) {
            weight_value = weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        shared_weight_tile[local_idx * HIDDEN_SIZE + hid] = weight_value;
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
            float local_grad = 0.0f;
            if (gate_idx < GATES_SIZE) {
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

                local_grad = grad_reset_pre;
                if (gate_type == 1) {
                    local_grad = grad_update_pre;
                } else if (gate_type == 2) {
                    local_grad = grad_recurrent_n;
                }
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

    // split6 用约 131.6KB dynamic shared memory 换更少 partial 路数。
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

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_kernel(
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
    constexpr int SPLIT_COUNT = 12;
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

    // split12 减少 partial 路数；split0 仍用寄存器保存自身 partial。
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
                    + a100_gru_h256_sum_split12_partials_from1(partial_sums, partial_base);
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
                    a100_gru_h256_sum_split12_partials(partial_sums, gate_partial_base);
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
            + a100_gru_h256_sum_split12_partials_from1(partial_sums, partial_base);
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split12_gate_cache_state_tiled_weight_shmem_split0_keep_unroll8_kernel(
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
    constexpr int SPLIT_COUNT = 12;
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

    // 仅改变 split12 shared-memory dot loop 的展开因子，用于验证调度/寄存器折中。
    float split0_partial_state = 0.0f;

#pragma unroll 8
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
                    + a100_gru_h256_sum_split12_partials_from1(partial_sums, partial_base);
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
                    a100_gru_h256_sum_split12_partials(partial_sums, gate_partial_base);
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
#pragma unroll 8
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
            + a100_gru_h256_sum_split12_partials_from1(partial_sums, partial_base);
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split24_gate_cache_state_tiled_weight_shmem_split0_keep_kernel(
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
    constexpr int SPLIT_COUNT = 24;
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

    // split24 观察更小 shared tile 与更多 cooperative blocks 是否能抵消 partial 路数增加。
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

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_gate_cache_state_tiled_weight_shmem_split0_keep_own_shmem_kernel(
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_storage[];
    float* shared_grad_gates = shared_storage;
    float* shared_own_partials = shared_storage + SPLIT_SIZE;
    float* shared_weight_tile = shared_storage + SPLIT_SIZE + HIDDEN_SIZE;
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int split_begin = split_idx * SPLIT_SIZE;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    // split0 仍用寄存器保存自身 partial；其它 split 用 shared 保存本 block 上一轮 partial。
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
                    + a100_gru_h256_sum_split16_partials_from1(partial_sums, partial_base);
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
                grad_hidden_acc = shared_own_partials[gate_hid]
                    + a100_gru_h256_sum_split16_partials_except(
                        partial_sums,
                        gate_partial_base,
                        split_idx);
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
        shared_own_partials[hid] = partial;
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = split0_partial_state
            + a100_gru_h256_sum_split16_partials_from1(partial_sums, partial_base);
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_grad_coeff_cache_state_tiled_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ grad_coeff_cache,
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
    constexpr int CACHE_SIZE = 5 * HIDDEN_SIZE;
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

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
                grad_hidden_acc = partial_sums[partial_base]
                    + partial_sums[partial_base + HIDDEN_SIZE]
                    + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[partial_base + 15 * HIDDEN_SIZE];
            }

            const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + hid;
            const float grad_out = grad_hidden_acc + grad_output[output_base];
            const float grad_reset_pre = grad_out * grad_coeff_cache[cache_base];
            const float grad_update_pre = grad_out
                * grad_coeff_cache[cache_base + HIDDEN_SIZE];
            const float grad_new_pre = grad_out
                * grad_coeff_cache[cache_base + 2 * HIDDEN_SIZE];
            const float grad_recurrent_n = grad_out
                * grad_coeff_cache[cache_base + 3 * HIDDEN_SIZE];
            grad_hidden_prev_direct = grad_out
                * grad_coeff_cache[cache_base + 4 * HIDDEN_SIZE];

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
            const int gate_idx = split_idx * SPLIT_SIZE + hid;
            const int gate_type = gate_idx / HIDDEN_SIZE;
            const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
            const int gate_hidden_base = batch_idx * HIDDEN_SIZE + gate_hid;
            const int gate_partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;
            const int gate_output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + gate_hid;

            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[gate_hidden_base];
            } else {
                grad_hidden_acc = partial_sums[gate_partial_base]
                    + partial_sums[gate_partial_base + HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 2 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 3 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 4 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 5 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 6 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 7 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 8 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 9 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 10 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 11 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 12 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 13 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 14 * HIDDEN_SIZE]
                    + partial_sums[gate_partial_base + 15 * HIDDEN_SIZE];
            }

            const int cache_base = (batch_idx * seq_len + step) * CACHE_SIZE + gate_hid;
            const float grad_out = grad_hidden_acc + grad_output[gate_output_base];
            float local_grad = grad_out * grad_coeff_cache[cache_base];
            if (gate_type == 1) {
                local_grad = grad_out * grad_coeff_cache[cache_base + HIDDEN_SIZE];
            } else if (gate_type == 2) {
                local_grad = grad_out * grad_coeff_cache[cache_base + 3 * HIDDEN_SIZE];
            }
            shared_grad_gates[hid] = local_grad;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        float partial = 0.0f;
#pragma unroll 4
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            const int gate_idx = split_begin + local_idx;
            partial += shared_grad_gates[local_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split32_gate_cache_state_tiled_kernel(
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
    constexpr int SPLIT_COUNT = 32;
    constexpr int SPLIT_SIZE = GATES_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 5;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

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
#pragma unroll
                for (int split = 0; split < SPLIT_COUNT; ++split) {
                    grad_hidden_acc += partial_sums[partial_base + split * HIDDEN_SIZE];
                }
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
            const int gate_idx = split_idx * SPLIT_SIZE + hid;
            const int gate_type = gate_idx / HIDDEN_SIZE;
            const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
            const int gate_hidden_base = batch_idx * HIDDEN_SIZE + gate_hid;
            const int gate_partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;
            const int gate_output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + gate_hid;

            float grad_hidden_acc = 0.0f;
            if (step == seq_len - 1) {
                grad_hidden_acc = grad_hidden_state[gate_hidden_base];
            } else {
#pragma unroll
                for (int split = 0; split < SPLIT_COUNT; ++split) {
                    grad_hidden_acc += partial_sums[gate_partial_base + split * HIDDEN_SIZE];
                }
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

        const int split_begin = split_idx * SPLIT_SIZE;
        float partial = 0.0f;
#pragma unroll 4
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            const int gate_idx = split_begin + local_idx;
            partial += shared_grad_gates[local_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        float grad_hidden = 0.0f;
#pragma unroll
        for (int split = 0; split < SPLIT_COUNT; ++split) {
            grad_hidden += partial_sums[partial_base + split * HIDDEN_SIZE];
        }
        grad_hidden_state[hidden_base] = grad_hidden;
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split16_state_global_gates_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 16;
    constexpr int SPLIT_SIZE = (GATES_SIZE + SPLIT_COUNT - 1) / SPLIT_COUNT;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 4;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                + partial_sums[partial_base + 15 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_start = (step * batch_size + batch_idx) * GATES_SIZE;
        const int step_gates_base = step_gates_start + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;
        float grad_hidden_prev_direct = 0.0f;

        if (split_idx == 0) {
            const float i_r = input_gates[input_base];
            const float i_z = input_gates[input_base + HIDDEN_SIZE];
            const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
            const float h_r = hidden_gates_steps[step_gates_base];
            const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
            const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
            const float h_prev = (step == 0)
                ? h0[hidden_base]
                : output[output_base - HIDDEN_SIZE];
            const float grad_out = grad_hidden_acc + grad_output[output_base];

            // 只在 split0 计算 pointwise，并把 gate 梯度作为全局缓存给其他 split 复用。
            const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
            const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
            const float new_gate = tanhf(i_n + reset_gate * h_n);
            const float grad_update = grad_out * (h_prev - new_gate);
            const float grad_new = grad_out * (1.0f - update_gate);
            const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
            const float grad_reset = grad_new_pre * h_n;
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
        }
        grid.sync();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = min(split_begin + SPLIT_SIZE, GATES_SIZE);

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += grad_hidden_gates_steps[step_gates_start + gate_idx]
                * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_sequence_cooperative_split32_state_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input_gates,
    const float* __restrict__ hidden_gates_steps,
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
    constexpr int SPLIT_COUNT = 32;
    constexpr int SPLIT_SIZE = (GATES_SIZE + SPLIT_COUNT - 1) / SPLIT_COUNT;

    extern __shared__ float shared_grad_gates[];
    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta >> 5;
    const int split_idx = global_cta & (SPLIT_COUNT - 1);
    const int hid = threadIdx.x;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;
    const int partial_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    for (int step = seq_len - 1; step >= 0; --step) {
        float grad_hidden_acc = 0.0f;
        if (step == seq_len - 1) {
            grad_hidden_acc = grad_hidden_state[hidden_base];
        } else {
            grad_hidden_acc = partial_sums[partial_base]
                + partial_sums[partial_base + HIDDEN_SIZE]
                + partial_sums[partial_base + 2 * HIDDEN_SIZE]
                + partial_sums[partial_base + 3 * HIDDEN_SIZE]
                + partial_sums[partial_base + 4 * HIDDEN_SIZE]
                + partial_sums[partial_base + 5 * HIDDEN_SIZE]
                + partial_sums[partial_base + 6 * HIDDEN_SIZE]
                + partial_sums[partial_base + 7 * HIDDEN_SIZE]
                + partial_sums[partial_base + 8 * HIDDEN_SIZE]
                + partial_sums[partial_base + 9 * HIDDEN_SIZE]
                + partial_sums[partial_base + 10 * HIDDEN_SIZE]
                + partial_sums[partial_base + 11 * HIDDEN_SIZE]
                + partial_sums[partial_base + 12 * HIDDEN_SIZE]
                + partial_sums[partial_base + 13 * HIDDEN_SIZE]
                + partial_sums[partial_base + 14 * HIDDEN_SIZE]
                + partial_sums[partial_base + 15 * HIDDEN_SIZE]
                + partial_sums[partial_base + 16 * HIDDEN_SIZE]
                + partial_sums[partial_base + 17 * HIDDEN_SIZE]
                + partial_sums[partial_base + 18 * HIDDEN_SIZE]
                + partial_sums[partial_base + 19 * HIDDEN_SIZE]
                + partial_sums[partial_base + 20 * HIDDEN_SIZE]
                + partial_sums[partial_base + 21 * HIDDEN_SIZE]
                + partial_sums[partial_base + 22 * HIDDEN_SIZE]
                + partial_sums[partial_base + 23 * HIDDEN_SIZE]
                + partial_sums[partial_base + 24 * HIDDEN_SIZE]
                + partial_sums[partial_base + 25 * HIDDEN_SIZE]
                + partial_sums[partial_base + 26 * HIDDEN_SIZE]
                + partial_sums[partial_base + 27 * HIDDEN_SIZE]
                + partial_sums[partial_base + 28 * HIDDEN_SIZE]
                + partial_sums[partial_base + 29 * HIDDEN_SIZE]
                + partial_sums[partial_base + 30 * HIDDEN_SIZE]
                + partial_sums[partial_base + 31 * HIDDEN_SIZE];
        }

        const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
        const int step_gates_base = (step * batch_size + batch_idx) * GATES_SIZE + hid;
        const int output_base = (batch_idx * seq_len + step) * HIDDEN_SIZE + hid;

        const float i_r = input_gates[input_base];
        const float i_z = input_gates[input_base + HIDDEN_SIZE];
        const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
        const float h_r = hidden_gates_steps[step_gates_base];
        const float h_z = hidden_gates_steps[step_gates_base + HIDDEN_SIZE];
        const float h_n = hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE];
        const float h_prev = (step == 0)
            ? h0[hidden_base]
            : output[output_base - HIDDEN_SIZE];
        const float grad_out = grad_hidden_acc + grad_output[output_base];

        // split32 进一步扫描并行度上限，预期可能被重复 pointwise 和规约抵消。
        const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
        const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
        const float new_gate = tanhf(i_n + reset_gate * h_n);
        const float grad_update = grad_out * (h_prev - new_gate);
        const float grad_new = grad_out * (1.0f - update_gate);
        const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
        const float grad_reset = grad_new_pre * h_n;
        const float grad_recurrent_n = grad_new_pre * reset_gate;
        const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
        const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
        const float grad_hidden_prev_direct = grad_out * update_gate;

        shared_grad_gates[hid] = grad_reset_pre;
        shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
        shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

        if (split_idx == 0) {
            grad_input_gates[input_base] = grad_reset_pre;
            grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
            grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
            grad_hidden_gates_steps[step_gates_base] = grad_reset_pre;
            grad_hidden_gates_steps[step_gates_base + HIDDEN_SIZE] = grad_update_pre;
            grad_hidden_gates_steps[step_gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;
        }
        __syncthreads();

        const int split_begin = split_idx * SPLIT_SIZE;
        const int split_end = min(split_begin + SPLIT_SIZE, GATES_SIZE);

        float partial = 0.0f;
#pragma unroll 4
        for (int gate_idx = split_begin; gate_idx < split_end; ++gate_idx) {
            partial += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        if (split_idx == 0) {
            partial += grad_hidden_prev_direct;
        }
        partial_sums[partial_base + split_idx * HIDDEN_SIZE] = partial;
        grid.sync();
    }

    if (split_idx == 0) {
        grad_hidden_state[hidden_base] = partial_sums[partial_base]
            + partial_sums[partial_base + HIDDEN_SIZE]
            + partial_sums[partial_base + 2 * HIDDEN_SIZE]
            + partial_sums[partial_base + 3 * HIDDEN_SIZE]
            + partial_sums[partial_base + 4 * HIDDEN_SIZE]
            + partial_sums[partial_base + 5 * HIDDEN_SIZE]
            + partial_sums[partial_base + 6 * HIDDEN_SIZE]
            + partial_sums[partial_base + 7 * HIDDEN_SIZE]
            + partial_sums[partial_base + 8 * HIDDEN_SIZE]
            + partial_sums[partial_base + 9 * HIDDEN_SIZE]
            + partial_sums[partial_base + 10 * HIDDEN_SIZE]
            + partial_sums[partial_base + 11 * HIDDEN_SIZE]
            + partial_sums[partial_base + 12 * HIDDEN_SIZE]
            + partial_sums[partial_base + 13 * HIDDEN_SIZE]
            + partial_sums[partial_base + 14 * HIDDEN_SIZE]
            + partial_sums[partial_base + 15 * HIDDEN_SIZE]
            + partial_sums[partial_base + 16 * HIDDEN_SIZE]
            + partial_sums[partial_base + 17 * HIDDEN_SIZE]
            + partial_sums[partial_base + 18 * HIDDEN_SIZE]
            + partial_sums[partial_base + 19 * HIDDEN_SIZE]
            + partial_sums[partial_base + 20 * HIDDEN_SIZE]
            + partial_sums[partial_base + 21 * HIDDEN_SIZE]
            + partial_sums[partial_base + 22 * HIDDEN_SIZE]
            + partial_sums[partial_base + 23 * HIDDEN_SIZE]
            + partial_sums[partial_base + 24 * HIDDEN_SIZE]
            + partial_sums[partial_base + 25 * HIDDEN_SIZE]
            + partial_sums[partial_base + 26 * HIDDEN_SIZE]
            + partial_sums[partial_base + 27 * HIDDEN_SIZE]
            + partial_sums[partial_base + 28 * HIDDEN_SIZE]
            + partial_sums[partial_base + 29 * HIDDEN_SIZE]
            + partial_sums[partial_base + 30 * HIDDEN_SIZE]
            + partial_sums[partial_base + 31 * HIDDEN_SIZE];
    }
}

extern "C" __global__
void a100_gru_h256_backward_step_recompute_kernel(
    const float* __restrict__ grad_hidden_next,
    const float* __restrict__ input_gates,
    const float* __restrict__ h0,
    const float* __restrict__ output,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_hh,
    float* __restrict__ grad_input_gates,
    float* __restrict__ grad_hidden_gates,
    float* __restrict__ grad_hidden_prev,
    int batch_size,
    int seq_len,
    int step)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATES_SIZE = 3 * HIDDEN_SIZE;

    extern __shared__ float shared[];
    float* shared_hidden = shared;
    float* shared_grad_gates = shared + HIDDEN_SIZE;

    const int batch_idx = blockIdx.x;
    const int hid = threadIdx.x;
    if (batch_idx >= batch_size || hid >= HIDDEN_SIZE) {
        return;
    }

    const int input_base = (batch_idx * seq_len + step) * GATES_SIZE + hid;
    const int gates_base = batch_idx * GATES_SIZE + hid;
    const int hidden_base = batch_idx * HIDDEN_SIZE + hid;

    const float h_prev = (step == 0)
        ? h0[hidden_base]
        : output[(batch_idx * seq_len + step - 1) * HIDDEN_SIZE + hid];
    shared_hidden[hid] = h_prev;
    __syncthreads();

    float h_r = bias_hh[hid];
    float h_z = bias_hh[hid + HIDDEN_SIZE];
    float h_n = bias_hh[hid + 2 * HIDDEN_SIZE];
#pragma unroll 4
    for (int k = 0; k < HIDDEN_SIZE; ++k) {
        const float hidden_value = shared_hidden[k];
        h_r += hidden_value * weight_hh[hid * HIDDEN_SIZE + k];
        h_z += hidden_value * weight_hh[(hid + HIDDEN_SIZE) * HIDDEN_SIZE + k];
        h_n += hidden_value * weight_hh[(hid + 2 * HIDDEN_SIZE) * HIDDEN_SIZE + k];
    }

    const float i_r = input_gates[input_base];
    const float i_z = input_gates[input_base + HIDDEN_SIZE];
    const float i_n = input_gates[input_base + 2 * HIDDEN_SIZE];
    const float grad_out = grad_hidden_next[hidden_base];

    const float reset_gate = 1.0f / (1.0f + expf(-(i_r + h_r)));
    const float update_gate = 1.0f / (1.0f + expf(-(i_z + h_z)));
    const float new_gate = tanhf(i_n + reset_gate * h_n);

    // 显存节省实验：hidden gates 在本 kernel 内重算，不再保存整条序列。
    const float grad_update = grad_out * (h_prev - new_gate);
    const float grad_new = grad_out * (1.0f - update_gate);
    const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
    const float grad_reset = grad_new_pre * h_n;
    const float grad_recurrent_n = grad_new_pre * reset_gate;
    const float grad_reset_pre = grad_reset * reset_gate * (1.0f - reset_gate);
    const float grad_update_pre = grad_update * update_gate * (1.0f - update_gate);
    const float grad_hidden_prev_direct = grad_out * update_gate;

    shared_grad_gates[hid] = grad_reset_pre;
    shared_grad_gates[hid + HIDDEN_SIZE] = grad_update_pre;
    shared_grad_gates[hid + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    grad_input_gates[input_base] = grad_reset_pre;
    grad_input_gates[input_base + HIDDEN_SIZE] = grad_update_pre;
    grad_input_gates[input_base + 2 * HIDDEN_SIZE] = grad_new_pre;
    grad_hidden_gates[gates_base] = grad_reset_pre;
    grad_hidden_gates[gates_base + HIDDEN_SIZE] = grad_update_pre;
    grad_hidden_gates[gates_base + 2 * HIDDEN_SIZE] = grad_recurrent_n;

    __syncthreads();

    float acc = grad_hidden_prev_direct;
#pragma unroll 3
    for (int gate_block = 0; gate_block < GATES_SIZE; gate_block += HIDDEN_SIZE) {
#pragma unroll 4
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            const int gate_idx = gate_block + k;
            acc += shared_grad_gates[gate_idx] * weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
    }

    grad_hidden_prev[hidden_base] = acc;
}

// 固定 hidden_size 版本用于断崖点附近的 A100 专用化实验。
#define DEFINE_A100_GRU_FUSED_SPECIALIZED_KERNEL(KERNEL_NAME, HIDDEN_SIZE, UNROLL_PRAGMA) \
extern "C" __global__ \
void KERNEL_NAME( \
    const float* __restrict__ input_gates, \
    const float* __restrict__ h0, \
    const float* __restrict__ weight_hh, \
    const float* __restrict__ bias_hh, \
    float* __restrict__ output, \
    int seq_len, \
    int hidden_size) \
{ \
    (void)hidden_size; \
    extern __shared__ float shared[]; \
    float* hidden = shared; \
    float* next_hidden = shared + HIDDEN_SIZE; \
 \
    const int batch_idx = blockIdx.x; \
    const int tid = threadIdx.x; \
    const int warp_lane = tid & (WARP_SIZE - 1); \
    const int lane = tid & 15; \
    const int group_idx = tid / 16; \
    const int groups_per_block = blockDim.x / 16; \
    const unsigned int group_mask = 0xffffu << (warp_lane & 16); \
 \
    for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) { \
        hidden[hid] = h0[batch_idx * HIDDEN_SIZE + hid]; \
    } \
    __syncthreads(); \
 \
    for (int step = 0; step < seq_len; ++step) { \
        const float* input_step = input_gates + (batch_idx * seq_len + step) * 3 * HIDDEN_SIZE; \
 \
        for (int hid = group_idx; hid < HIDDEN_SIZE; hid += groups_per_block) { \
            float acc_r = (lane == 0) ? bias_hh[hid] : 0.0f; \
            float acc_z = (lane == 0) ? bias_hh[HIDDEN_SIZE + hid] : 0.0f; \
            float acc_n = (lane == 0) ? bias_hh[2 * HIDDEN_SIZE + hid] : 0.0f; \
 \
            UNROLL_PRAGMA \
            for (int k = lane; k < HIDDEN_SIZE; k += 16) { \
                const float hidden_value = hidden[k]; \
                acc_r += hidden_value * weight_hh[hid * HIDDEN_SIZE + k]; \
                acc_z += hidden_value * weight_hh[(HIDDEN_SIZE + hid) * HIDDEN_SIZE + k]; \
                acc_n += hidden_value * weight_hh[(2 * HIDDEN_SIZE + hid) * HIDDEN_SIZE + k]; \
            } \
 \
            acc_r = half_warp_reduce_sum(acc_r, group_mask); \
            acc_z = half_warp_reduce_sum(acc_z, group_mask); \
            acc_n = half_warp_reduce_sum(acc_n, group_mask); \
 \
            if (lane == 0) { \
                const float i_r = input_step[hid]; \
                const float i_z = input_step[HIDDEN_SIZE + hid]; \
                const float i_n = input_step[2 * HIDDEN_SIZE + hid]; \
                const float reset_gate = 1.0f / (1.0f + expf(-(i_r + acc_r))); \
                const float update_gate = 1.0f / (1.0f + expf(-(i_z + acc_z))); \
                const float new_gate = tanhf(i_n + reset_gate * acc_n); \
                const float hidden_prev = hidden[hid]; \
                next_hidden[hid] = new_gate + update_gate * (hidden_prev - new_gate); \
            } \
        } \
        __syncthreads(); \
 \
        for (int hid = tid; hid < HIDDEN_SIZE; hid += blockDim.x) { \
            const float hidden_next = next_hidden[hid]; \
            hidden[hid] = hidden_next; \
            output[(batch_idx * seq_len + step) * HIDDEN_SIZE + hid] = hidden_next; \
        } \
        __syncthreads(); \
    } \
}

DEFINE_A100_GRU_FUSED_SPECIALIZED_KERNEL(
    a100_gru_forward_from_gates_fused_specialized_h128_kernel,
    128,
    )
DEFINE_A100_GRU_FUSED_SPECIALIZED_KERNEL(
    a100_gru_forward_from_gates_fused_specialized_h130_kernel,
    130,
    _Pragma("unroll 1"))
DEFINE_A100_GRU_FUSED_SPECIALIZED_KERNEL(
    a100_gru_forward_from_gates_fused_specialized_h160_kernel,
    160,
    )

#undef DEFINE_A100_GRU_FUSED_SPECIALIZED_KERNEL
