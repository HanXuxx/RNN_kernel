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
