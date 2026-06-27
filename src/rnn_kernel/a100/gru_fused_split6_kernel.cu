#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" __global__ __launch_bounds__(256, 1)
void a100_gru_h256_stacked_backward_split6_weight_shmem_kernel(
    const float* __restrict__ grad_output_top,
    const float* __restrict__ grad_h_n,
    const float* __restrict__ x,
    const float* __restrict__ h0,
    const float* __restrict__ weight_ih_l0,
    const float* __restrict__ weight_hh_l0,
    const float* __restrict__ weight_ih_l1,
    const float* __restrict__ weight_hh_l1,
    const float* __restrict__ weight_ih_l2,
    const float* __restrict__ weight_hh_l2,
    const float* __restrict__ weight_ih_l3,
    const float* __restrict__ weight_hh_l3,
    const float* __restrict__ all_outputs,
    const float* __restrict__ gate_cache_all,
    float* __restrict__ recurrent_partial_sums,
    float* __restrict__ input_partial_sums,
    float* __restrict__ grad_layer_outputs,
    float* __restrict__ grad_x,
    float* __restrict__ grad_h0,
    float* __restrict__ grad_input_gates_all,
    float* __restrict__ grad_hidden_gates_all,
    int batch_size,
    int seq_len,
    int input_size,
    int num_layers)
{
    constexpr int HIDDEN_SIZE = 256;
    constexpr int GATE_SIZE = 3 * HIDDEN_SIZE;
    constexpr int CACHE_SIZE = 4 * HIDDEN_SIZE;
    constexpr int MAX_LAYERS = 4;
    constexpr int SPLIT_COUNT = 6;
    constexpr int SPLIT_SIZE = GATE_SIZE / SPLIT_COUNT;

    extern __shared__ float shared_storage[];
    float* shared_hidden_grad_gates = shared_storage;
    float* shared_input_grad_gates = shared_hidden_grad_gates + SPLIT_SIZE;
    float* shared_weight_hh_tile = shared_input_grad_gates + SPLIT_SIZE;

    cg::grid_group grid = cg::this_grid();

    const int global_cta = blockIdx.x;
    const int batch_idx = global_cta / SPLIT_COUNT;
    const int split_idx = global_cta - batch_idx * SPLIT_COUNT;
    const int hid = threadIdx.x;
    const int split_begin = split_idx * SPLIT_SIZE;
    const int partial_hid_base = batch_idx * SPLIT_COUNT * HIDDEN_SIZE + hid;

    if (batch_idx >= batch_size || num_layers < 1 || num_layers > MAX_LAYERS) {
        return;
    }

    const float* weight_ih_layers[MAX_LAYERS] = {
        weight_ih_l0,
        weight_ih_l1,
        weight_ih_l2,
        weight_ih_l3,
    };
    const float* weight_hh_layers[MAX_LAYERS] = {
        weight_hh_l0,
        weight_hh_l1,
        weight_hh_l2,
        weight_hh_l3,
    };

    for (int layer = num_layers - 1; layer >= 0; --layer) {
        const float* weight_ih = weight_ih_layers[layer];
        const float* weight_hh = weight_hh_layers[layer];
        float split0_partial_state = 0.0f;

        // 每个 split 缓存 128x256 的 recurrent weight tile，沿 time step 复用。
#pragma unroll 8
        for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
            const int gate_idx = split_begin + local_idx;
            shared_weight_hh_tile[local_idx * HIDDEN_SIZE + hid] =
                weight_hh[gate_idx * HIDDEN_SIZE + hid];
        }
        __syncthreads();

        for (int step = seq_len - 1; step >= 0; --step) {
            float grad_hidden_prev_direct = 0.0f;
            const int output_offset =
                ((layer * batch_size + batch_idx) * seq_len + step) * HIDDEN_SIZE;
            const int gate_offset =
                ((layer * batch_size + batch_idx) * seq_len + step) * GATE_SIZE;
            const int cache_offset =
                ((layer * batch_size + batch_idx) * seq_len + step) * CACHE_SIZE;
            const int batch_step_hidden_offset =
                (batch_idx * seq_len + step) * HIDDEN_SIZE;
            const int batch_step_input_offset =
                (batch_idx * seq_len + step) * input_size;
            const int h0_base = (layer * batch_size + batch_idx) * HIDDEN_SIZE;

            if (split_idx == 0) {
                float grad_hidden_acc = 0.0f;
                if (step == seq_len - 1) {
                    grad_hidden_acc = grad_h_n[h0_base + hid];
                } else {
                    grad_hidden_acc = split0_partial_state;
#pragma unroll
                    for (int partial_idx = 1; partial_idx < SPLIT_COUNT; ++partial_idx) {
                        grad_hidden_acc +=
                            recurrent_partial_sums[partial_hid_base + partial_idx * HIDDEN_SIZE];
                    }
                }

                const float direct_grad = (layer == num_layers - 1)
                    ? grad_output_top[batch_step_hidden_offset + hid]
                    : grad_layer_outputs[output_offset + hid];
                const float grad_hidden_next = grad_hidden_acc + direct_grad;
                const float hidden_prev = (step == 0)
                    ? h0[h0_base + hid]
                    : all_outputs[output_offset - HIDDEN_SIZE + hid];
                const float reset_gate = gate_cache_all[cache_offset + hid];
                const float update_gate = gate_cache_all[cache_offset + HIDDEN_SIZE + hid];
                const float new_gate = gate_cache_all[cache_offset + 2 * HIDDEN_SIZE + hid];
                const float hidden_gate_n = gate_cache_all[cache_offset + 3 * HIDDEN_SIZE + hid];

                const float grad_new = grad_hidden_next * (1.0f - update_gate);
                const float grad_update = grad_hidden_next * (hidden_prev - new_gate);
                const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
                const float grad_hidden_n = grad_new_pre * reset_gate;
                const float grad_reset = grad_new_pre * hidden_gate_n;
                const float grad_input_r = grad_reset * reset_gate * (1.0f - reset_gate);
                const float grad_input_z = grad_update * update_gate * (1.0f - update_gate);
                grad_hidden_prev_direct = grad_hidden_next * update_gate;

                grad_input_gates_all[gate_offset + hid] = grad_input_r;
                grad_input_gates_all[gate_offset + HIDDEN_SIZE + hid] = grad_input_z;
                grad_input_gates_all[gate_offset + 2 * HIDDEN_SIZE + hid] = grad_new_pre;
                grad_hidden_gates_all[gate_offset + hid] = grad_input_r;
                grad_hidden_gates_all[gate_offset + HIDDEN_SIZE + hid] = grad_input_z;
                grad_hidden_gates_all[gate_offset + 2 * HIDDEN_SIZE + hid] = grad_hidden_n;

                if (hid < SPLIT_SIZE) {
                    shared_hidden_grad_gates[hid] = grad_input_r;
                    shared_input_grad_gates[hid] = grad_input_r;
                }
            } else if (hid < SPLIT_SIZE) {
                const int gate_idx = split_begin + hid;
                const int gate_type = gate_idx / HIDDEN_SIZE;
                const int gate_hid = gate_idx - gate_type * HIDDEN_SIZE;
                const int gate_output_offset =
                    ((layer * batch_size + batch_idx) * seq_len + step) * HIDDEN_SIZE
                    + gate_hid;
                const int gate_cache_offset =
                    ((layer * batch_size + batch_idx) * seq_len + step) * CACHE_SIZE
                    + gate_hid;
                const int gate_h0_base = (layer * batch_size + batch_idx) * HIDDEN_SIZE;
                const int gate_partial_base =
                    batch_idx * SPLIT_COUNT * HIDDEN_SIZE + gate_hid;

                float grad_hidden_acc = 0.0f;
                if (step == seq_len - 1) {
                    grad_hidden_acc = grad_h_n[gate_h0_base + gate_hid];
                } else {
#pragma unroll
                    for (int partial_idx = 0; partial_idx < SPLIT_COUNT; ++partial_idx) {
                        grad_hidden_acc +=
                            recurrent_partial_sums[gate_partial_base + partial_idx * HIDDEN_SIZE];
                    }
                }

                const float direct_grad = (layer == num_layers - 1)
                    ? grad_output_top[batch_step_hidden_offset + gate_hid]
                    : grad_layer_outputs[gate_output_offset];
                const float grad_hidden_next = grad_hidden_acc + direct_grad;
                const float hidden_prev = (step == 0)
                    ? h0[gate_h0_base + gate_hid]
                    : all_outputs[gate_output_offset - HIDDEN_SIZE];
                const float reset_gate = gate_cache_all[gate_cache_offset];
                const float update_gate = gate_cache_all[gate_cache_offset + HIDDEN_SIZE];
                const float new_gate = gate_cache_all[gate_cache_offset + 2 * HIDDEN_SIZE];
                const float hidden_gate_n = gate_cache_all[gate_cache_offset + 3 * HIDDEN_SIZE];

                const float grad_new = grad_hidden_next * (1.0f - update_gate);
                const float grad_update = grad_hidden_next * (hidden_prev - new_gate);
                const float grad_new_pre = grad_new * (1.0f - new_gate * new_gate);
                const float grad_hidden_n = grad_new_pre * reset_gate;
                const float grad_reset = grad_new_pre * hidden_gate_n;
                const float grad_input_r = grad_reset * reset_gate * (1.0f - reset_gate);
                const float grad_input_z = grad_update * update_gate * (1.0f - update_gate);

                float local_hidden_grad = grad_input_r;
                float local_input_grad = grad_input_r;
                if (gate_type == 1) {
                    local_hidden_grad = grad_input_z;
                    local_input_grad = grad_input_z;
                } else if (gate_type == 2) {
                    local_hidden_grad = grad_hidden_n;
                    local_input_grad = grad_new_pre;
                }
                shared_hidden_grad_gates[hid] = local_hidden_grad;
                shared_input_grad_gates[hid] = local_input_grad;
            }
            __syncthreads();

            float recurrent_partial = 0.0f;
            float input_partial = 0.0f;
#pragma unroll 8
            for (int local_idx = 0; local_idx < SPLIT_SIZE; ++local_idx) {
                const int gate_idx = split_begin + local_idx;
                recurrent_partial += shared_hidden_grad_gates[local_idx]
                    * shared_weight_hh_tile[local_idx * HIDDEN_SIZE + hid];
                if (layer != 0) {
                    input_partial += shared_input_grad_gates[local_idx]
                        * weight_ih[gate_idx * HIDDEN_SIZE + hid];
                } else if (hid < input_size) {
                    input_partial += shared_input_grad_gates[local_idx]
                        * weight_ih[gate_idx * input_size + hid];
                }
            }

            if (split_idx == 0) {
                recurrent_partial += grad_hidden_prev_direct;
                split0_partial_state = recurrent_partial;
            }
            recurrent_partial_sums[partial_hid_base + split_idx * HIDDEN_SIZE] =
                recurrent_partial;
            const int input_partial_base =
                ((step & 1) * batch_size * SPLIT_COUNT + batch_idx * SPLIT_COUNT + split_idx)
                * HIDDEN_SIZE
                + hid;
            input_partial_sums[input_partial_base] = input_partial;
            grid.sync();

            if (split_idx == 0) {
                float input_grad = input_partial;
#pragma unroll
                for (int partial_idx = 1; partial_idx < SPLIT_COUNT; ++partial_idx) {
                    const int peer_base =
                        ((step & 1) * batch_size * SPLIT_COUNT
                            + batch_idx * SPLIT_COUNT
                            + partial_idx)
                        * HIDDEN_SIZE
                        + hid;
                    input_grad += input_partial_sums[peer_base];
                }
                if (layer == 0) {
                    if (hid < input_size) {
                        grad_x[batch_step_input_offset + hid] = input_grad;
                    }
                } else {
                    const int prev_output_offset =
                        (((layer - 1) * batch_size + batch_idx) * seq_len + step) * HIDDEN_SIZE
                        + hid;
                    grad_layer_outputs[prev_output_offset] = input_grad;
                }
            }
        }

        if (split_idx == 0) {
            float grad_hidden0 = split0_partial_state;
#pragma unroll
            for (int partial_idx = 1; partial_idx < SPLIT_COUNT; ++partial_idx) {
                grad_hidden0 +=
                    recurrent_partial_sums[partial_hid_base + partial_idx * HIDDEN_SIZE];
            }
            grad_h0[(layer * batch_size + batch_idx) * HIDDEN_SIZE + hid] = grad_hidden0;
        }
        grid.sync();
    }
}
