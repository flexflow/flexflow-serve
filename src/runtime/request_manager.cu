/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flashinfer/decode_attention_decl.cuh"
#include "flashinfer/prefill_attention_decl.cuh"

#include "flexflow/request_manager.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

using namespace Legion;

using flashinfer::BatchDecodeHandler;
using flashinfer::BatchPrefillHandler;
using flashinfer::LogitsPostHook;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

void RequestManager::load_tokens_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  BatchConfig::TokenId dram_copy[BatchConfig::MAX_NUM_TOKENS];

  // Extreme long prompts are not supported, only load up to
  // BatchConfig::max_tokens_per_batch() as prompt
  if (batch_config->num_tokens > BatchConfig::max_tokens_per_batch() &&
      batch_config->get_mode() == INC_DECODING_MODE) {
    printf("Warning: too many tokens in prompt, only load up to %d tokens\n",
           BatchConfig::max_tokens_per_batch());
    printf("Got: %d tokens\n", batch_config->num_tokens);

    // pid_t pid = getpid();
    // std::string filename = "bc_" + std::to_string(pid) + ".txt";
    // std::ofstream file(filename);
    // if (file.is_open()) {
    //     file << *batch_config << std::endl;
    //     file.close();
    //     std::cout << "String written to file: " << filename << std::endl;
    // } else {
    //     std::cout << "Unable to open file: " << filename << std::endl;
    // }

  } else if (batch_config->num_tokens >
                 BatchConfig::max_verify_tokens_per_batch() &&
             batch_config->get_mode() != INC_DECODING_MODE) {
    printf("Warning: Speculative decoding. too many tokens in prompt, only "
           "load up to %d tokens\n",
           BatchConfig::max_verify_tokens_per_batch());
    printf("Got: %d tokens\n", batch_config->num_tokens);
  }

  for (int i = 0; i < batch_config->num_tokens; i++) {
    dram_copy[i] = batch_config->tokensInfo[i].token_id;
  }
  TokenId *fb_ptr = helperGetTensorPointerWO<TokenId>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(batch_config->num_tokens <= domain.get_volume());
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpyAsync(fb_ptr,
                            dram_copy,
                            sizeof(TokenId) * batch_config->num_tokens,
                            cudaMemcpyHostToDevice,
                            stream));
}

// q_indptr: the start offset of q in the batch for each request,
//           the length is `num_requests + 1`: [0, num_q_0, num_q_0 + num_q_1,
//           ..., num_q_0 + num_q_1 + ... + num_q_{num_requests - 1}]
// kv_indptr: the start offset of kv page_indices for each request,
//            the length is `num_requests + 1`.
// kv_indices: the page indices for kv, the length is `num_kv_pages`.
// kv_last_page_len: the cache length in the last page for each request,
//                   the length is `num_requests`.
// qk_indptr: the start offset of custom_mask in the flattened mask for each
//            request, the length is `num_requests + 1`. It can be calculated as
//            accumulative `ceil(qk_len / 8)`.
__global__ void
    prepare_inference_params_kernel(int const num_requests,
                                    BatchConfig::PerRequestInfo const *request_infos,
                                    const bool *request_completed,
                                    uint32_t const max_num_pages,
                                    int32_t *q_indptr,
                                    int32_t *kv_indptr,
                                    int32_t *kv_indices,
                                    int32_t *kv_last_page_len,
                                    int32_t *qk_indptr) {
  int const request_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (request_idx >= num_requests) {
    return;
  }

  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0, q_lens = 0, qk_lens = 0;
  int indices_offset = 0, indices_lens = 0, kv_len = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (!request_completed[requext_idx_in_batch]) {
      cnt_1++;
      int q_len = request_infos[requext_idx_in_batch].num_tokens_in_batch;
      q_lens += q_len;
      kv_len = request_infos[requext_idx_in_batch].num_tokens_in_batch +
               request_infos[requext_idx_in_batch].first_token_depth_in_request;
      qk_lens += (q_len * kv_len + 7) / 8;
      indices_offset = indices_lens;
      indices_lens += (kv_len + kPagesize - 1) / kPagesize;
    }
  }

  if (request_idx == 0) {
    q_indptr[0] = 0;
    kv_indptr[0] = 0;
    qk_indptr[0] = 0;
  }
  __syncthreads();
  q_indptr[request_idx + 1] = q_lens;
  kv_indptr[request_idx + 1] = indices_lens;
  for (int i = indices_offset; i < indices_lens; i++) {
    kv_indices[i] = max_num_pages * requext_idx_in_batch + (i - indices_offset);
  }
  kv_last_page_len[request_idx] = (kv_len - 1) % kPagesize + 1;
  qk_indptr[request_idx + 1] = qk_lens;
}

void RequestManager::load_batch_config_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 0);
  assert(task->regions.size() == 0);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  // copy meta data to workSpace
  FFHandler handle = *((FFHandler const *)task->local_args);
  checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->tokens_info,
                            &(batch_config->tokensInfo),
                            sizeof(BatchConfig::tokensInfo),
                            cudaMemcpyHostToDevice,
                            stream));

  checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->requestsInfo,
                            &(batch_config->requestsInfo),
                            sizeof(BatchConfig::requestsInfo),
                            cudaMemcpyHostToDevice,
                            stream));

  // load speculative metadata
  if (batch_config->get_mode() == BEAM_SEARCH_MODE) {
    BeamSearchBatchConfig const *beam_batch_config =
        static_cast<BeamSearchBatchConfig const *>(batch_config);

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->beamTokenInfo,
                              &(beam_batch_config->beamTokenInfo),
                              sizeof(BeamSearchBatchConfig::beamTokenInfo),
                              cudaMemcpyHostToDevice,
                              stream));

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->beamRequestsInfo,
                              &(beam_batch_config->beamRequestsInfo),
                              sizeof(BeamSearchBatchConfig::beamRequestsInfo),
                              cudaMemcpyHostToDevice,
                              stream));

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->causalMask,
                              &(beam_batch_config->causalMask),
                              sizeof(BatchConfig::causalMask),
                              cudaMemcpyHostToDevice,
                              stream));

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->request_completed,
                              &(batch_config->request_completed),
                              sizeof(BatchConfig::request_completed),
                              cudaMemcpyHostToDevice,
                              stream));

  } else if (batch_config->get_mode() == TREE_VERIFY_MODE) {
    TreeVerifyBatchConfig const *tree_batch_config =
        static_cast<TreeVerifyBatchConfig const *>(batch_config);

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->causalMask,
                              &(tree_batch_config->causalMask),
                              sizeof(BatchConfig::causalMask),
                              cudaMemcpyHostToDevice,
                              stream));

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->committed_tokens,
                              &(tree_batch_config->committed_tokens),
                              sizeof(TreeVerifyBatchConfig::committed_tokens),
                              cudaMemcpyHostToDevice,
                              stream));

    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata->request_completed,
                              &(batch_config->request_completed),
                              sizeof(BatchConfig::request_completed),
                              cudaMemcpyHostToDevice,
                              stream));
  }
  

  // load attention metadata
  if (batch_config->get_mode() == INC_DECODING_MODE) {
    int batch_size = batch_config->num_active_requests();
    if (batch_size > 0 && handle.incr_attention_metadata->enabled()) {
      // calculate the attention meta data
      {
        // BatchConfig::PerRequestInfo *request_infos = reinterpret_cast<BatchConfig::PerRequestInfo *>(static_cast<char *>(handle.batch_config_metadata) + sizeof(BatchConfig::tokensInfo));
        // bool *request_available = reinterpret_cast<bool *>(static_cast<char *>(handle.batch_config_metadata) + sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo));
        uint32_t max_num_pages = round_up_pages(BatchConfig::max_sequence_length());
        int parallelism = batch_size;
        prepare_inference_params_kernel<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
            batch_size,
            batch_config->requestsInfo,
            batch_config->request_completed,
            max_num_pages,
            handle.incr_attention_metadata->q_indptr,
            handle.incr_attention_metadata->kv_indptr,
            handle.incr_attention_metadata->kv_indices,
            handle.incr_attention_metadata->kv_last_page_len,
            handle.incr_attention_metadata->qk_indptr);
      }

      // prepare attention forward handler
      {
        static int32_t  q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
                        kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
                        kv_last_page_len_h[BatchConfig::MAX_NUM_REQUESTS];
        q_indptr_h[0] = 0;
        kv_indptr_h[0] = 0;
        for (int req_idx = 0, indptr_idx = 0; req_idx < batch_config->max_requests_per_batch(); req_idx++) {
          if (!batch_config->request_completed[req_idx]) {
            int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
            int kv_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch + batch_config->requestsInfo[req_idx].first_token_depth_in_request;
            q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
            kv_indptr_h[indptr_idx + 1] = kv_indptr_h[indptr_idx] + round_up_pages(kv_len);
            kv_last_page_len_h[indptr_idx] = (kv_len - 1) % kPagesize + 1;
            indptr_idx++;
          }
        }

        if (handle.incr_attention_metadata->prompt_handler_collections.count(batch_size) == 0) {
          handle.incr_attention_metadata->prompt_handler_collections[batch_size] = static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
        }
        BatchPrefillHandler *prefill_handler = static_cast<BatchPrefillHandler *>(handle.incr_attention_metadata->prompt_handler_collections[batch_size]);

        prefill_handler->SetCUDAStream(stream);
        prefill_handler->BeginForward<half, int32_t>(
            static_cast<void *>(handle.incr_attention_metadata->float_workspace),
            handle.incr_attention_metadata->float_workspace_size,
            static_cast<void *>(handle.incr_attention_metadata->int_workspace),
            handle.incr_attention_metadata->int_workspace_size,
            static_cast<int32_t *>(q_indptr_h),
            static_cast<int32_t *>(kv_indptr_h),
            batch_size,
            handle.incr_attention_metadata->num_q_heads(),
            handle.incr_attention_metadata->num_kv_heads(),
            handle.incr_attention_metadata->head_dim(),
            kPagesize);
      }
    }
  } else if (batch_config->get_mode() == BEAM_SEARCH_MODE) {
    assert(false && "Not implemented");
  } else if (batch_config->get_mode() == TREE_VERIFY_MODE) {
    assert(false && "Not implemented");
  } else {
    assert(false && "Not implemented");
  }
}

void RequestManager::load_positions_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  int const offset = *((int const *)task->args);
  int *pos_ptr = helperGetTensorPointerWO<int>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  int dram_copy[BatchConfig::MAX_NUM_TOKENS];

  for (int i = 0; i < batch_config->num_tokens; i++) {
    dram_copy[i] = batch_config->tokensInfo[i].abs_depth_in_request + offset;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpyAsync(pos_ptr,
                            dram_copy,
                            sizeof(int) * batch_config->num_tokens,
                            cudaMemcpyHostToDevice,
                            stream));
}

}; // namespace FlexFlow
