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
  // printf("Entering load_tokens_task\n");

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  BatchConfig::TokenId dram_copy[BatchConfig::MAX_NUM_TOKENS];

  // Extreme long prompts are not supported, only load up to
  // BatchConfig::max_tokens_per_batch() as prompt
  if (batch_config->num_inference_tokens() > BatchConfig::max_tokens_per_batch() &&
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

  } else if (batch_config->num_inference_tokens() >
                 BatchConfig::max_verify_tokens_per_batch() &&
             batch_config->get_mode() != INC_DECODING_MODE) {
    printf("Warning: Speculative decoding. too many tokens in prompt, only "
           "load up to %d tokens\n",
           BatchConfig::max_verify_tokens_per_batch());
    printf("Got: %d tokens\n", batch_config->num_tokens);
  }

  for (int i = 0; i < batch_config->num_active_tokens(); i++) {
    dram_copy[i] = batch_config->tokensInfo[i].token_id;
  }
  TokenId *fb_ptr = helperGetTensorPointerWO<TokenId>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(batch_config->num_active_tokens() <= domain.get_volume());
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpyAsync(fb_ptr,
                            dram_copy,
                            sizeof(TokenId) * batch_config->num_active_tokens(),
                            cudaMemcpyHostToDevice,
                            stream));
}

void prepare_inference_params_kernel_h(
    BatchConfig const *batch_config,
    std::vector<int32_t> &q_indptr_h,
    std::vector<int32_t> &kv_indptr_h,
    std::vector<int32_t> &kv_page_indices_h,
    std::vector<int32_t> &kv_last_page_len_h) {
  // printf("Entering prepare_inference_params_kernel_h\n");

  PageManager *pm = PageManager::get_page_manager();

  // std::cout << "prepare_inference_params_kernel_h: " << *batch_config <<
  // std::endl;

  q_indptr_h.clear();
  kv_indptr_h.clear();
  kv_page_indices_h.clear();
  kv_last_page_len_h.clear();

  q_indptr_h.push_back(0);
  kv_indptr_h.push_back(0);

  for (int req_idx = 0; req_idx < batch_config->max_requests_per_batch();
       req_idx++) {
    if (batch_config->request_completed[req_idx] ||
        batch_config->requestsInfo[req_idx].finetuning_request) {
      continue;
    }

    // q_indptr: first token offset in batch, plus one token at the end
    // representing the total number of tokens in batch
    q_indptr_h.push_back(
        q_indptr_h.back() +
        batch_config->requestsInfo[req_idx].num_tokens_in_batch);

    // kv_indptr: starting index of KV cache pages for each request in logical
    // page table

    int num_pages_used_by_req = pm->get_num_pages_used_by_req(
        batch_config->requestsInfo[req_idx].request_guid);
    assert(num_pages_used_by_req >= 1);
    kv_indptr_h.push_back(kv_indptr_h.back() + num_pages_used_by_req);

    // kv_page_indices_h: physical indices of KV cache pages in use by each
    // request (not just the pages used by the tokens in the current batch)
    std::vector<int> req_page_indices = pm->get_req_page_indices(
        batch_config->requestsInfo[req_idx].request_guid);
    kv_page_indices_h.insert(kv_page_indices_h.end(),
                             req_page_indices.begin(),
                             req_page_indices.end());

    // kv_last_page_len_h: number of tokens in the last page in use by each
    // request
    kv_last_page_len_h.push_back(pm->get_num_tokens_in_last_used_page(
        batch_config->requestsInfo[req_idx].request_guid));
  }

  // check sizes
  int batch_size = batch_config->num_active_requests() -
                   batch_config->num_finetuning_fwd_requests() -
                   batch_config->num_finetuning_bwd_requests();
  assert(batch_size > 0);
  // printf("q_indptr_h size: %lu\n", q_indptr_h.size());
  // printf("kv_indptr_h size: %lu\n", kv_indptr_h.size());
  // printf("kv_page_indices_h size: %lu\n", kv_page_indices_h.size());
  // printf("kv_last_page_len_h size: %lu\n", kv_last_page_len_h.size());
  // printf("batch_size: %i\n", batch_size);
  assert(q_indptr_h.size() == batch_size + 1);
  assert(kv_indptr_h.size() == batch_size + 1);
  assert(kv_page_indices_h.size() >= batch_size);
  assert(kv_last_page_len_h.size() == batch_size);
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
  // printf("Entering load_batch_config_task\n");

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
  int batch_size = batch_config->num_active_requests() -
                   batch_config->num_finetuning_fwd_requests() -
                   batch_config->num_finetuning_bwd_requests();
  if (batch_config->get_mode() == INC_DECODING_MODE && batch_size > 0 &&
      handle.incr_attention_metadata->enabled()) {
    // assert(handle.incr_attention_metadata->enabled());
    // printf("Entering here, handler: %p\n", handle.incr_attention_metadata);
    std::vector<int32_t> q_indptr_h;
    std::vector<int32_t> kv_indptr_h;
    std::vector<int32_t> kv_page_indices_h;
    std::vector<int32_t> kv_last_page_len_h;
    // calculate the attention meta data
    prepare_inference_params_kernel_h(batch_config,
                                      q_indptr_h,
                                      kv_indptr_h,
                                      kv_page_indices_h,
                                      kv_last_page_len_h);
    checkCUDA(cudaMemcpyAsync(handle.incr_attention_metadata->q_indptr,
                              q_indptr_h.data(),
                              sizeof(int32_t) * q_indptr_h.size(),
                              cudaMemcpyHostToDevice,
                              stream));
    checkCUDA(cudaMemcpyAsync(handle.incr_attention_metadata->kv_indptr,
                              kv_indptr_h.data(),
                              sizeof(int32_t) * kv_indptr_h.size(),
                              cudaMemcpyHostToDevice,
                              stream));
    checkCUDA(cudaMemcpyAsync(handle.incr_attention_metadata->kv_indices,
                              kv_page_indices_h.data(),
                              sizeof(int32_t) * kv_page_indices_h.size(),
                              cudaMemcpyHostToDevice,
                              stream));
    checkCUDA(cudaMemcpyAsync(handle.incr_attention_metadata->kv_last_page_len,
                              kv_last_page_len_h.data(),
                              sizeof(int32_t) * kv_last_page_len_h.size(),
                              cudaMemcpyHostToDevice,
                              stream));
    // prepare attention forward handler
    if (handle.incr_attention_metadata->prompt_handler_collections.count(
            batch_size) == 0) {
      handle.incr_attention_metadata->prompt_handler_collections[batch_size] =
          static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
    }
    BatchPrefillHandler *handler = static_cast<BatchPrefillHandler *>(
        handle.incr_attention_metadata->prompt_handler_collections[batch_size]);
    handler->SetCUDAStream(stream);
    // static int step=0;
    PageManager *pm = PageManager::get_page_manager();
    // printf("BatchPrefillHandler %p\n", handler);
    // std::cout << "STEP " << step << ": " << *pm << std::endl;
    // step+=1;
    // std::cout << "batch_config: " << *batch_config << std::endl;
    // std::cout << "q_indptr_h: ";
    // for (int i = 0; i < q_indptr_h.size(); i++) {
    //   std::cout << q_indptr_h[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "kv_indptr_h: ";
    // for (int i = 0; i < kv_indptr_h.size(); i++) {
    //   std::cout << kv_indptr_h[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "kv_page_indices_h: ";
    // for (int i = 0; i < kv_page_indices_h.size(); i++) {
    //   std::cout << kv_page_indices_h[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "kv_last_page_len_h: ";
    // for (int i = 0; i < kv_last_page_len_h.size(); i++) {
    //   std::cout << kv_last_page_len_h[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "batch_size: " << batch_size << std::endl;

    // std::cout << "num_q_heads: " <<
    // handle.incr_attention_metadata->num_q_heads() << std::endl; std::cout <<
    // "num_kv_heads: " << handle.incr_attention_metadata->num_kv_heads() <<
    // std::endl; std::cout << "head_dim: " <<
    // handle.incr_attention_metadata->head_dim() << std::endl; std::cout <<
    // "tokens_per_page: " << pm->get_tokens_per_page() << std::endl; std::cout
    // << "float_workspace_size: " <<
    // handle.incr_attention_metadata->float_workspace_size << std::endl;
    // std::cout << "int_workspace_size: " <<
    // handle.incr_attention_metadata->int_workspace_size << std::endl;

    handler->BeginForward<half, int32_t>(
        static_cast<void *>(handle.incr_attention_metadata->float_workspace),
        handle.incr_attention_metadata->float_workspace_size,
        static_cast<void *>(handle.incr_attention_metadata->int_workspace),
        handle.incr_attention_metadata->int_workspace_size,
        static_cast<int32_t *>(q_indptr_h.data()),
        static_cast<int32_t *>(kv_indptr_h.data()),
        batch_size,
        handle.incr_attention_metadata->num_q_heads(),
        handle.incr_attention_metadata->num_kv_heads(),
        handle.incr_attention_metadata->head_dim(),
        pm->get_tokens_per_page());
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
