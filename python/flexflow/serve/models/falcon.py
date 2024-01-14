# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flexflow.core import *
from .base import FlexFlowModel
import random, torch


class FalconConfig:
    def __init__(self, hf_config):
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 64

        self.bias = hf_config.bias
        self.hidden_dropout = hf_config.hidden_dropout
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.multi_query = (
            hf_config.multi_query if "multi_query" in hf_config.__dict__ else True
        )
        self.new_decoder_architecture = hf_config.new_decoder_architecture

        self.num_attention_heads = hf_config.num_attention_heads
        self.num_kv_heads = (
            hf_config.num_kv_heads
            if (self.new_decoder_architecture or not self.multi_query)
            else 1
        )
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.n_layer = hf_config.num_hidden_layers
        self.parallel_attn = hf_config.parallel_attn
        self.vocab_size = hf_config.vocab_size

        # Standardized FlexFlow num heads fields below
        # self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.num_kv_heads


class FlexFlowFalcon(FlexFlowModel):
    def __init__(
        self,
        mode,
        generation_config,
        ffconfig,
        hf_config,
        data_type,
        max_tokens_per_batch,
        weights_filepath="",
        tokenizer_filepath="",
    ):
        self.mode = mode
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        self.data_type = data_type
        self.falcon_config = FalconConfig(hf_config)
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.falcon_config.max_spec_tree_token_num
        )

        # Sanity checks
        if self.falcon_config.hidden_size % self.falcon_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.falcon_config.hidden_size}) is not divisible by num_attention_heads ({self.falcon_config.num_attention_heads})"
            )
        if (
            self.falcon_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.falcon_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of q attention heads ({self.falcon_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        self.build_model(
            max_tokens_per_batch
            if self.mode == InferenceMode.INC_DECODING_MODE
            else max_verify_tokens_per_batch
        )

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [max_tokens_per_batch, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = ffmodel.embedding(
            input_tensor,
            self.falcon_config.vocab_size,
            self.falcon_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="word_embeddings",
        )
        axes = [
            0,
        ]

        for i in range(self.falcon_config.n_layer):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                att_norm = ffmodel.layer_norm(
                    token,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers_{i}_input_layernorm"
                    if not self.falcon_config.new_decoder_architecture
                    else f"layers_{i}_ln_attn",
                )
            else:
                token, att_norm = ffmodel.residual_layer_norm(
                    token,
                    mha,
                    mlp_output,
                    True,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers_{i}_input_layernorm"
                    if not self.falcon_config.new_decoder_architecture
                    else f"layers_{i}_ln_attn",
                )

            # MLP norm (identical to att norm for old architecture)
            if not self.falcon_config.new_decoder_architecture:
                mlp_norm = att_norm
            else:
                # Residual has already computed by attn norm (token = token + mha + mlp_output)
                mlp_norm = ffmodel.layer_norm(
                    token,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers_{i}_ln_mlp",
                )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multiquery_self_attention(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.num_attention_heads,
                    self.falcon_config.num_kv_heads,
                    self.falcon_config.head_dim,
                    self.falcon_config.head_dim,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multiquery_self_attention_verify(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.num_attention_heads,
                    self.falcon_config.num_kv_heads,
                    self.falcon_config.head_dim,
                    self.falcon_config.head_dim,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multiquery_self_attention(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.num_attention_heads,
                    self.falcon_config.num_kv_heads,
                    self.falcon_config.head_dim,
                    self.falcon_config.head_dim,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            else:
                assert False

            dense_h_to_4h = ffmodel.dense(
                mlp_norm,
                self.falcon_config.hidden_size * 4,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_mlp_dense_h_to_4h",
            )
            dense_h_to_4h = ffmodel.gelu(dense_h_to_4h)
            mlp_output = ffmodel.dense(
                dense_h_to_4h,
                self.falcon_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_mlp_dense_4h_to_h",
            )

        _, ln_f = ffmodel.residual_layer_norm(
            token,
            mha,
            mlp_output,
            True,
            axes,
            True,
            self.falcon_config.layer_norm_epsilon,
            name="ln_f",
        )
        lm_head = ffmodel.dense(
            ln_f,
            self.falcon_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(lm_head, -1)
            # output = ffmodel.beam_top_k(softmax, self.falcon_config.max_beam_width, False)
            output = ffmodel.argmax(softmax, True)
        else:
            if self.generation_config.do_sample:
                dense = ffmodel.scalar_true_divide(
                    lm_head, self.generation_config.temperature, False
                )
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.sampling(softmax, self.generation_config.topp)
            else:
                # output = ffmodel.arg_top_k(lm_head, 1, False)
                output = ffmodel.argmax(lm_head, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        num_kv_heads = (
            model.config.num_kv_heads
            if (model.config.new_decoder_architecture or not model.config.multi_query)
            else 1
        )
        for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("transformer_h_", "layers_")
                .replace("transformer_", "")
                .replace("self_attention_dense", "attention_wo")
            )
            # Split Q,K,V attention weights
            if "self_attention_query_key_value" in name:
                name_q = name.replace("self_attention_query_key_value", "attention_wq")
                name_k = name.replace("self_attention_query_key_value", "attention_wk")
                name_v = name.replace("self_attention_query_key_value", "attention_wv")
                # We split first dim of tensor, which is the output dimension. Second dimension is the input dimension, and is always equal to the hidden size
                q, k, v = torch.split(
                    params,
                    [
                        model.config.head_dim * model.config.num_attention_heads,
                        model.config.head_dim * num_kv_heads,
                        model.config.head_dim * num_kv_heads,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
        # LM head weight
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head_weight")
        )
