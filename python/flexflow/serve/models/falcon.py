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
        self.bias = hf_config.bias
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.multi_query = hf_config.multi_query
        self.n_head = (
            hf_config.n_head
            if "n_head" in hf_config.__dict__
            else hf_config.num_attention_heads
        )
        self.n_head_kv = hf_config.n_head_kv if "n_head_kv" in hf_config.__dict__ else 1
        self.n_layer = (
            hf_config.n_layer
            if "n_layer" in hf_config.__dict__
            else hf_config.num_hidden_layers
        )
        self.parallel_attn = hf_config.parallel_attn
        self.vocab_size = hf_config.vocab_size
        self.rotary_embedding_meta = RotaryEmbeddingMeta(
            apply_rotary_embedding=True,
            rope_theta=(
                hf_config.rope_theta if "rope_theta" in hf_config.__dict__ else 10000.0
            ),
        )
        if "rope_scaling" in hf_config.__dict__:
            if hf_config.rope_scaling is not None:
                self.rotary_embedding_meta.rope_type = hf_config.rope_scaling[
                    "rope_type"
                ]
                self.rotary_embedding_meta.factor = hf_config.rope_scaling["factor"]
                self.rotary_embedding_meta.low_freq_factor = hf_config.rope_scaling[
                    "low_freq_factor"
                ]
                self.rotary_embedding_meta.high_freq_factor = hf_config.rope_scaling[
                    "high_freq_factor"
                ]
                self.rotary_embedding_meta.original_max_position_embeddings = (
                    hf_config.rope_scaling["original_max_position_embeddings"]
                )
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.n_head_kv


class FlexFlowFalcon(FlexFlowModel):
    def __init__(
        self,
        ffmodel: FFModel,
        mode: InferenceMode,
        generation_config: GenerationConfig,
        ffconfig: FFConfig,
        hf_config: any,
        data_type: DataType,
    ):
        self.ffmodel = ffmodel
        self.mode = mode
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        self.data_type = data_type
        self.falcon_config = FalconConfig(hf_config)
        self.maxint = 2**31 - 1

        # Sanity checks
        if self.falcon_config.hidden_size % self.falcon_config.n_head != 0:
            raise ValueError(
                f"Hidden size ({self.falcon_config.hidden_size}) is not divisible by n_head ({self.falcon_config.n_head})"
            )
        if (
            self.falcon_config.n_head < self.ffconfig.tensor_parallelism_degree
            or self.falcon_config.n_head % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of q attention heads ({self.falcon_config.n_head}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )
        assert self.falcon_config.hidden_size % self.falcon_config.n_head == 0
        self.head_dim = self.falcon_config.hidden_size // self.falcon_config.n_head
        self.tot_num_heads = (
            self.falcon_config.n_head + 2 * self.falcon_config.n_head_kv
        )
        self.build_model()

    def build_model(self):
        is_spec = self.mode != InferenceMode.INC_DECODING_MODE
        self.rm = RequestManager()
        batch_tensor_num_tokens = self.rm.get_max_tokens_per_batch()
        if is_spec:
            batch_tensor_num_tokens = self.rm.get_max_verify_tokens_per_batch()
        elif peft_finetuning_enabled(self.ffconfig.peft_support_mode):
            batch_tensor_num_tokens = self.rm.get_max_sequence_length()

        tokens_dims = [batch_tensor_num_tokens, 1]
        input_tensor = self.ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = self.ffmodel.embedding(
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
            self.ffmodel.set_transformer_layer_id(i)

            if i == 0:
                att_norm = self.ffmodel.layer_norm(
                    token,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers.{i}.input_layernorm",
                )
            else:
                token, att_norm = self.ffmodel.residual_layer_norm(
                    token,
                    mha,
                    mlp_output,
                    True,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers.{i}.input_layernorm",
                )

            qkv_proj = self.ffmodel.dense(
                att_norm,
                self.head_dim * self.tot_num_heads,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attention.qkv_proj",
            )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                o_proj = self.ffmodel.spec_inc_multihead_self_attention(
                    qkv_proj,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.falcon_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attention",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention_verify(
                    qkv_proj,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.falcon_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attention",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention(
                    qkv_proj,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.falcon_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attention",
                )
            else:
                assert False

            mha = self.ffmodel.dense(
                o_proj,
                self.falcon_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attention.o_proj",
            )

            dense_h_to_4h = self.ffmodel.dense(
                att_norm,
                self.falcon_config.hidden_size * 4,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.dense_h_to_4h",
            )
            dense_h_to_4h = self.ffmodel.gelu(dense_h_to_4h)
            mlp_output = self.ffmodel.dense(
                dense_h_to_4h,
                self.falcon_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.dense_4h_to_h",
            )

        _, ln_f = self.ffmodel.residual_layer_norm(
            token,
            mha,
            mlp_output,
            True,
            axes,
            True,
            self.falcon_config.layer_norm_epsilon,
            name="ln_f",
        )
        lm_head = self.ffmodel.dense(
            ln_f,
            self.falcon_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = self.ffmodel.softmax(lm_head, -1)
            # output = self.ffmodel.beam_top_k(softmax, self.falcon_config.max_beam_width, False)
            output = self.ffmodel.argmax(softmax, True)
        else:
            if self.generation_config.do_sample:
                dense = self.ffmodel.scalar_true_divide(
                    lm_head, self.generation_config.temperature, False
                )
                softmax = self.ffmodel.softmax(dense, -1)
                output = self.ffmodel.sampling(softmax, self.generation_config.topp)
            else:
                # output = self.ffmodel.arg_top_k(lm_head, 1, False)
                softmax = self.ffmodel.softmax(lm_head, -1)
                output = self.ffmodel.argmax(softmax, False)

        if peft_enabled(self.ffconfig.peft_support_mode):
            # TODO: add attention projections
            self.ffmodel.add_lora_layers(["dense_h_to_4h", "dense_4h_to_h"])

    # TODO: finish this
    def convert_hf_weight_name(name):
        return (
            name.replace("transformer.h.", "layers.")
            .replace("transformer.", "")
            .replace("self_attention.dense", "self_attention.o_proj")
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        n_head = (
            model.config.n_head
            if "n_head" in model.config.__dict__
            else model.config.num_attention_heads
        )
        for name, params in model.named_parameters():
            name = FlexFlowFalcon.convert_hf_weight_name(name)
            # Split Q,K,V attention weights
            if "self_attention.query_key_value" in name:
                name_q = name.replace(
                    "self_attention.query_key_value", "self_attention.q_proj"
                )
                name_k = name.replace(
                    "self_attention.query_key_value", "self_attention.k_proj"
                )
                name_v = name.replace(
                    "self_attention.query_key_value", "self_attention.v_proj"
                )
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // n_head,
                        model.config.hidden_size // n_head,
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
            os.path.join(dst_folder, "lm_head.weight")
        )
