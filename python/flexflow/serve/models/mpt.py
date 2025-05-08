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
import random, torch, shutil


class MPTConfig:
    def __init__(self, hf_config):
        self.hidden_size = hf_config.d_model
        self.n_heads = hf_config.n_heads
        self.n_layers = hf_config.n_layers
        self.vocab_size = hf_config.vocab_size
        self.rotary_embedding_meta = RotaryEmbeddingMeta(apply_rotary_embedding=False)
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.n_heads
        self.num_key_value_heads = hf_config.n_heads


class FlexFlowMPT(FlexFlowModel):
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
        self.mpt_config = MPTConfig(hf_config)
        self.maxint = 2**31 - 1

        # Sanity checks
        if self.mpt_config.hidden_size % self.mpt_config.n_heads != 0:
            raise ValueError(
                f"Hidden size ({self.mpt_config.hidden_size}) is not divisible by n_head ({self.mpt_config.n_heads})"
            )

        # Sanity checks
        if (
            self.mpt_config.n_heads < self.ffconfig.tensor_parallelism_degree
            or self.mpt_config.n_heads % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.mpt_config.n_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        assert (
            self.mpt_config.hidden_size % self.ffconfig.tensor_parallelism_degree == 0
        )
        self.head_dim = self.mpt_config.hidden_size // self.mpt_config.n_heads
        self.tot_num_heads = 3 * self.mpt_config.n_heads
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
        input = self.ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        hidden_states = self.ffmodel.embedding(
            input,
            self.mpt_config.vocab_size,
            self.mpt_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wte",
        )

        axes = [
            0,
        ]

        for i in range(self.mpt_config.n_layers):
            self.ffmodel.set_transformer_layer_id(i)

            if i == 0:
                layernorm_output = self.ffmodel.layer_norm(
                    hidden_states,
                    axes,
                    True,
                    1e-05,
                    False,
                    name=f"layers.{i}.norm_1",
                )
            else:
                hidden_states, layernorm_output = self.ffmodel.residual_layer_norm(
                    intermediate_output,
                    hidden_states,
                    None,
                    False,
                    axes,
                    True,
                    1e-05,
                    False,
                    name=f"layers.{i}.norm_1",
                )

            qkv_proj = self.ffmodel.dense(
                layernorm_output,
                self.head_dim * self.tot_num_heads,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.attn.qkv_proj",
            )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                o_proj = self.ffmodel.spec_inc_multihead_self_attention(
                    qkv_proj,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.n_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.mpt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention_verify(
                    qkv_proj,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.n_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.mpt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention(
                    qkv_proj,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.n_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.mpt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            else:
                assert False

            attn_outputs = self.ffmodel.dense(
                o_proj,
                self.mpt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.attn.o_proj",
            )

            hidden_states, layernorm_output = self.ffmodel.residual_layer_norm(
                attn_outputs,
                hidden_states,
                None,
                False,
                axes,
                True,
                1e-05,
                False,
                name=f"layers.{i}.norm_2",
            )
            # mlp
            layernorm_output = self.ffmodel.dense(
                layernorm_output,
                4 * self.mpt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.ffn.up_proj",
            )
            layernorm_output = self.ffmodel.gelu(layernorm_output)
            intermediate_output = self.ffmodel.dense(
                layernorm_output,
                self.mpt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.ffn.down_proj",
            )

        _, all_final_norm = self.ffmodel.residual_layer_norm(
            intermediate_output,
            hidden_states,
            None,
            False,
            axes,
            True,
            1e-05,
            False,
            name=f"norm_f",
        )
        lm_head = self.ffmodel.dense(
            all_final_norm,
            self.mpt_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.generation_config.do_sample:
            dense = self.ffmodel.scalar_true_divide(
                lm_head, self.generation_config.temperature, False
            )
            softmax = self.ffmodel.softmax(dense, -1)
            output = self.ffmodel.sampling(softmax, self.generation_config.topp)
        else:
            softmax = self.ffmodel.softmax(lm_head, -1)
            output = self.ffmodel.argmax(softmax, False)

        if peft_enabled(self.ffconfig.peft_support_mode):
            # TODO: add attention projections
            self.ffmodel.add_lora_layers(["up_proj", "down_proj"])

    # TODO: finish this
    def convert_hf_weight_name(name):
        return (
            name.replace("transformer.blocks.", "layers.")
            .replace("transformer.", "")
            .replace("attn.out_proj", "attn.o_proj")
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = FlexFlowMPT.convert_hf_weight_name(name)
            if "Wqkv" in name:
                name_q = name.replace("attn.Wqkv", "attn.q_proj")
                name_k = name.replace("attn.Wqkv", "attn.k_proj")
                name_v = name.replace("attn.Wqkv", "attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.d_model,
                        model.config.d_model,
                        model.config.d_model,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))

        shutil.copy(
            os.path.join(dst_folder, "wte.weight"),
            os.path.join(dst_folder, "lm_head.weight"),
        )
