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


class STARCODERConfig:
    def __init__(self, hf_config):
        self.dropout_p = hf_config.attn_pdrop
        self.hidden_size = hf_config.n_embd
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.max_position_embeddings = hf_config.n_positions
        self.num_hidden_layers = hf_config.n_layer
        self.vocab_size = hf_config.vocab_size
        self.intermediate_size = hf_config.n_inner
        self.n_head_kv = 1 if hf_config.multi_query else hf_config.n_head
        self.rotary_embedding_meta = RotaryEmbeddingMeta(apply_rotary_embedding=False)
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.n_head
        self.num_key_value_heads = self.n_head_kv


class FlexFlowSTARCODER(FlexFlowModel):
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
        self.starcoder_config = STARCODERConfig(hf_config)
        self.maxint = 2**31 - 1

        # Sanity checks
        if (
            self.starcoder_config.hidden_size
            % self.starcoder_config.num_attention_heads
            != 0
        ):
            raise ValueError(
                f"Hidden size ({self.starcoder_config.hidden_size}) is not divisible by n_head ({self.starcoder_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.starcoder_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.starcoder_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.starcoder_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
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
        position_tensor = self.ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        self.ffmodel.set_position_offset(0)
        token = self.ffmodel.embedding(
            input_tensor,
            self.starcoder_config.vocab_size,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wte",
        )
        positional_embedding = self.ffmodel.embedding(
            position_tensor,
            self.starcoder_config.max_position_embeddings,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wpe",
        )

        axes = [
            0,
        ]

        for i in range(self.starcoder_config.num_hidden_layers):
            self.ffmodel.set_transformer_layer_id(i)

            hidden_states, ln_1 = self.ffmodel.residual_layer_norm(
                token if i == 0 else residual,
                positional_embedding if i == 0 else c_proj,
                None,
                False,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers.{i}.ln_1",
            )

            qkv_proj = self.ffmodel.dense(
                ln_1,
                3 * self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.self_attn.qkv_proj",
            )

            assert self.mode == InferenceMode.INC_DECODING_MODE
            o_proj = self.ffmodel.inc_multihead_self_attention(
                qkv_proj,
                self.starcoder_config.hidden_size,
                self.starcoder_config.num_attention_heads,
                self.starcoder_config.n_head_kv,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                0.0,  # dropout
                False,  # add_zero_attn
                DataType.DT_NONE,  # data_type
                None,  # kernel initializer
                self.starcoder_config.rotary_embedding_meta,
                name=f"layers.{i}.attn.c_attn",
            )

            mha = self.ffmodel.dense(
                o_proj,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.o_proj",
            )

            residual, l2_norm = self.ffmodel.residual_layer_norm(
                hidden_states,
                mha,
                None,
                False,
                residual,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers.{i}.ln_2",
            )

            # mlp

            c_fc = self.ffmodel.dense(
                l2_norm,
                self.starcoder_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.mlp.c_fc",
            )
            activation = self.ffmodel.gelu(c_fc, False)
            c_proj = self.ffmodel.dense(
                activation,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.mlp.c_proj",
            )

        _, ln_f = self.ffmodel.residual_layer_norm(
            residual,
            c_proj,
            None,
            False,
            axes,
            True,
            self.starcoder_config.layer_norm_epsilon,
            name=f"ln_f",
        )
        lm_head = self.ffmodel.dense(
            ln_f,
            self.starcoder_config.vocab_size,
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
            self.ffmodel.add_lora_layers(["c_fc", "c_proj"])

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = name.replace("transformer.h", "layers").replace("transformer.", "")
            if "attn.c_attn.weight" in name:
                name_q = name.replace("attn.c_attn", "attn.c_attn.q_proj")
                name_k = name.replace("attn.c_attn", "attn.c_attn.k_proj")
                name_v = name.replace("attn.c_attn", "attn.c_attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // model.config.num_attention_heads,
                        model.config.hidden_size // model.config.num_attention_heads,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            elif "attn.c_attn.bias" in name:
                name_q = name.replace("attn.c_attn", "attn.c_attn.q_proj")
                name_k = name.replace("attn.c_attn", "attn.c_attn.k_proj")
                name_v = name.replace("attn.c_attn", "attn.c_attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // model.config.num_attention_heads,
                        model.config.hidden_size // model.config.num_attention_heads,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            elif "attn.c_proj.bias" in name:
                name = name.replace("attn.c_proj", "attn.c_attn.o_proj")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            elif "attn.c_proj.weight" in name:
                name = name.replace("attn.c_proj", "attn.c_attn.o_proj")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head.weight")
        )
