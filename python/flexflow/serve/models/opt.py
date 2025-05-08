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
import random, shutil


class OPTConfig:
    def __init__(self, hf_config):
        self.do_layer_norm_before = hf_config.do_layer_norm_before
        self.dropout = hf_config.dropout
        self.enable_bias = hf_config.enable_bias
        self.ffn_dim = hf_config.ffn_dim
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_elementwise_affine = hf_config.layer_norm_elementwise_affine
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.word_embed_proj_dim = hf_config.word_embed_proj_dim
        self.rotary_embedding_meta = RotaryEmbeddingMeta(apply_rotary_embedding=False)
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = hf_config.num_attention_heads


class FlexFlowOPT(FlexFlowModel):
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
        self.opt_config = OPTConfig(hf_config)
        self.maxint = 2**31 - 1

        # Sanity checks
        if self.opt_config.hidden_size % self.opt_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.opt_config.hidden_size}) is not divisible by n_head ({self.opt_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.opt_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.opt_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.opt_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        assert self.opt_config.hidden_size % self.opt_config.num_attention_heads == 0
        self.head_dim = (
            self.opt_config.hidden_size // self.opt_config.num_attention_heads
        )
        self.tot_num_heads = 3 * self.opt_config.num_attention_heads
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

        # OPT model positional embedding start offset is 2
        self.ffmodel.set_position_offset(2)
        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = self.ffmodel.embedding(
            input_tensor,
            self.opt_config.vocab_size,
            self.opt_config.word_embed_proj_dim,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_tokens",
        )
        positional_embedding = self.ffmodel.embedding(
            position_tensor,
            self.opt_config.max_position_embeddings,
            self.opt_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_positions",
        )

        axes = [
            0,
        ]

        for i in range(self.opt_config.num_hidden_layers):
            self.ffmodel.set_transformer_layer_id(i)

            if self.opt_config.do_layer_norm_before:
                residual, hidden_states = self.ffmodel.residual_layer_norm(
                    token if i == 0 else residual,
                    positional_embedding if i == 0 else fc2,
                    None,
                    False,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers.{i}.self_attn_layer_norm",
                )
            else:
                hidden_states = self.ffmodel.add(token, positional_embedding)
                residual = hidden_states

            qkv_proj = self.ffmodel.dense(
                hidden_states,
                3 * self.opt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.self_attn.qkv_proj",
            )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                o_proj = self.ffmodel.spec_inc_multihead_self_attention(
                    qkv_proj,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.num_attention_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.opt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention_verify(
                    qkv_proj,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.num_attention_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.opt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                o_proj = self.ffmodel.inc_multihead_self_attention(
                    qkv_proj,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.num_attention_heads,
                    self.head_dim,
                    self.head_dim,
                    0.0,  # dropout
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    self.opt_config.rotary_embedding_meta,
                    True,  # scaling_query
                    self.head_dim ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            else:
                assert False

            mha = self.ffmodel.dense(
                o_proj,
                self.opt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.o_proj",
            )
            # This is either a before or after attention LayerNorm. In both cases, we need to compute the LN here.
            residual, ff_norm = self.ffmodel.add_bias_residual_layer_norm(
                mha,
                residual,
                axes,
                self.opt_config.layer_norm_elementwise_affine,
                1e-05,
                name=f"layers.{i}.add_bias_residual_layer_norm",
            )

            if not self.opt_config.do_layer_norm_before:
                residual = ff_norm

            fc1 = self.ffmodel.dense(
                ff_norm,
                self.opt_config.ffn_dim,
                ActiMode.AC_MODE_RELU,
                True,
                name=f"layers.{i}.fc1",
            )
            fc2 = self.ffmodel.dense(
                fc1,
                self.opt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.fc2",
            )

            if not self.opt_config.do_layer_norm_before:
                _, residual = self.ffmodel.residual_layer_norm(
                    residual,
                    fc2,
                    None,
                    False,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers.{i}.final_layer_norm",
                )

        _, all_final_norm = self.ffmodel.residual_layer_norm(
            residual,
            fc2,
            None,
            False,
            axes,
            self.opt_config.layer_norm_elementwise_affine,
            1e-05,
            name=f"final_layer_norm",
        )
        lm_head = self.ffmodel.dense(
            all_final_norm,
            self.opt_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = self.ffmodel.softmax(lm_head, -1)
            # output = self.ffmodel.beam_top_k(softmax, self.opt_config.max_beam_width, False)
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
            self.ffmodel.add_lora_layers(["fc1", "fc2"])

    def convert_hf_weight_name(name):
        return (
            name.replace("decoder.", "")
            .replace("model.", "")
            .replace("self_attn.out_proj", "self_attn.o_proj")
            .replace("self_attn.o_proj.bias", "add_bias_residual_layer_norm.attn_bias")
            .replace(
                ".final_layer_norm", ".add_bias_residual_layer_norm"
            )  # important to use the leading "_" to avoid matching the last LayerNorm
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = FlexFlowOPT.convert_hf_weight_name(name)
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
        # copy embedding weights
        shutil.copy(
            os.path.join(dst_folder, "embed_tokens.weight"),
            os.path.join(dst_folder, "lm_head.weight"),
        )
