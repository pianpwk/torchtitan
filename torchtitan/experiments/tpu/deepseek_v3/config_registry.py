# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.components.datasets import (
    c4_test_dataloader,
    TPUGrainDataLoader,
)
from torchtitan.experiments.tpu.components.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.components.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.deepseek_v3 import model_registry
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.experiments.tpu.graph_trainer.deepseek_v3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.models.common.moe import MoE
from torchtitan.tools.profiler import Profiler


_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)

# libtpu flags for the 4x4x4 recipes: the Qwen3-235B offload recipe's
# (qwen3/config_registry.py: _Q235_COMPILED_OFFLOAD_FLAGS), copied as the
# starting point. Each was measured on Q235, none yet on DeepSeek-V3.
_DSV3_4X4X4_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {
        "xla_tpu_scoped_vmem_limit_kib": 131072,
        # Keep activation host copies off SparseCore, where the FSDP
        # collectives run.
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        # Prove copy/overwrite ordering across donated aliases instead of
        # snapshotting activations defensively.
        "xla_tpu_copy_insertion_use_region_analysis": True,
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
        # The native non-leading-axis FSDP gathers (expert banks, wkv_a and
        # the router gate are on dim 1 here) need async all-gathers kept as
        # all-gathers rather than rewritten to all-reduce plus slice.
        "xla_tpu_prefer_async_allgather_to_allreduce": True,
    },
)


def _tpu_moe_converters() -> list:
  return [
      TpuTokenDispatcherConverter.Config(),
      TpuGroupedExpertsConverter.Config(),
      TpuRoutedExpertsConverter.Config(),
  ]


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def _disable_expert_bias_balancer(config: TPUTrainerConfig) -> None:
  """Turn off the expert-bias balancer, keeping DeepSeek's aux loss.

  The balancer (``load_balance_coeff``) updates ``expert_bias`` in place from
  a counter that ``MoE.forward`` accumulates; neither has been exercised on the
  TPU graph path. DeepSeek-V3's own microbatch-wise aux loss (coeff 1e-3,
  computed locally with no collective) stays on. This is a numerics change
  against upstream: recipes that call this are labelled as such.
  """
  assert config.model_spec is not None
  for _, moe, _, _ in config.model_spec.model.traverse(MoE.Config):
    moe.load_balance_coeff = None


def deepseek_v3_debugmodel() -> TPUTrainerConfig:
  """DeepSeek-v3 debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", attn_backend="sdpa"),
      dataloader=c4_test_dataloader(),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          max_context_length=2048,
          steps=10,
      ),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
  )


def graph_trainer_deepseek_v3_debugmodel() -> TpuGraphTrainer.Config:
  """DeepSeek-v3 debug model using TPU GraphTrainer."""
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      deepseek_v3_debugmodel(), graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
  )
  return config


def _with_tpu_kernels(config: TPUTrainerConfig) -> None:
  """TPU MoE modules, SimpleFSDP, and Tokamax Splash for MLA attention.

  Tokamax Splash takes MLA's unequal head dims (q/k 192, v 128) and turns the
  per-document positions into segment ids, so packed rows stop attending
  across documents as they do under the plain ``sdpa`` backend. Every query
  head has its own K/V head here, so there is nothing to stack.
  """
  assert config.model_spec is not None
  config.model_spec = model_registry(
      config.model_spec.flavor,
      seq_len=config.training.max_context_length,
      attn_backend="sdpa",
      converters=_tpu_moe_converters(),
  )
  _disable_expert_bias_balancer(config)
  config.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
  )
  config.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      use_tokamax_splash_attention_kernel=True,
      # Tokamax has no document-aware block skipping; segment ids still mask
      # packed documents exactly.
      sa_use_document_aware_block_skipping=False,
  )
  # Disable Shardy HLO sharding v3 (enabled by default in XLA cl/979222288 /
  # libtpu 0.0.48; see cl/979992733), which segfaults in
  # mlir::sdy::buildTensorFactorShardings on GraphTrainer.
  config.xla_flags = tpu_flags.merge_flags(
      config.xla_flags,
      xla_enable_hlo_sharding_v3=False,
  )


def graph_trainer_deepseek_v3_debugmodel_tokamax() -> TpuGraphTrainer.Config:
  """Debug model on the TPU stack: TPU MoE, SimpleFSDP, Tokamax Splash.

  With the 4x4x4 recipes' execution policy: one graph-split executable per
  layer and layer-boundary host offload. Compiled as one program instead, XLA
  fails in the vmapped Tokamax forward ("reshape should have supported layout
  before reaching the emitter", libtpu 0.0.47); split, each layer compiles.
  Numerics differ from ``deepseek_v3_debugmodel`` by design: document-masked
  attention instead of causal-only, and no expert-bias balancer.
  """
  base = deepseek_v3_debugmodel()
  _with_tpu_kernels(base)
  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base, graph_trainer_model_registry
  )
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      components=["model"],
      memory_policy="full_and_offload",
  )
  split = config.compile.graph_split
  split.enabled = True
  split.forward_layers_per_executable = 1
  split.forward_isolate_prefix = True
  split.backward_partition_mode = "by_layer"
  split.backward_layers_per_executable = 1
  split.cpu_offload_prefetch_executables = 1
  return config


def _deepseek_v3_4x4x4(
    flavor: str, *, tokens_per_rank: int
) -> TpuGraphTrainer.Config:
  """What every DeepSeek-V3 recipe on a TPU7x 4x4x4 slice (128 ranks) shares.

  Model and data: ``flavor`` at 4096-token sequences, random init, C4 through
  Grain, the DeepSeek-V2 tokenizer (102,400 tokens: the 236B vocabulary, and
  in range of 671B's 129,280-row embedding). AdamW, cosine schedule.

  Kernels: TPU MoE modules, Tokamax Splash attention, Pallas cross entropy,
  MegaBlox grouped matmuls in the kernel-native weight layout; no expert-bias
  balancer (a numerics change against upstream, see
  ``_disable_expert_bias_balancer``).

  Execution: SimpleFSDP over all 128 ranks, GraphTrainer with one compiled
  executable per layer, full remat with layer-boundary host offload.
  """
  base = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/DeepSeek-V2",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry(flavor, seq_len=4096),
      dataloader=TPUGrainDataLoader.Config(dataset="c4", shuffle=False),
      optimizer=default_adamw(lr=2.2e-4, implementation="fused"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          max_context_length=4096,
          num_tokens_per_microbatch_per_dp_rank=tokens_per_rank,
          # One microbatch per step.
          num_tokens_per_train_step=128 * tokens_per_rank,
          steps=15,
      ),
      checkpoint=CheckpointManager.Config(enable=False),
      profiler=Profiler.Config(enable_profiling=False),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(),
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
  )
  _with_tpu_kernels(base)
  _enable_pallas_cross_entropy(base)
  base.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
      data_parallel_shard_degree=128,
      expert_parallel_degree=1,
      fsdp_reshard_after_forward="always",
  )
  for _, experts, _, _ in base.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    # Not the paired W1/W3 kernel: its forward holds one whole expert weight
    # of each, double-buffered, in VMEM (70.5 of TPU7x's 64 MiB at D = 5120).
    experts.use_pallas_paired_grouped_mm = False
    experts.use_kernel_native_weight_layout = True

  config = TpuGraphTrainer.Config.from_tpu_trainer_config(
      base, graph_trainer_model_registry
  )
  config.compile = TpuGraphTrainerCompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
      memory_policy="full_and_offload",
      compile_grad_norm=True,
      enable_remat_barriers=False,
      fsdp_barrier_policy="none",
      optimizer_compile="eager",
  )
  split = config.compile.graph_split
  split.enabled = True
  split.forward_layers_per_executable = 1
  split.forward_isolate_prefix = True
  split.backward_layers_per_executable = 1
  # Issue each offloaded activation's host-to-device copy in the executable
  # right before its consumer.
  split.cpu_offload_prefetch_executables = 1
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      _DSV3_4X4X4_FLAGS,
  )
  return config


def deepseek_v3_236b_fsdp_activation_offload_4x4x4() -> TpuGraphTrainer.Config:
  """DeepSeek-V3 236B (DeepSeek-V2 shape), FSDP-only on a TPU7x 4x4x4 slice.

  ``_deepseek_v3_4x4x4`` with 8 sequences per rank. Each layer's expert
  all-gathers run under its Splash attention, and its gradient reduce-scatters
  in the next backward executable (``fsdp_pipeline``).

  Shard axes at 128 ranks: the 102,400-token vocabulary divides; the 160-expert
  banks, ``wkv_a`` (576 rows) and the router gate (160 rows) move to an axis
  that divides (see ``configure_deepseek_v3_simple_fsdp_shard_dims``).
  """
  config = _deepseek_v3_4x4x4("236B", tokens_per_rank=8 * 4096)

  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.pallas_grouped_mm_tile_m = 256
    # A dWeight is tiled along K, its weight's input dimension: D = 5120 for
    # W1/W3, F = 1536 for W2. 1024 divides D. Unmeasured.
    experts.pallas_grouped_mm_tgmm_tiling = (512, 1024, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (256, 768, 0)
    # 160 experts do not divide 128 ranks, so W2 is sharded on its trailing D
    # axis and stored [E, D, F]: tile D, keep F whole.
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)

  split = config.compile.graph_split
  split.expert_all_gathers_under_splash = True
  split.backward_partition_mode = "fsdp_pipeline"
  split.pipeline_kinds = ("rs",)
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
  )
  return config


def deepseek_v3_671b_4x4x4(*, fp8: bool = True) -> TpuGraphTrainer.Config:
  """DeepSeek-V3 671B on a TPU7x 4x4x4 slice.

  FP8 (the default): 6.04 s / step, 33.0% MFU, measured before the MaxText
  MLA attention below (not yet measured on the 4x4x4; one 671B MLA layer, fwd
  + bwd on one core: 53.3 -> 31.9 ms). ``fp8=False`` drops the FP8 and
  selective-checkpointing parts below.

  ``_deepseek_v3_4x4x4`` at 8192 tokens per rank, plus:

  * As in MaxText's DeepSeek-V3 recipe: routed experts over 8-way
    Ring-of-Experts EP (each EP group is one host's 4 chips, 2 cores each)
    times 16-way expert FSDP, dense weights FSDP over all 128 ranks; each EP
    rank's routed rows capped at 1.25x the routes it expects (routes past it
    are dropped); BF16 gradient reductions and BF16 Adam moments (FP32
    parameters kept).
  * Routing scores applied before W2 (``scale_before_down_proj``), so the
    backward does not recompute the W2 forward. Same math, different rounding.
  * Forward: each layer's parameter all-gathers are issued in the previous
    executable. Backward: each layer's gradient reduce-scatters are waited on
    in the next executable. The latency-hiding layer scheduler is off: it
    pairs the deferred reduce-scatter waits with fusions right after the
    forward gathers, exposing them.

  With ``fp8``, MaxText's ``fp8_full`` numerics (static e4m3 scale for
  weights and activations, absmax e5m2 for output gradients) and kernels:

  * Routed experts on MaxText's fused gmmv2 FP8 kernels: the W1|W3 GEMM
    gathers its rows from the all-gathered tokens in-kernel, the W2 GEMM
    combines in its epilogue. Expert weights are quantized on their FSDP shard
    and all-gathered as FP8; the routed tokens and the combine's output
    gradient are all-gathered as FP8; expert gradients are reduce-scattered in
    BF16.
  * MLA attention as in MaxText's Lineage DeepSeek-V3
    (``TpuMaxTextMLAAttention``): the q / kv up projections in its fused FP8
    kernel, which writes q / k / v in the layouts Tokamax Splash reads (no
    concat, k_pe broadcast or transposes), then Tokamax Splash.
  * Dense linears MaxText runs in FP8 (``Fp8FullLinear``): the MLA q down-
    and output projections, the kv down-projection and the dense and
    shared-expert MLPs. The router and lm_head stay BF16.
  * Selective activation checkpointing: each layer's Splash output and routing
    are saved for backward instead of recomputed. With no recompute left to
    hide them under, each backward layer's expert-bank gathers are issued in
    the previous backward executable, and the norm-weight gradient
    reductions are deferred to the last one.
  """
  config = _deepseek_v3_4x4x4("671B", tokens_per_rank=8192)
  config.optimizer.implementation = "fused_opt_states_bf16"
  config.training.mixed_precision_reduce = "bfloat16"

  config.parallelism.expert_parallel_degree = 8
  for _, dispatcher, _, _ in config.model_spec.model.traverse(
      TpuAllToAllTokenDispatcher.Config
  ):
    dispatcher.ep_transport = "ring_of_experts"
    dispatcher.capacity_factor = 1.25
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuRoutedExperts.Config
  ):
    experts.scale_before_down_proj = True
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    # MegaBlox tiles for D = 7168, F = 2048, measured at 81,920 routed rows x
    # 32 local experts: 512 rows with a whole-F tile on W1/W3 and (2048, 1792)
    # on W2 run the grouped matmuls at 750-850 TF/s (vs ~380 with 256-row,
    # 1024 x 1024 tiles).
    experts.pallas_grouped_mm_tile_m = 512
    experts.pallas_grouped_mm_tile_k = 1024
    experts.pallas_grouped_mm_tile_n = 2048
    experts.pallas_grouped_mm_w2_tiling = (2048, 1792)
    experts.pallas_grouped_mm_tgmm_tiling = (512, 1024, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (512, 2048, 1024)
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    experts.fp8_bf16_gradient_reduce_scatter = fp8
    experts.use_fp8_fused_roe_kernels = fp8
    experts.fp8_fused_roe_grad_all_gather = fp8
  if fp8:
    # Imported here so that other recipes do not load the FP8 kernels.
    from torchtitan.experiments.tpu.components.fp8_full_linear import Fp8FullLinearConverter  # pylint: disable=g-import-not-at-top

    from torchtitan.experiments.tpu.deepseek_v3.maxtext_mla_attention import TpuMaxTextMLAAttentionConverter  # pylint: disable=g-import-not-at-top

    # wq_b / wkv_b run inside TpuMaxTextMLAAttention's fused FP8 kernel.
    config.model_spec.model = Fp8FullLinearConverter(
        Fp8FullLinearConverter.Config(
            fqns=(
                "attention.wo",
                "attention.wq_a",
                "attention.wkv_a",
                "feed_forward",
                "shared_experts",
            )
        )
    ).convert(config.model_spec.model)
    config.model_spec.model = TpuMaxTextMLAAttentionConverter(
        TpuMaxTextMLAAttentionConverter.Config()
    ).convert(config.model_spec.model)

  split = config.compile.graph_split
  split.forward_prefetch_parameter_gathers = True
  split.backward_partition_mode = "fsdp_pipeline"
  split.pipeline_kinds = ("rs",)
  # The norm-weight gathers (KiB) stay in place: moved ahead they queue
  # behind the expert gathers as synchronous all-reduces (~10 ms / layer).
  split.prefetch_min_gather_bytes = 1 << 20
  if fp8:
    config.compile.memory_policy = "tpu_selective_and_offload"
    config.compile.save_regions = ("attention.inner_attention", "moe.routing")
    # Prefetch only the expert banks, forward and backward: smaller gathers
    # moved ahead queue behind them as synchronous all-reduces.
    split.pipeline_kinds = ("ag", "rs")
    split.prefetch_min_gather_bytes = 256 << 20
    split.backward_defer_reduction_max_bytes = 1 << 20
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
      xla_tpu_enable_latency_hiding_layer_scheduler=tpu_flags.REMOVE,
  )
  return config
