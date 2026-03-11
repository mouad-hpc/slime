"""Bridge / LoRA model setup helpers.

Extracted from ``model.py`` to keep the main training module focused on
forward / backward / optimizer logic.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass

from megatron.core.utils import get_attr_wrapped_model

from .lora_utils import create_lora_instance

logger = logging.getLogger(__name__)


@dataclass
class _BridgeWrapperConfig:
    """Configuration for Megatron-Bridge module wrapping."""

    is_value_model: bool = False
    wrap_with_ddp: bool = True
    use_distributed_optimizer: bool = True


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _make_value_model_hook(hidden_size: int, sequence_parallel: bool):
    """Create a pre-wrap hook that replaces the output layer with a value head."""
    from megatron.core import parallel_state

    from .model_provider import LinearForLastLayer

    def hook(model):
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        model_list = _ensure_model_list(model)
        assert len(model_post_process) == len(model_list), "Model list length and post process list length must match."

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index]:
                continue
            model_chunk.output_layer = LinearForLastLayer(
                input_size=hidden_size,
                output_size=1,
                sequence_parallel=sequence_parallel,
            )

    return hook


def _get_model_config_from_wrapped(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


_qwen35_bridges_registered = False


def _register_qwen35_bridges():
    """Register Qwen3.5 model architectures with megatron-bridge.

    Standard megatron-bridge does not know about Qwen3.5. The coding-famer
    fork (``coding-famer/Megatron-Bridge-slime@qwen35``) adds native support,
    but if that fork is not installed we fall back to aliasing Qwen3.5 to the
    closest existing Qwen3 VL bridges so ``AutoBridge.from_hf_pretrained``
    can at least resolve the architecture.
    """
    global _qwen35_bridges_registered
    if _qwen35_bridges_registered:
        return
    _qwen35_bridges_registered = True

    try:
        from megatron.bridge.models.conversion.model_bridge import (
            get_model_bridge,
            register_bridge_implementation,
        )
    except ImportError:
        return

    registry = getattr(get_model_bridge, "_exact_types", {})

    # Dense VLM (e.g. Qwen3.5-4B)
    _try_register(
        registry,
        register_bridge_implementation,
        arch_name="Qwen3_5ForConditionalGeneration",
        bridge_module="megatron.bridge.models.qwen_vl.qwen3_vl_bridge",
        bridge_class_name="Qwen3VLBridge",
        target_module="megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model",
        target_class_name="Qwen3VLModel",
    )

    # MoE VLM (e.g. Qwen3.5-35B-A3B)
    _try_register(
        registry,
        register_bridge_implementation,
        arch_name="Qwen3_5MoeForConditionalGeneration",
        bridge_module="megatron.bridge.models.qwen_vl.qwen3_vl_bridge",
        bridge_class_name="Qwen3VLMoEBridge",
        target_module="megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model",
        target_class_name="Qwen3VLModel",
    )


def _try_register(registry, register_fn, *, arch_name, bridge_module, bridge_class_name, target_module, target_class_name):
    """Register a single architecture if not already present."""
    import importlib

    # Check both string and class keys
    already = arch_name in registry
    if not already:
        try:
            import transformers
            cls = getattr(transformers, arch_name, None)
            if cls is not None:
                already = cls in registry
        except Exception:
            pass

    if already:
        logger.debug("%s already registered in megatron-bridge", arch_name)
        return

    try:
        bridge_mod = importlib.import_module(bridge_module)
        bridge_cls = getattr(bridge_mod, bridge_class_name)
        target_mod = importlib.import_module(target_module)
        target_cls = getattr(target_mod, target_class_name)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Cannot register %s: %s bridge not available (%s). "
            "Install the coding-famer Megatron-Bridge fork for Qwen3.5 support.",
            arch_name, bridge_class_name, exc,
        )
        return

    register_fn(source=arch_name, target=target_cls, bridge_class=bridge_cls)
    logger.info("Registered %s → %s (target=%s)", arch_name, bridge_class_name, target_class_name)


def _patch_qwen35_config(bridge):
    """Patch Qwen3.5 text_config for compatibility with Qwen3 VL bridges.

    Qwen3.5 configs differ from Qwen3 VL in two ways:
    1. ``rope_theta`` is nested inside ``rope_parameters`` instead of top-level.
    2. MoE variants lack ``intermediate_size`` (the bridge reads it for ffn_hidden_size).
    """
    hf_config = getattr(bridge, "hf_pretrained", None)
    if hf_config is None:
        return
    hf_config = getattr(hf_config, "config", hf_config)
    text_config = getattr(hf_config, "text_config", hf_config)

    if not hasattr(text_config, "rope_theta"):
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params and "rope_theta" in rope_params:
            text_config.rope_theta = rope_params["rope_theta"]
            logger.info("Patched text_config.rope_theta = %s", text_config.rope_theta)

    if not hasattr(text_config, "intermediate_size"):
        fallback = getattr(text_config, "shared_expert_intermediate_size", None) or getattr(text_config, "moe_intermediate_size", None)
        if fallback is not None:
            text_config.intermediate_size = fallback
            logger.info("Patched text_config.intermediate_size = %s", fallback)


def _setup_lora_model_via_bridge(args: Namespace) -> list:
    """Build Megatron model with LoRA using Megatron-Bridge.

    This handles:
    1. Creating the Bridge and Provider
    2. Creating and registering the LoRA pre-wrap hook
    3. Registering value-model hooks if needed
    4. Building the DDP-wrapped model

    Args:
        args: Training arguments.

    Returns:
        List of DDP-wrapped model chunks with LoRA applied.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Qwen3.5 models are not natively registered in megatron-bridge.
    # Register them with existing Qwen3 VL bridges before calling AutoBridge.
    _register_qwen35_bridges()

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Qwen3.5 stores rope_theta inside rope_parameters, but the Qwen3 VL bridge
    # reads it as a top-level attribute on text_config.  Patch it through.
    _patch_qwen35_config(bridge)

    provider = bridge.to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    lora = create_lora_instance(args)

    def apply_lora_hook(model_chunks):
        transformed = lora(model_chunks, training=True)
        lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_lora_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model
