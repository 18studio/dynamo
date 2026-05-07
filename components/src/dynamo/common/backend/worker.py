# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python shim over ``dynamo._core.backend.Worker``.

The lifecycle state machine, signal handling, discovery unregister,
grace-period sleep, drain, cleanup, and 3-phase runtime shutdown all live
in Rust (``dynamo_backend_common::Worker``). This module only:

  * exposes the engine-author-friendly ``WorkerConfig`` dataclass with a
    ``from_runtime_config`` helper, and
  * drives the Rust ``Worker`` for a given ``LLMEngine`` instance.

Engine semantics (``start``/``generate``/``abort``/``drain``/``cleanup``)
remain the only thing engine authors implement.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

from dynamo._core import backend as _backend
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.runtime.logging import configure_dynamo_logging

from .engine import LLMEngine

logger = logging.getLogger(__name__)

# Map the user-facing `dynamo.common.constants.DisaggregationMode` (which
# carries 4 modes including ENCODE) to the 3-mode Rust enum. ENCODE is not
# supported by the unified abstraction yet — multimodal encode workers stay
# on the legacy main.py path until they migrate.
_DISAGG_MODE_TO_RUST = {
    DisaggregationMode.AGGREGATED: _backend.DisaggregationMode.Aggregated,
    DisaggregationMode.PREFILL: _backend.DisaggregationMode.Prefill,
    DisaggregationMode.DECODE: _backend.DisaggregationMode.Decode,
}


def _to_rust_disaggregation_mode(mode: DisaggregationMode):
    try:
        return _DISAGG_MODE_TO_RUST[mode]
    except KeyError as e:
        raise NotImplementedError(
            f"DisaggregationMode.{mode.name} is not supported by the unified "
            "backend abstraction; use the legacy backend entry point for this "
            "worker role"
        ) from e


def _coerce_disagg_mode(value) -> DisaggregationMode:
    """Defensively map a runtime-config field to the unified
    `DisaggregationMode`. Any value that isn't an instance of the
    expected enum (e.g. trtllm's locally-defined enum, an unset `None`)
    falls back to `AGGREGATED` — the engine's `from_args` is expected to
    pass an explicit `disaggregation_mode=` override via `**overrides`
    when its native type doesn't match."""
    if isinstance(value, DisaggregationMode):
        return value
    return DisaggregationMode.AGGREGATED


@dataclass
class WorkerConfig:
    namespace: str
    component: str = "backend"
    endpoint: str = "generate"
    model_name: str = ""
    served_model_name: Optional[str] = None
    model_input: ModelInput = field(default_factory=lambda: ModelInput.Tokens)
    endpoint_types: str = "chat,completions"
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"
    event_plane: Optional[str] = None
    use_kv_events: bool = False
    custom_jinja_template: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    exclude_tools_when_tool_choice_none: bool = True
    enable_local_indexer: bool = True
    metrics_labels: list[tuple[str, str]] = field(default_factory=list)
    # Disaggregation role; default AGGREGATED keeps existing callers unchanged.
    # The Rust Worker reads this for registration (Prefill→ModelType::Prefill,
    # Decode→disable local indexer); engines read it from their own runtime
    # config to switch per-mode protocol behavior in `generate()`.
    disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED

    @classmethod
    def from_runtime_config(
        cls,
        runtime_cfg,
        model_name: str,
        served_model_name: Optional[str] = None,
        model_input: Optional[ModelInput] = None,
        **overrides,
    ) -> "WorkerConfig":
        """Build from any object that carries DynamoRuntimeConfig fields.

        Works with vllm.Config, trtllm.Config (inherit DynamoRuntimeConfig
        directly) and sglang DynamoConfig (nested in config.dynamo_args).
        """
        kwargs = {
            "namespace": runtime_cfg.namespace,
            "component": getattr(runtime_cfg, "component", None) or "backend",
            "endpoint": getattr(runtime_cfg, "endpoint", None) or "generate",
            "model_name": model_name,
            "served_model_name": served_model_name,
            "endpoint_types": getattr(
                runtime_cfg, "endpoint_types", "chat,completions"
            ),
            "discovery_backend": runtime_cfg.discovery_backend,
            "request_plane": runtime_cfg.request_plane,
            "event_plane": runtime_cfg.event_plane,
            "use_kv_events": getattr(runtime_cfg, "use_kv_events", False),
            "custom_jinja_template": getattr(
                runtime_cfg, "custom_jinja_template", None
            ),
            "tool_call_parser": getattr(runtime_cfg, "dyn_tool_call_parser", None),
            "reasoning_parser": getattr(runtime_cfg, "dyn_reasoning_parser", None),
            "exclude_tools_when_tool_choice_none": getattr(
                runtime_cfg, "exclude_tools_when_tool_choice_none", True
            ),
            "enable_local_indexer": getattr(runtime_cfg, "enable_local_indexer", True),
            # Backends carry the resolved DisaggregationMode under different
            # field names — vLLM/TRT-LLM use `disaggregation_mode`, SGLang
            # uses `serving_mode`. Probe both before falling back to AGGREGATED.
            #
            # Type-check the value: TRT-LLM defines its OWN
            # `DisaggregationMode` enum (`dynamo.trtllm.constants`) with
            # different string values from the unified `DisaggregationMode`
            # used here. If we let a foreign enum through, the dataclass
            # silently stores a wrong-typed value and downstream
            # `_to_rust_disaggregation_mode` errors with KeyError. Reject
            # at the boundary; trtllm's `from_args` passes an explicit
            # override via `**overrides` so this fallback never bites it.
            "disaggregation_mode": _coerce_disagg_mode(
                getattr(
                    runtime_cfg,
                    "disaggregation_mode",
                    getattr(runtime_cfg, "serving_mode", None),
                )
            ),
        }
        if model_input is not None:
            kwargs["model_input"] = model_input
        kwargs.update(overrides)
        return cls(**kwargs)


class Worker:
    """Drive the Rust ``Worker`` for a single ``LLMEngine`` instance."""

    def __init__(self, engine: LLMEngine, config: WorkerConfig):
        self.engine = engine
        self.config = config

    async def run(self) -> None:
        configure_dynamo_logging()

        if self.config.use_kv_events:
            # The runtime auto-detects NATS now; the field is preserved on
            # the dataclass for source-compat with existing callers but no
            # longer plumbed anywhere. Surface the silent-drop loudly so
            # operators don't assume their setting took effect.
            warnings.warn(
                "WorkerConfig.use_kv_events is deprecated and ignored. NATS "
                "enablement is determined automatically from the event-plane "
                "configuration; remove this argument.",
                DeprecationWarning,
                stacklevel=2,
            )

        runtime_cfg = _backend.RuntimeConfig(
            discovery_backend=self.config.discovery_backend,
            request_plane=self.config.request_plane,
            event_plane=self.config.event_plane,
        )
        worker_cfg = _backend.WorkerConfig(
            namespace=self.config.namespace,
            component=self.config.component,
            endpoint=self.config.endpoint,
            model_name=self.config.model_name,
            served_model_name=self.config.served_model_name,
            model_input=self.config.model_input,
            endpoint_types=self.config.endpoint_types,
            custom_jinja_template=self.config.custom_jinja_template,
            tool_call_parser=self.config.tool_call_parser,
            reasoning_parser=self.config.reasoning_parser,
            exclude_tools_when_tool_choice_none=(
                self.config.exclude_tools_when_tool_choice_none
            ),
            enable_local_indexer=self.config.enable_local_indexer,
            metrics_labels=list(self.config.metrics_labels),
            disaggregation_mode=_to_rust_disaggregation_mode(
                self.config.disaggregation_mode
            ),
            runtime=runtime_cfg,
        )

        loop = asyncio.get_running_loop()
        worker = _backend.Worker(self.engine, worker_cfg, loop)
        await worker.run()
