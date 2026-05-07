# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import re
import sys
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

from tensorrt_llm.llmapi import (
    DisaggregatedParams as LlmDisaggregatedParams,
)
from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.sampling_params import GuidedDecodingParams
from torch.cuda import device_count

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions

logger = logging.getLogger(__name__)

# Match the legacy non-unified path in `trtllm/main.py` so prefill drain
# behavior is consistent across both entry points.
_DRAIN_TIMEOUT_S = 30.0
_DRAIN_POLL_INTERVAL_S = 0.5

# Range for the per-worker `disagg_machine_id` used as the snowflake bits
# of every `disagg_request_id`. Matches the legacy non-unified path
# (`workers/llm_worker.py: connection_id() % 1021`). 1021 is the largest
# 10-bit prime; we use it (and not 1024) to spread machine_ids more evenly
# under modulo distributions of small worker pools.
_DISAGG_MACHINE_ID_MAX = 1021

# `dynamo.trtllm.constants.DisaggregationMode` predates the unified
# abstraction and uses different string values from
# `dynamo.common.constants.DisaggregationMode` ("prefill_and_decode" vs.
# "agg"). Map by name so the unified WorkerConfig sees a value it
# recognises.
_TRTLLM_TO_COMMON_DISAGG = {
    DisaggregationMode.AGGREGATED: CommonDisaggregationMode.AGGREGATED,
    DisaggregationMode.PREFILL: CommonDisaggregationMode.PREFILL,
    DisaggregationMode.DECODE: CommonDisaggregationMode.DECODE,
    DisaggregationMode.ENCODE: CommonDisaggregationMode.ENCODE,
}


class TrtllmLLMEngine(LLMEngine):
    def __init__(
        self,
        engine_args: dict[str, Any],
        model_name: str,
        served_model_name: str | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int | None = None,
        max_num_tokens: int | None = None,
        kv_block_size: int = 32,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ):
        self.engine_args = engine_args
        self.model_name = model_name
        self.served_model_name = served_model_name
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.kv_block_size = kv_block_size
        # Disaggregation role; consulted in `generate()` to switch between
        # context_only / generation_only / context_and_generation handling
        # of TRT-LLM's `LlmDisaggregatedParams`.
        self.disaggregation_mode = disaggregation_mode
        self._engine: TensorRTLLMEngine | None = None
        self._default_sampling_params = SamplingParams(detokenize=False)
        self._active_requests: dict[str, Any] = {}
        # 10-bit machine_id for the TRT-LLM snowflake `disagg_request_id`.
        # That ID is the cluster-wide key the PYTHON transceiver uses to
        # match prefill→decode contexts, so two prefill replicas using
        # the same machine_id will eventually mint colliding IDs and
        # route a decode pull to the wrong context. Randomize per-process
        # to mirror the legacy `connection_id() % 1021` pattern; the Rust
        # `Worker` owns the endpoint in the unified path so we can't
        # derive from it here.
        self._disagg_machine_id = random.randint(0, _DISAGG_MACHINE_ID_MAX - 1)

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TrtllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        gpus_per_node = config.gpus_per_node or device_count()

        engine_args = {
            "model": str(config.model),
            "scheduler_config": SchedulerConfig(),
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "backend": Backend.PYTORCH,
            "kv_cache_config": KvCacheConfig(
                free_gpu_memory_fraction=config.free_gpu_memory_fraction,
            ),
            "gpus_per_node": gpus_per_node,
            "max_num_tokens": config.max_num_tokens,
            "max_seq_len": config.max_seq_len,
            "max_beam_width": config.max_beam_width,
            "max_batch_size": config.max_batch_size,
        }

        # Apply --extra-engine-args (YAML) and --override-engine-args (JSON)
        # the same way the legacy `dynamo.trtllm` worker does
        # (workers/llm_worker.py:285-298). Without this, profiler / parallel
        # scheduler caps like `--override-engine-args '{"kv_cache_config":
        # {"max_tokens": N}}'` are silently ignored on the unified path,
        # causing tests to allocate at the engine config's default fraction
        # and OOM the GPU under parallel load.
        if config.extra_engine_args:
            engine_args = update_llm_args_with_extra_options(
                engine_args, config.extra_engine_args
            )
        if config.override_engine_args:
            try:
                overrides = json.loads(config.override_engine_args)
            except json.JSONDecodeError as e:
                logging.error("Failed to parse override_engine_args as JSON: %s", e)
                sys.exit(1)
            if not isinstance(overrides, dict):
                logging.error(
                    "override_engine_args must be a JSON object, got %s",
                    type(overrides).__name__,
                )
                sys.exit(1)
            logging.info("Applying engine arg overrides: %s", overrides)
            warn_override_collisions(engine_args, overrides)
            deep_update(engine_args, overrides)

        # Pull the *post-override* values from engine_args so the engine instance
        # (and the EngineConfig the frontend reads in start()) stays in sync with
        # what the underlying TRT-LLM engine actually got.
        engine = cls(
            engine_args=engine_args,
            model_name=config.model,
            served_model_name=config.served_model_name,
            max_seq_len=engine_args.get("max_seq_len", config.max_seq_len),
            max_batch_size=engine_args.get("max_batch_size", config.max_batch_size),
            max_num_tokens=engine_args.get("max_num_tokens", config.max_num_tokens),
            kv_block_size=config.kv_block_size,
            disaggregation_mode=config.disaggregation_mode,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            disaggregation_mode=_TRTLLM_TO_COMMON_DISAGG[config.disaggregation_mode],
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        self._engine = TensorRTLLMEngine(self.engine_args, self.disaggregation_mode)
        await self._engine.initialize()

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            context_length=self.max_seq_len,
            kv_cache_block_size=self.kv_block_size,
            max_num_seqs=self.max_batch_size,
            max_num_batched_tokens=self.max_num_tokens,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self._engine is not None, "Engine not initialized"

        token_ids = request.get("token_ids", [])
        sampling_params = self._override_sampling_params(
            self._default_sampling_params, request
        )

        # Disagg dispatch — TRT-LLM uses an explicit `LlmDisaggregatedParams`
        # struct on every generate call. Prefill builds a context_only
        # struct, packs the resulting handle into the response terminal.
        # Decode pulls the prefill peer's handle off `prefill_result` and
        # flips `request_type` to generation_only so TRT-LLM skips the
        # context phase and resumes from the imported KV cache.
        disaggregated_params: LlmDisaggregatedParams | None = None
        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL
        is_decode = self.disaggregation_mode == DisaggregationMode.DECODE

        if is_prefill:
            disaggregated_params = LlmDisaggregatedParams(
                request_type="context_only",
                disagg_request_id=get_global_disagg_request_id(
                    self._disagg_machine_id
                ),
            )
        elif is_decode:
            prefill_result = require_prefill_result(
                request, _TRTLLM_TO_COMMON_DISAGG[self.disaggregation_mode]
            )
            disaggregated_params = self._decode_prefill_handoff(prefill_result)

        stop_conditions = request.get("stop_conditions", {})
        if is_prefill:
            # Prefill workers only need the KV cache populated for the
            # prompt; one token is enough. Override regardless of what
            # the client asked for.
            sampling_params.max_tokens = 1
        else:
            max_tokens = stop_conditions.get("max_tokens")
            if max_tokens is not None:
                sampling_params.max_tokens = max_tokens
            elif self.max_seq_len is not None:
                sampling_params.max_tokens = max(1, self.max_seq_len - len(token_ids))

        ignore_eos = stop_conditions.get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        # Prefill returns a single non-streaming response carrying the KV
        # transfer handle; switching off streaming here matches the
        # legacy TRT-LLM disagg path and keeps the wire-format symmetric.
        streaming = not is_prefill
        generation_result = self._engine.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=streaming,
            disaggregated_params=disaggregated_params,
        )

        request_id = context.id()
        if request_id is not None:
            self._active_requests[request_id] = generation_result

        try:
            # TensorRT-LLM reports cumulative token_ids for each output choice.
            # With n>1, choices are interleaved, so a single cursor would make
            # choice 1 inherit choice 0's offset. Track the emitted length per
            # output index and convert each cumulative list into a Dynamo delta.
            output_tokens_per_choice: dict[int, int] = {}
            async for res in generation_result:
                if not res.outputs and not res.finished:
                    yield {"finish_reason": "error", "token_ids": [], "index": 0}
                    break

                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                    next_total = len(output.token_ids)
                    # The engine returns all tokens generated so far for this
                    # choice. Calculate only the new tokens generated in this
                    # iteration to create the delta.
                    out: GenerateChunk = {
                        "token_ids": output.token_ids[tokens_so_far:],
                        "index": output_idx,
                    }

                    if output.finish_reason:
                        out["finish_reason"] = str(output.finish_reason)

                    if out.get("finish_reason") or res.finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        prompt_tokens = len(token_ids)
                        total_completion_tokens = sum(
                            len(o.token_ids) for o in res.outputs
                        )
                        out["completion_usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": prompt_tokens + total_completion_tokens,
                        }
                        # Prefill terminal carries the encoded handoff
                        # payload so the frontend's PrefillRouter can
                        # forward it to the decode peer (matches the
                        # legacy PrefillHandler wire format).
                        if is_prefill:
                            params_dict = self._encode_prefill_handoff(
                                output, disaggregated_params
                            )
                            if params_dict is not None:
                                out["disaggregated_params"] = params_dict  # type: ignore[typeddict-unknown-key]

                    # Yield the chunk to the client and update the token count
                    # for this output choice.
                    yield out
                    output_tokens_per_choice[output_idx] = next_total
        finally:
            if request_id is not None:
                self._active_requests.pop(request_id, None)

    @staticmethod
    def _decode_prefill_handoff(prefill_result: dict[str, Any]) -> LlmDisaggregatedParams:
        """Decode the prefill peer's handoff payload into a TRT-LLM
        `LlmDisaggregatedParams` ready to drive a generation_only call.
        Mirrors `HandlerBase._decode_disaggregated_params_from_prefill`.
        """
        params_dict = dict(prefill_result.get("disaggregated_params") or {})
        if not params_dict:
            raise ValueError(
                "decode worker received prefill_result without "
                "disaggregated_params; the prefill peer must populate "
                "this for TRT-LLM's KV transfer to import the cache"
            )
        # The prefill encoder may add a worker_id for routing — drop it
        # before constructing the codec dataclass.
        params_dict.pop("worker_id", None)
        DisaggregatedParamsCodec.deserialize_first_gen_log_probs(params_dict)
        params_dict.pop("_epd_metadata", None)
        decoded = DisaggregatedParamsCodec.decode(DisaggregatedParams(**params_dict))
        decoded.request_type = "generation_only"
        # Multimodal embedding handles are already baked into the imported
        # KV cache; clearing them avoids a TRT-LLM validation error in
        # generation_only mode.
        if (
            hasattr(decoded, "multimodal_embedding_handles")
            and decoded.multimodal_embedding_handles
        ):
            decoded.multimodal_embedding_handles = None
        return decoded

    @staticmethod
    def _encode_prefill_handoff(
        output: Any, input_params: LlmDisaggregatedParams | None
    ) -> dict[str, Any] | None:
        """Pack the engine's output `disaggregated_params` for transport.
        Falls back to the input params if the engine didn't override them
        (TRT-LLM occasionally returns None from a successful prefill)."""
        params_to_encode = (
            output.disaggregated_params
            if output.disaggregated_params is not None
            else input_params
        )
        encoded = DisaggregatedParamsCodec.encode(params_to_encode)
        if encoded is None:
            logger.error(
                "PREFILL: encoded disaggregated_params is None; the decode peer will fail"
            )
            return None
        params_dict = asdict(encoded)
        DisaggregatedParamsCodec.serialize_first_gen_log_probs(params_dict)
        return params_dict

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if request_id is not None:
            generation_result = self._active_requests.get(request_id)
            if generation_result is not None:
                generation_result.abort()
                logger.debug("Aborted request %s", request_id)

    async def drain(self) -> None:
        """Wait for in-flight requests to finish before cleanup.

        Only meaningful on prefill workers: their NIXL transfers may still
        be reading GPU memory when a decode peer is in the middle of a
        request, and freeing that memory under an active transfer
        crashes the decode worker (issue #7319). Mirrors the legacy
        `_make_drain_callback` polling behaviour from `trtllm/main.py`.
        """
        if (
            self._engine is None
            or self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            return

        deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_S
        logger.info(
            "Draining in-flight requests on prefill worker (timeout=%.1fs)",
            _DRAIN_TIMEOUT_S,
        )
        while asyncio.get_running_loop().time() < deadline:
            try:
                stats_iter = self._engine.llm.get_stats_async(timeout=2)
                stat = await anext(stats_iter)
                active = stat.get("numActiveRequests", 0)
                queued = stat.get("numQueuedRequests", 0)
                if active + queued == 0:
                    logger.info("All in-flight requests drained")
                    return
                logger.info(
                    "Waiting for %d in-flight request(s) (active=%d, queued=%d)",
                    active + queued,
                    active,
                    queued,
                )
            except Exception as e:
                logger.debug("Stats poll failed during drain: %s", e)
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)
        logger.warning(
            "Drain timeout (%.1fs) reached; proceeding with shutdown — "
            "some NIXL transfers may still be in flight",
            _DRAIN_TIMEOUT_S,
        )

    async def cleanup(self) -> None:
        if self._engine is not None:
            await self._engine.cleanup()
            logger.info("TensorRT-LLM engine shutdown")

    @staticmethod
    def _override_sampling_params(
        sampling_params: SamplingParams, request: GenerateRequest
    ) -> SamplingParams:
        overrides = {
            key: value
            for key, value in request.get("sampling_options", {}).items()
            if value is not None
        }

        guided_decoding = overrides.pop("guided_decoding", None)
        if guided_decoding is not None and isinstance(guided_decoding, dict):
            regex = guided_decoding.get("regex")
            choice = guided_decoding.get("choice")
            if choice and not regex:
                valid_choices = [c for c in choice if c is not None]
                if valid_choices:
                    regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"
            overrides["guided_decoding"] = GuidedDecodingParams(
                json=guided_decoding.get("json"),
                regex=regex,
                grammar=guided_decoding.get("grammar"),
                json_object=guided_decoding.get("json_object", False),
                structural_tag=guided_decoding.get("structural_tag"),
            )

        n = overrides.get("n")
        if (
            isinstance(n, int)
            and not isinstance(n, bool)
            and n > 1
            and hasattr(sampling_params, "best_of")
        ):
            # Dynamo does not expose best_of here, but TRT-LLM validates that
            # its internal best_of is at least n when cloning SamplingParams.
            # Keep that private field in lockstep so OpenAI n>1 requests do
            # not fail before generation starts.
            best_of = getattr(sampling_params, "best_of", None)
            if best_of is None or best_of < n:
                overrides["best_of"] = n

        return dataclasses.replace(sampling_params, **overrides)
