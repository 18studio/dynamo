# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import importlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import yaml
from vllm_omni.distributed.omni_connectors import (
    ConnectorSpec,
    OmniConnectorFactory,
    initialize_orchestrator_connectors,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import _apply_omni_final_stage_metadata
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.entrypoints.utils import load_stage_configs_from_model
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import _build_sampling_params, parse_omni_request

logger = logging.getLogger(__name__)


@dataclass
class _Proxy:
    """Expose previous-stage output attributes to vLLM-Omni v0.20 processors."""

    engine_outputs: Any = None
    original_prompt: Any = None

    def __getattr__(self, name: str) -> Any:
        if self.engine_outputs:
            value = getattr(self.engine_outputs[0], name)
            if name == "prompt_token_ids" and not value:
                return _prompt_token_ids_from_prompt(self.original_prompt) or value
            return value
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


class OmniStageWorker:
    """Single-stage worker: fetches inputs → runs processor → runs engine → writes output.

    For stage 0: gets engine_inputs directly from request.
    For stage N > 0: fetches previous stage outputs from connectors via stage_connector_refs,
    runs the pre-processor (e.g. thinker2talker) to produce this stage's engine inputs,
    then runs the engine.

    Non-final stages write output to a connector and yield stage_connector_refs for the router.
    Final stages write to SHM and yield shm_meta for the router to format.
    """

    def __init__(
        self,
        engine: StageEngine,
        stage_config: Any,
        connectors: dict,
        stage_id: int,
        output_modalities: list | None = None,
        default_video_fps: int = 16,
    ) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.connectors = connectors  # {(from_stage, to_stage): vllm_omni connector}
        self._output_modalities = output_modalities or []
        self._default_video_fps = default_video_fps
        self.stage_config = stage_config

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        final_stage_id = req.final_stage_id
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)

        # --- Resolve engine inputs ---
        sampling_params_list_override: dict | None = None
        if stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
            sampling_params_list_override = req.sampling_params_list
            try:
                stage_list = self._fetch_stage_inputs(
                    stage_connector_refs, request_id, original_prompt
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return

            input_count = sum(
                1 for item in stage_list if item.engine_outputs is not None
            )
            if input_count != len(self._engine_input_source or stage_connector_refs):
                logger.warning(
                    "Stage %d: expected %d stage inputs, got %d",
                    self.stage_id,
                    len(self._engine_input_source or stage_connector_refs),
                    input_count,
                )

            if self._processor is not None:
                prompt = self._run_processor(
                    stage_list,
                    stage_connector_refs,
                    original_prompt,
                )
                if isinstance(prompt, list) and not prompt:
                    yield {
                        "error": (
                            f"Stage {self.stage_id}: processor produced no engine "
                            "inputs from upstream stage output"
                        ),
                        "finished": True,
                    }
                    return
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
            else:
                # No processor: check if the upstream output has the
                # structure needed to build an OmniEngineCoreRequest
                # (e.g. code2wav receiving token_ids from talker).
                # Otherwise fall back to passing the raw data directly.
                upstream = stage_list[-1].engine_outputs[0]
                if hasattr(upstream, "outputs") and upstream.outputs:
                    try:
                        prompt = self._build_engine_core_request_from_upstream(
                            stage_list,
                            request_id,
                            sampling_params_list_override,
                            final_stage_id,
                        )
                    except RuntimeError as e:
                        yield {"error": str(e), "finished": True}
                        return
                else:
                    prompt = upstream
        elif req.request_id is not None:
            # Stage 0 via router: raw request forwarded with request_id — parse it.
            parsed = await parse_omni_request(
                request,
                self._output_modalities,
                self._default_video_fps,
                tokenizer_getter=self.engine.get_tokenizer,
            )
            prompt = parsed["engine_inputs"]
            original_prompt = parsed["original_prompt"]
            sampling_params_list_override = parsed["sampling_params_list"]
        else:
            # Direct frontend → stage (single-stage, no router).
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
        if stage_connector_refs:
            try:
                prompt = self._build_engine_core_request_from_stage_prompt(
                    prompt, request_id, sp, final_stage_id
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return
        else:
            prompt = self._prepare_initial_stage_prompt(
                prompt, request_id, sp, final_stage_id
            )
        last_result = None
        interstage_result = None

        try:
            async for chunk in self.engine.generate(
                prompt,
                request_id=request_id,
                sampling_params_list=sp,
            ):
                last_result = chunk
                if _has_multimodal_stage_payload(chunk):
                    interstage_result = chunk
        except Exception as e:
            logger.error(
                "Stage %d engine error for %s: %s",
                self.stage_id,
                request_id,
                e,
                exc_info=True,
            )
            yield {"error": str(e), "finished": True}
            return

        _ensure_cumulative_token_ids(last_result)
        _ensure_cumulative_token_ids(interstage_result)

        # --- Write output ---
        # Check for a downstream connector first, regardless of final_output.
        # In vllm-omni's native mode, multiple stages can set final_output=True
        # (meaning "produces user-visible output"). In Dynamo's disaggregated
        # mode the actual pipeline topology — connector edges from the YAML —
        # determines whether output should go to a connector or to SHM.
        from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
        connector = self.connectors.get((from_s, to_s))
        if connector is not None:
            connector_result = interstage_result or last_result
            try:
                ok, _, metadata = connector.put(  # type: ignore[arg-type]
                    from_s, to_s, request_id, connector_result
                )
            except Exception as e:
                logger.error(
                    "Stage %d: connector.put() raised %s: %s",
                    self.stage_id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                yield {"error": f"connector.put() raised: {e}", "finished": True}
                return
            if not ok:
                yield {"error": "connector.put() failed", "finished": True}
                return
            out: dict = {
                "original_prompt": original_prompt,
                "stage_connector_refs": {
                    **{str(k): v for k, v in stage_connector_refs.items()},
                    str(self.stage_id): metadata,
                },
                "finished": True,
            }
            if final_stage_id is not None:
                out["final_stage_id"] = final_stage_id
            if getattr(self.stage_config, "final_output", False):
                out["shm_meta"] = shm_write_bytes(
                    serialize_obj(last_result),
                    name=f"{request_id}-stage-{self.stage_id}",
                )
            if sampling_params_list_override is not None:
                out["sampling_params_list"] = sampling_params_list_override
            yield out
            return

        # Final stage → router: write output to shared memory and return the SHM handle.
        # The router reads it back via shm_deserialize() to format the response.
        #
        # NOTE: This is a single-node-only workaround — SHM requires the final stage
        # worker and the router to reside on the same machine. A proper multi-node
        # solution would use a connector edge (like inter-stage connectors) instead.
        # Tracked in TODO: shm_meta should be replaced by a YAML-configured connector edge.
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=request_id)
        yield {"shm_meta": shm_meta, "finished": True}

    def _build_engine_core_request_from_upstream(
        self,
        stage_list: list[_Proxy],
        request_id: str,
        sampling_params_list_override: dict | None,
        final_stage_id: int | None,
    ):
        """Build an OmniEngineCoreRequest from the upstream stage output.

        Used for stages without a custom processor (e.g. code2wav).  Mirrors
        what the native orchestrator does via ``build_engine_core_request_from_tokens``
        and ``_forward_to_next_stage``.  Building an ``EngineCoreRequest``
        bypasses ``InputProcessor.process_inputs()`` which would fail for
        non-autoregressive stages (``worker_type: generation``) with
        "This model does not support generation".

        Raises RuntimeError on unexpected upstream output structure.
        """
        try:
            # engine_outputs[0]: first (and only) RequestOutput — Dynamo
            # processes one request at a time per stage.
            # outputs[0]: first CompletionOutput (n=1 sampling).
            # Matches native orchestrator's process_engine_inputs pattern.
            upstream = stage_list[-1].engine_outputs[0]
            token_ids = upstream.outputs[0].token_ids
        except (IndexError, AttributeError) as e:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot extract token_ids from "
                f"upstream output: {e}"
            ) from e

        tokens_prompt = OmniTokensPrompt(prompt_token_ids=list(token_ids))
        return self._build_engine_core_request_from_stage_prompt(
            tokens_prompt,
            request_id,
            _build_sampling_params(self.stage_config, sampling_params_list_override),
            final_stage_id,
        )

    def _build_engine_core_request_from_stage_prompt(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list | None,
        final_stage_id: int | None = None,
    ) -> Any:
        """Wrap downstream token prompts the same way vLLM-Omni's orchestrator does."""
        if isinstance(prompt, OmniEngineCoreRequest):
            return self._apply_final_stage_metadata(prompt, final_stage_id)
        has_token_ids = hasattr(prompt, "prompt_token_ids") or (
            isinstance(prompt, dict) and "prompt_token_ids" in prompt
        )
        if not has_token_ids or not sampling_params_list:
            return prompt

        params = sampling_params_list[0]
        prompt = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=prompt,
            params=params,
        )
        # Pre-built EngineCoreRequests skip the output processor registration
        # in _build_add_request_message (the isinstance(prompt, EngineCoreRequest)
        # branch bypasses that block).  Register manually so that the engine's
        # output processor can match the response back to this request.
        prompt.external_req_id = prompt.request_id
        self.engine.engine.output_processors[0].add_request(
            request=prompt,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        return self._apply_final_stage_metadata(prompt, final_stage_id)

    def _prepare_initial_stage_prompt(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list | None,
        final_stage_id: int | None,
    ) -> Any:
        """Pre-tokenize stage-0 prompts when downstream stages need sidecars.

        Dynamo embeds each vLLM-Omni stage in its own single-stage AsyncOmni.
        The embedded orchestrator must still finish locally at stage 0, but the
        EngineCoreRequest needs the global final stage metadata so vLLM-Omni
        emits multimodal payloads for downstream Dynamo stages.
        """
        if final_stage_id is None or final_stage_id <= self.stage_id:
            return prompt
        if isinstance(prompt, OmniEngineCoreRequest):
            return self._apply_final_stage_metadata(prompt, final_stage_id)

        engine_core = getattr(self.engine, "engine", None)
        build_message = getattr(engine_core, "_build_add_request_message", None)
        if build_message is None:
            logger.warning(
                "Stage %d: cannot prebuild EngineCoreRequest with final_stage_id=%s",
                self.stage_id,
                final_stage_id,
            )
            return prompt

        msg = build_message(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )
        return msg.get("prompt", prompt)

    def _apply_final_stage_metadata(
        self, prompt: Any, final_stage_id: int | None
    ) -> Any:
        if final_stage_id is None or final_stage_id <= self.stage_id:
            return prompt
        return _apply_omni_final_stage_metadata(prompt, final_stage_id)

    def _fetch_stage_inputs(
        self,
        stage_connector_refs: dict[int, Any],
        request_id: str,
        original_prompt: Any = None,
    ) -> list[_Proxy]:
        """Fetch previous stage outputs from connectors for the processor/engine.

        Fetches only the stages listed in engine_input_source (or all refs if empty).
        Returns a sparse list indexed by original stage id.
        Raises RuntimeError on any failure so the caller can propagate it as an error chunk.
        """
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        stage_list = [
            _Proxy(original_prompt=original_prompt)
            for _ in range(max(sources, default=-1) + 1)
        ]
        for stage_k in sources:
            if (meta_k := stage_connector_refs.get(stage_k)) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector ref for source stage {stage_k}"
                )
            if (
                connector := self.connectors.get(_connector_key(stage_k, self.stage_id))
            ) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector for edge ({stage_k}→{self.stage_id})"
                )
            try:
                payload = connector.get(
                    str(stage_k), str(self.stage_id), request_id, metadata=meta_k
                )
            except Exception as e:
                raise RuntimeError(
                    f"Stage {self.stage_id}: connector.get() failed: {e}"
                ) from e
            payload_data = payload[0] if isinstance(payload, tuple) else payload
            if not payload_data:
                raise RuntimeError(
                    f"Stage {self.stage_id}: empty payload from connector ({stage_k}→{self.stage_id})"
                )
            engine_inputs = (
                payload_data.get("engine_inputs")
                if isinstance(payload_data, dict)
                else payload_data
            )
            _ensure_cumulative_token_ids(engine_inputs)
            stage_list[stage_k] = _Proxy(
                engine_outputs=[engine_inputs],
                original_prompt=original_prompt,
            )
        return stage_list

    def _run_processor(
        self,
        stage_list: list[_Proxy],
        stage_connector_refs: dict[int, Any],
        original_prompt: Any,
    ) -> Any:
        """Invoke vLLM-Omni v0.20 stage processors."""
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        source_outputs = [stage_list[stage_id] for stage_id in sources]
        return self._processor(
            source_outputs,
            original_prompt,
            self._requires_mm,
            None,
        )


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Initialize a single omni stage worker.

    Mirrors init_omni() setup pattern exactly to avoid routing/handler issues.
    """
    if config.stage_id is None:
        raise ValueError("--stage-id is required for stage worker initialization")
    stage_id: int = config.stage_id
    stage_configs = load_omni_stage_configs(config.model, config.stage_configs_path)
    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]
    stage_type: str = getattr(my_config, "stage_type", "llm")

    # Stage worker registers at {ns}.{model_stage}.generate — NOT {ns}.backend.generate.
    # Router registers at {ns}.backend.generate and discovers workers by model_stage.
    model_stage = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")
    generate_endpoint = runtime.endpoint(f"{config.namespace}.{model_stage}.generate")
    shutdown_endpoints[:] = [generate_endpoint]

    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)
    _, connectors = initialize_orchestrator_connectors(config.stage_configs_path)  # type: ignore[arg-type]
    connectors = ensure_omni_stage_connectors(stage_configs, connectors)

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        output_modalities=config.output_modalities,
        default_video_fps=config.default_video_fps,
        stage_id=stage_id,
    )

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Stage %d: non-leader DP rank %d; waiting for shutdown",
            stage_id,
            config.engine_args.data_parallel_rank,
        )
        if shutdown_event is not None:
            await shutdown_event.wait()
        return

    logger.info(
        "Stage %d: serving internal stage endpoint '%s' (not registering model)",
        stage_id,
        generate_endpoint,
    )
    health_check_payload = (
        await VllmOmniHealthCheckPayload.create(engine)  # type: ignore[arg-type]
    ).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            worker.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (
                    prometheus_names.labels.MODEL,
                    config.served_model_name or config.model,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    config.served_model_name or config.model,
                ),
            ],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logger.error("Stage %d: endpoint failed: %s", stage_id, e)
        raise


def _connector_key(from_stage: int | str, to_stage: int | str) -> tuple[str, str]:
    """Build the connector dict key used by initialize_orchestrator_connectors."""
    return (str(from_stage), str(to_stage))


def ensure_omni_stage_connectors(
    stage_configs: list[Any],
    connectors: dict[tuple[str, str], Any],
) -> dict[tuple[str, str], Any]:
    """Add default SHM connectors for v0.20 deploy config stage dependencies."""
    resolved = dict(connectors)
    for idx, stage_config in enumerate(stage_configs):
        to_stage = getattr(stage_config, "stage_id", idx)
        for from_stage in getattr(stage_config, "engine_input_source", []) or []:
            edge = _connector_key(from_stage, to_stage)
            if edge in resolved:
                continue
            resolved[edge] = OmniConnectorFactory.create_connector(
                ConnectorSpec(
                    name="SharedMemoryConnector",
                    extra={"shm_threshold_bytes": 65536},
                )
            )
    return resolved


def load_omni_stage_configs(model: str, stage_configs_path: str | None) -> list[Any]:
    """Load vLLM-Omni v0.20 deploy configs."""
    if stage_configs_path is None:
        raise ValueError("--stage-configs-path is required")
    return load_stage_configs_from_model(model, deploy_config_path=stage_configs_path)


def _load_processor(func_path: str | None) -> Any:
    """Load a processor function from a dotted module path, or return None."""
    if not func_path:
        return None
    module_path, func_name = func_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), func_name)


def _ensure_cumulative_token_ids(result: Any) -> None:
    """Expose token_ids in the cumulative field used by vLLM-Omni processors."""
    for output in getattr(result, "outputs", []) or []:
        if not hasattr(output, "cumulative_token_ids") and hasattr(output, "token_ids"):
            output.cumulative_token_ids = list(output.token_ids)
        _ensure_completion_multimodal_output(result, output)


def _has_multimodal_stage_payload(result: Any) -> bool:
    """Return true when a chunk carries latent payload for a downstream stage."""
    multimodal_output = getattr(result, "multimodal_output", None)
    if isinstance(multimodal_output, dict) and multimodal_output:
        return True
    for output in getattr(result, "outputs", []) or []:
        multimodal_output = getattr(output, "multimodal_output", None)
        if isinstance(multimodal_output, dict) and multimodal_output:
            return True
    return False


def _ensure_completion_multimodal_output(result: Any, output: Any) -> None:
    """Make request-level multimodal payload visible to v0.20 processors."""
    request_mm = getattr(result, "multimodal_output", None)
    if not isinstance(request_mm, dict) or not request_mm:
        return

    output_mm = getattr(output, "multimodal_output", None)
    if not isinstance(output_mm, dict) or not output_mm:
        setattr(output, "multimodal_output", dict(request_mm))
        return

    for key, value in request_mm.items():
        output_mm.setdefault(key, value)


def _prompt_token_ids_from_prompt(prompt: Any) -> list[int] | None:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else None
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
        return list(token_ids) if token_ids else None
    token_ids = getattr(prompt, "prompt_token_ids", None)
    return list(token_ids) if token_ids else None


def _create_engine(model: str, stage_config: Any, stage_type: str) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML."""
    single_stage_config = {
        "stage_args": [_stage_config_to_dict(stage_config, stage_type)],
        "runtime": {"edges": []},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    try:
        return AsyncOmni(model=model, stage_configs_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def _stage_config_to_dict(stage_config: Any, stage_type: str) -> dict:
    """Convert a parsed stage config to a single-stage YAML dict."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    def _to_plain(obj: Any) -> Any:
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        return obj

    engine_args = _to_plain(stage_config.engine_args)
    if isinstance(engine_args, dict):
        # Dynamo runs each vLLM-Omni stage in its own AsyncOmni and owns
        # inter-stage transfer through the router/connectors. vLLM-Omni
        # async_chunk expects one native orchestrator to pre-submit downstream
        # stages, so single-stage workers must emit sync handoff outputs.
        engine_args["async_chunk"] = False

    result: dict = {
        "stage_id": 0,
        "stage_type": stage_type,
        "engine_args": engine_args,
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    for key in ("default_sampling_params", "is_comprehension"):
        val = getattr(stage_config, key, None)
        if val is not None:
            result[key] = _to_plain(val)

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt["devices"] = "0"
        result["runtime"] = rt

    return result


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
