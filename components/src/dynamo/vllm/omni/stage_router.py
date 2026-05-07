# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

from dynamo import prometheus_names
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.llm import ModelInput, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.stage_worker import _resolve_model_type, load_omni_stage_configs
from dynamo.vllm.omni.types import StageOutput
from dynamo.vllm.omni.utils import shm_deserialize

logger = logging.getLogger(__name__)


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        self.config = config
        self.stage_configs = load_omni_stage_configs(config.model, stage_configs_path)
        self.stage_clients: Dict[str, Any] = {}

        media_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._formatter = OutputFormatter(
            model_name=config.served_model_name or config.model,
            media_fs=media_fs,
            media_http_url=config.media_output_http_url,
            default_fps=config.default_video_fps,
        )

    def set_stage_client(self, model_stage: str, client: Any) -> None:
        self.stage_clients[model_stage] = client
        logger.info("Registered stage client: %s", model_stage)

    async def generate(
        self,
        request: dict,
        context,  # noqa: ARG002 — context unused; router generates its own request_id
    ) -> AsyncGenerator[dict, None]:
        request_id = str(uuid.uuid4())
        _, request_type = parse_request_type(request, self.config.output_modalities)
        requested_modalities = _requested_modalities(
            request, self.config.output_modalities, request_type
        )
        final_stage_indices = _requested_final_stage_indices(
            self.stage_configs, requested_modalities
        )
        last_stage_idx = (
            max(final_stage_indices)
            if final_stage_indices
            else len(self.stage_configs) - 1
        )

        stage_outputs: List[StageOutput] = []
        for stage_idx, stage_cfg in enumerate(self.stage_configs[: last_stage_idx + 1]):
            model_stage = getattr(
                stage_cfg.engine_args, "model_stage", f"stage{stage_idx}"
            )
            client = self.stage_clients.get(model_stage)
            if client is None:
                yield {
                    "error": f"No client for stage '{model_stage}'",
                    "finished": True,
                }
                return

            if stage_idx == 0:
                # This is a workaround for now to pass in the raw request to stage 0. StageRequest validates it but ignores any unknown keys, so it gets passed through.
                stage_request = {
                    "request_id": request_id,
                    "final_stage_id": last_stage_idx,
                    **request,
                }
            else:
                stage_request = stage_outputs[-1].to_next_stage_request(request_id)

            raw_stage_output = {}
            logger.info(
                "Router: stage %d request keys=%s",
                stage_idx,
                list(stage_request.keys()),
            )
            # For now, it is just one chunk output from the stage. Keeping the loop style in mind if in future we decide to stream multiple chunks from the stage.
            async for chunk in await client.round_robin(stage_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                raw_stage_output.update(data)
            stage_outputs.append(StageOutput.model_validate(raw_stage_output))

            if stage_outputs[-1].error:
                yield {"error": stage_outputs[-1].error, "finished": True}
                return

        # Build formatting context from the original request
        nvext = request.get("nvext") or {}
        fmt_ctx: Dict[str, Any] = {}
        if nvext.get("fps") is not None:
            fmt_ctx["fps"] = nvext["fps"]
        if nvext.get("speed") is not None:
            fmt_ctx["speed"] = nvext["speed"]
        # If the request type is AUDIO_GENERATION,
        # we need to normalize the data_source and response_format to
        # align with other modalities.
        response_format = (
            request.get("data_source")
            if request_type == RequestType.AUDIO_GENERATION
            else request.get("response_format")
        )
        output_format = (
            request.get("response_format")
            if request_type == RequestType.AUDIO_GENERATION
            else request.get("output_format")
        )
        if response_format is not None:
            fmt_ctx["response_format"] = response_format
        if output_format is not None:
            fmt_ctx["output_format"] = output_format

        formatted = False
        for stage_idx, stage_output in enumerate(stage_outputs):
            if not _should_format_stage(
                self.stage_configs[stage_idx],
                requested_modalities,
                fallback=not final_stage_indices and stage_idx == last_stage_idx,
            ):
                continue
            if not stage_output.shm_meta:
                final_output_type = getattr(
                    self.stage_configs[stage_idx], "final_output_type", "unknown"
                )
                yield {
                    "error": (
                        f"No SHM output from final stage {stage_idx} "
                        f"({final_output_type})"
                    ),
                    "finished": True,
                }
                return
            async for chunk in self._format_output(
                stage_output, request_id, request_type, fmt_ctx
            ):
                formatted = True
                yield chunk

        if not formatted:
            yield {"error": "No SHM output from final stage", "finished": True}

    async def _format_output(
        self,
        stage_output: StageOutput,
        request_id: str,
        request_type: RequestType,
        ctx: dict,
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from SHM and format via OutputFormatter."""
        shm_meta = stage_output.shm_meta
        if not shm_meta:
            logger.warning("Router: no shm_meta in stage output")
            return

        result = shm_deserialize(shm_meta)
        chunk = await self._formatter.format(
            result, request_id, request_type=request_type, **ctx
        )
        if chunk:
            yield chunk
        else:
            final_output_type = getattr(result, "final_output_type", "unknown")
            logger.warning(
                "Router: formatter returned None, final_output_type=%s",
                final_output_type,
            )
            yield {
                "error": f"Formatter returned no output for type '{final_output_type}'",
                "finished": True,
            }


def _requested_modalities(
    request: dict,
    configured_modalities: list[str],
    request_type: RequestType,
) -> list[str]:
    """Return output modalities requested by this request."""
    requested = request.get("modalities")
    if isinstance(requested, list) and requested:
        return [str(modality).lower() for modality in requested]
    request_type_value = getattr(request_type, "value", str(request_type))
    if request_type_value == RequestType.IMAGE_GENERATION.value:
        return ["image"]
    if request_type_value == RequestType.VIDEO_GENERATION.value:
        return ["video"]
    if request_type_value == RequestType.AUDIO_GENERATION.value:
        return ["audio"]
    return [str(modality).lower() for modality in configured_modalities or []]


def _requested_final_stage_indices(
    stage_configs: list[Any], requested_modalities: list[str]
) -> list[int]:
    requested = set(requested_modalities)
    return [
        idx
        for idx, stage_config in enumerate(stage_configs)
        if getattr(stage_config, "final_output", False)
        and getattr(stage_config, "final_output_type", None) in requested
    ]


def _should_format_stage(
    stage_config: Any, requested_modalities: list[str], *, fallback: bool = False
) -> bool:
    if fallback:
        return True
    if not getattr(stage_config, "final_output", False):
        return False
    return getattr(stage_config, "final_output_type", None) in set(requested_modalities)


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
) -> None:
    """Initialize OmniStageRouter as a Dynamo backend endpoint."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint or 'generate'}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    router = OmniStageRouter(config, config.stage_configs_path)  # type: ignore[arg-type]

    setup_metrics_collection(config, generate_endpoint, logger)

    # Discover stage endpoints
    for stage_cfg in router.stage_configs:
        model_stage = getattr(
            stage_cfg.engine_args, "model_stage", f"stage{stage_cfg.stage_id}"
        )
        client = await runtime.endpoint(
            f"{config.namespace}.{model_stage}.generate"
        ).client()
        await client.wait_for_instances()
        router.set_stage_client(model_stage, client)

    final_cfg = router.stage_configs[-1]
    final_output_type = getattr(final_cfg, "final_output_type", "image")
    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        model_type = _resolve_model_type(final_output_type)

    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
    )
    logger.info("OmniStageRouter registered at '%s'", generate_endpoint)

    try:
        await generate_endpoint.serve_endpoint(
            router.generate,
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
        )
    except Exception as e:
        logger.error("OmniStageRouter endpoint failed: %s", e)
        raise
