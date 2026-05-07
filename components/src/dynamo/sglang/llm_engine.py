# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
from collections.abc import AsyncGenerator
from typing import Any

import sglang as sgl

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import ModelInput
from dynamo.sglang._compat import (
    NetworkAddress,
    get_local_ip_auto,
    get_scheduler_info,
)
from dynamo.sglang.args import parse_args

logger = logging.getLogger(__name__)

# Mirrors `_warmup_prefill_engine` in init_llm.py — must finish before the
# prefill worker starts serving so the first real request doesn't pay the
# JIT/CUDA-graph compile cost. SGLang's FAKE_BOOTSTRAP_HOST drives the
# warmup through the disagg path without needing a real decode peer.
_PREFILL_WARMUP_TIMEOUT_S = 1800.0

# Operators can opt out of the prefill warmup for fast-iteration / smoke
# environments where the warmup adds avoidable startup latency. The default
# (`0`/unset) keeps warmup on; set to `1`/`true` to skip.
_DYN_SGLANG_SKIP_WARMUP_ENV = "DYN_SGLANG_SKIP_PREFILL_WARMUP"


def _warmup_enabled() -> bool:
    raw = os.environ.get(_DYN_SGLANG_SKIP_WARMUP_ENV, "")
    return raw.strip().lower() not in ("1", "true", "yes", "on")


class SglangLLMEngine(LLMEngine):
    def __init__(self, server_args, dynamo_args, serving_mode: DisaggregationMode):
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        # SGLang historically calls the disagg role `serving_mode`. We keep
        # the field name to match the rest of the SGLang component, but
        # this is the same `DisaggregationMode` used by the unified
        # WorkerConfig and the Rust runtime.
        self.serving_mode = serving_mode
        self.engine = None
        self._bootstrap_host: str | None = None
        self._bootstrap_port: int | None = None
        self._input_param_manager = None
        self._skip_tokenizer_init = server_args.skip_tokenizer_init
        self._use_sglang_tokenizer = dynamo_args.use_sglang_tokenizer
        # Tracks background tasks that consume the prefill engine's stream
        # after the bootstrap chunk has been yielded. The generator must
        # return promptly so the Rust PrefillRouter can advance to decode;
        # the actual KV transfer continues asynchronously inside this task.
        # Tracked here so `cleanup()` can cancel anything still in flight.
        self._prefill_consume_tasks: set[asyncio.Task[Any]] = set()

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SglangLLMEngine, WorkerConfig]:
        config = await parse_args(argv if argv is not None else sys.argv[1:])
        server_args = config.server_args
        dynamo_args = config.dynamo_args

        model_input = (
            ModelInput.Text if dynamo_args.use_sglang_tokenizer else ModelInput.Tokens
        )

        engine = cls(server_args, dynamo_args, config.serving_mode)
        worker_config = WorkerConfig.from_runtime_config(
            dynamo_args,
            model_name=server_args.model_path,
            served_model_name=server_args.served_model_name,
            model_input=model_input,
            disaggregation_mode=config.serving_mode,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        self.engine = sgl.Engine(server_args=self.server_args)

        tokenizer = (
            self.engine.tokenizer_manager.tokenizer
            if not self._skip_tokenizer_init
            else None
        )
        self._input_param_manager = InputParamManager(tokenizer)

        if self.serving_mode == DisaggregationMode.PREFILL:
            # Cache bootstrap host/port now so generate() doesn't have to
            # touch the engine internals on every request, and warm the
            # engine through the disagg path so the first real request
            # doesn't pay the JIT/cuda-graph compile cost.
            self._bootstrap_host, self._bootstrap_port = self._resolve_bootstrap_info()
            if _warmup_enabled():
                await self._warmup_prefill()
            else:
                logger.info(
                    "Skipping SGLang prefill warmup (%s set)",
                    _DYN_SGLANG_SKIP_WARMUP_ENV,
                )

        # Capacity fields -- sourced the same way as register.py in the
        # non-unified path so the Rust runtime gets consistent values.
        total_kv_blocks = None
        scheduler_info = get_scheduler_info(self.engine)
        max_total_tokens = scheduler_info.get("max_total_num_tokens")
        page_size = self.server_args.page_size
        if max_total_tokens and page_size:
            total_kv_blocks = (max_total_tokens + page_size - 1) // page_size

        # Prefer explicit max_prefill_tokens; fall back to max_total_num_tokens
        # from the scheduler so the planner always has a prefill load signal.
        max_num_batched_tokens = (
            getattr(self.server_args, "max_prefill_tokens", None) or max_total_tokens
        )

        return EngineConfig(
            model=self.server_args.model_path,
            served_model_name=self.server_args.served_model_name,
            context_length=self.server_args.context_length,
            kv_cache_block_size=page_size,
            total_kv_blocks=total_kv_blocks,
            max_num_seqs=getattr(self.server_args, "max_running_requests", None),
            max_num_batched_tokens=max_num_batched_tokens,
            # Only populated for prefill workers; the Rust Worker reads
            # these to publish ModelRuntimeConfig.disaggregated_endpoint
            # so the frontend's PrefillRouter can take its optimised
            # Bootstrap path (route decode concurrent with prefill).
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None, "Engine not initialized"

        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        # SGLang disagg works via a bootstrap handshake: prefill and decode
        # workers exchange (host, port, room) and SGLang's NIXL transport
        # pulls the KV cache from prefill to decode using that triple.
        #
        # Two bootstrap paths the Rust PrefillRouter can take:
        #   * Bootstrap path (fast)   -- router resolves bootstrap upfront
        #     from ModelRuntimeConfig.disaggregated_endpoint and writes
        #     it onto BOTH prefill_req.bootstrap_info and
        #     decode_req.bootstrap_info before calling either worker.
        #     Decode runs concurrently with prefill.
        #   * Completed path (slower) -- router waits for prefill to finish,
        #     extracts disaggregated_params from the prefill response,
        #     attaches it to decode_req.prefill_result.
        # We support both: read bootstrap_info first (Bootstrap path),
        # fall back to prefill_result.disaggregated_params (Completed path).
        bootstrap_kwargs: dict[str, Any] = {}
        if self.serving_mode == DisaggregationMode.PREFILL:
            bootstrap_kwargs = self._resolve_prefill_bootstrap(request)
        elif self.serving_mode == DisaggregationMode.DECODE:
            bootstrap_kwargs = self._resolve_decode_bootstrap(request)

        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            rid=context.trace_id,
            **bootstrap_kwargs,
        )

        # ORDER MATTERS for prefill: the engine's async_generate must
        # have registered the bootstrap room (the await above) BEFORE we
        # yield the bootstrap chunk. Otherwise the decode peer can be
        # released by the Rust router and try to connect to a room that
        # doesn't exist yet.
        if self.serving_mode == DisaggregationMode.PREFILL:
            yield {
                "token_ids": [],
                "index": 0,
                "disaggregated_params": dict(bootstrap_kwargs),  # type: ignore[typeddict-unknown-key]
            }
            # Drain the engine stream in a background task so this
            # generator returns immediately. The Rust PrefillRouter
            # waits for the prefill stream to end before forwarding to
            # decode (Completed path); a synchronous drain here would
            # block waiting for decode to pull KV — but decode can't
            # start until the router sees the prefill stream end.
            # That's a deadlock.
            #
            # Background task instead: generator returns after the
            # bootstrap yield, the router advances to decode, decode
            # connects to the bootstrap room, KV transfer completes,
            # background task finishes naturally.
            task = asyncio.create_task(self._consume_prefill_stream(stream, context))
            self._prefill_consume_tasks.add(task)
            task.add_done_callback(self._prefill_consume_tasks.discard)
            return

        async for res in stream:
            # SGLang includes an output index when n>1. Preserve it so the
            # Rust/OpenAI response layer can keep choices separate; default to
            # 0 for legacy/non-n chunks.
            output_idx = res.get("index") or 0
            out: GenerateChunk = {"token_ids": [], "index": output_idx}
            meta_info = res["meta_info"]
            finish_reason = meta_info["finish_reason"]

            output_ids = res.get("output_ids", [])
            if not output_ids and not finish_reason:
                if context.is_stopped():
                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                    completion_tokens = meta_info.get("completion_tokens", 0)
                    yield {
                        "token_ids": [],
                        "index": output_idx,
                        "finish_reason": "cancelled",
                        "completion_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    break
                continue

            out["token_ids"] = output_ids

            if finish_reason:
                prompt_tokens = meta_info["prompt_tokens"]
                completion_tokens = meta_info["completion_tokens"]
                out["finish_reason"] = finish_reason["type"]
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            if context.is_stopped():
                prompt_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                yield {
                    "token_ids": output_ids,
                    "index": output_idx,
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                break

            yield out

    async def abort(self, context: Context) -> None:
        rid = context.trace_id
        if self.engine is not None and rid is not None:
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                self.engine.tokenizer_manager.abort_request(rid=rid, abort_all=False)
                logger.debug("Aborted request %s", rid)

    async def cleanup(self) -> None:
        # Cancel any prefill consume tasks still running. They're
        # background drains of the engine stream — losing them on
        # shutdown is fine; the engine.shutdown() below tears the whole
        # thing down anyway.
        for task in self._prefill_consume_tasks:
            if not task.done():
                task.cancel()
        self._prefill_consume_tasks.clear()

        if self.engine is not None:
            self.engine.shutdown()
            logger.info("SGLang engine shutdown")

    def _resolve_prefill_bootstrap(
        self, request: GenerateRequest
    ) -> dict[str, Any]:
        """Pick the (host, port, room) triple this prefill request will use.

        Priority:
          1. ``request.bootstrap_info`` — the Rust router's Bootstrap path
             populated this from ``ModelRuntimeConfig.disaggregated_endpoint``
             and a router-generated room. Both worker sides see the same
             triple, so the room matches across prefill and decode.
          2. Fall back to the engine's cached host/port plus a locally
             generated room. Used when the Rust router can't take the
             Bootstrap path (no published endpoint, multi-node fallout, etc.).
             The Completed path will forward this triple to the decode
             peer via ``prefill_result.disaggregated_params``.
        """
        assert (
            self._bootstrap_host is not None and self._bootstrap_port is not None
        ), "prefill workers must resolve bootstrap host/port in start()"

        bootstrap_info_from_req = request.get("bootstrap_info") or {}
        if isinstance(bootstrap_info_from_req, dict) and bootstrap_info_from_req:
            host = bootstrap_info_from_req.get("bootstrap_host", self._bootstrap_host)
            port = bootstrap_info_from_req.get("bootstrap_port", self._bootstrap_port)
            room = bootstrap_info_from_req.get("bootstrap_room")
        else:
            host, port, room = self._bootstrap_host, self._bootstrap_port, None

        if room is None:
            room = random.randint(0, 2**63 - 1)
        return {
            "bootstrap_host": host,
            "bootstrap_port": port,
            "bootstrap_room": room,
        }

    @staticmethod
    def _resolve_decode_bootstrap(request: GenerateRequest) -> dict[str, Any]:
        """Pull the bootstrap triple off a decode request.

        The Rust router writes this either to ``request.bootstrap_info``
        (Bootstrap path) or to ``request.prefill_result.disaggregated_params``
        (Completed path). Either is valid; both contain the same fields.
        """
        bootstrap_info = request.get("bootstrap_info")
        if not bootstrap_info:
            prefill_result = request.get("prefill_result")
            if prefill_result is not None:
                bootstrap_info = prefill_result.get("disaggregated_params")

        if not bootstrap_info:
            raise ValueError(
                "decode worker received request without bootstrap info; "
                "expected the Rust PrefillRouter to populate either "
                "bootstrap_info (Bootstrap path) or "
                "prefill_result.disaggregated_params (Completed path)"
            )

        try:
            return {
                "bootstrap_host": bootstrap_info["bootstrap_host"],
                "bootstrap_port": bootstrap_info["bootstrap_port"],
                "bootstrap_room": bootstrap_info["bootstrap_room"],
            }
        except KeyError as e:
            raise ValueError(
                "decode worker received bootstrap info missing required "
                f"field: {e.args[0]} (need host/port/room)"
            ) from e

    @staticmethod
    async def _consume_prefill_stream(
        stream: AsyncGenerator[Any, None], context: Context
    ) -> None:
        """Drain a prefill engine stream after the bootstrap chunk has
        been yielded. Errors are swallowed (best-effort): if the
        decode peer never connects, this loop just exits when the
        engine times out or the request is aborted."""
        try:
            async for _ in stream:
                if context.is_stopped():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "prefill consume task ended with exception", exc_info=True
            )

    def _resolve_bootstrap_info(self) -> tuple[str, int]:
        """Mirrors `BaseWorkerHandler._get_bootstrap_info` — returns the
        `(host, port)` tuple this prefill worker advertises to decode
        peers. Source of truth is the SGLang engine's ServerArgs."""
        assert self.engine is not None
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            dist_init = NetworkAddress.parse(inner_tm.server_args.dist_init_addr)
            bootstrap_host = (
                NetworkAddress(dist_init.resolved().host, bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
        else:
            bootstrap_host = (
                NetworkAddress(get_local_ip_auto(), bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
        return bootstrap_host, bootstrap_port

    async def _warmup_prefill(self) -> None:
        """Drive a warmup request through the disagg path so the first
        client request doesn't pay JIT/CUDA-graph compile time. Uses
        SGLang's FAKE_BOOTSTRAP_HOST so no decode peer is required.
        Failure aborts startup — registering an unwarmed prefill worker
        leads to silent request drops in production."""
        assert self.engine is not None
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        }

        async def _do_warmup() -> None:
            results = await self.engine.async_generate(
                input_ids=[0, 1, 2, 3],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_port=self._bootstrap_port,
                bootstrap_room=999999,
            )
            async for _ in results:
                pass

        logger.info("SGLang prefill warmup starting...")
        await asyncio.wait_for(_do_warmup(), timeout=_PREFILL_WARMUP_TIMEOUT_S)
        logger.info("SGLang prefill warmup complete")

    def _build_sampling_params(self, request: GenerateRequest) -> dict:
        if not self._use_sglang_tokenizer:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "n": sampling_opts.get("n"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
                **self._get_guided_decoding_params(
                    sampling_opts.get("guided_decoding")
                ),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "n": request.get("n"),
                "max_new_tokens": request.get("max_tokens"),
                **self._get_guided_decoding_params(request.get("guided_decoding")),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}

    @staticmethod
    def _get_guided_decoding_params(guided_decoding: object) -> dict:
        if isinstance(guided_decoding, dict):
            json_schema = guided_decoding.get("json")
            if json_schema is not None:
                return {"json_schema": json.dumps(json_schema)}
            structural_tag = guided_decoding.get("structural_tag")
            if structural_tag is not None:
                if hasattr(structural_tag, "model_dump"):
                    structural_tag = structural_tag.model_dump()
                return {"structural_tag": json.dumps(structural_tag)}
        return {}

    def _get_input_param(self, request: GenerateRequest) -> dict:
        assert self._input_param_manager is not None, "Engine not initialized"
        request_input = self._input_param_manager.get_input_param(
            request, use_tokenizer=self._use_sglang_tokenizer
        )
        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }
