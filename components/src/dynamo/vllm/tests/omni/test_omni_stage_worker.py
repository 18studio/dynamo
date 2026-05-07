# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageWorker.

No GPU, no vllm_omni — uses mock StageEngine matching AsyncOmni.generate() signature.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from dynamo.vllm.omni.stage_worker import (
        OmniStageWorker,
        _Proxy,
        _stage_config_to_dict,
        ensure_omni_stage_connectors,
        load_omni_stage_configs,
    )
    from dynamo.vllm.omni.utils import _build_sampling_params
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    """Satisfies StageEngine Protocol — matches AsyncOmni interface."""

    engine = None  # satisfies StageEngine.engine

    def __init__(self, output=None):
        self.received_prompt = None
        self.received_request_id = None
        self.received_sampling_params_list = None
        self._output = output or {"output": "mock", "finished": True}

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        self.received_prompt = prompt
        self.received_request_id = request_id
        self.received_sampling_params_list = sampling_params_list

        async def _gen():
            yield self._output

        return _gen()

    async def get_tokenizer(self):
        return None


class _ErrorEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            raise RuntimeError("engine exploded")
            yield  # make it an async generator

        return _gen()


class _MultiChunkEngine(_MockEngine):
    def __init__(self, chunks):
        super().__init__()
        self._chunks = chunks

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        self.received_prompt = prompt
        self.received_request_id = request_id
        self.received_sampling_params_list = sampling_params_list

        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


class _MockContext:
    def id(self):
        return "test-req-id"


def _make_stage_config(**overrides):
    defaults = dict(
        stage_type="llm",
        final_output=False,
        final_output_type="text",
        engine_input_source=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_worker(engine=None, stage_config=None, connectors=None, stage_id=0):
    return OmniStageWorker(
        engine=engine or _MockEngine(),
        stage_config=stage_config or _make_stage_config(),
        connectors=connectors or {},
        stage_id=stage_id,
    )


def test_proxy_forwards_attributes_to_first_engine_output():
    output = SimpleNamespace(outputs=["image"], token_ids=[1, 2, 3])
    proxy = _Proxy(engine_outputs=[output])

    assert proxy.outputs == ["image"]
    assert proxy.token_ids == [1, 2, 3]


def test_proxy_uses_original_prompt_token_ids_when_stage_output_has_none():
    output = SimpleNamespace(outputs=["audio"], prompt_token_ids=[])
    proxy = _Proxy(
        engine_outputs=[output],
        original_prompt={"prompt_token_ids": [151644, 872, 198]},
    )

    assert proxy.prompt_token_ids == [151644, 872, 198]


def test_proxy_prefers_chat_template_prompt_ids_over_raw_stage_ids():
    output = SimpleNamespace(outputs=["audio"], prompt_token_ids=[872, 198, 9707])
    proxy = _Proxy(
        engine_outputs=[output],
        original_prompt={"prompt_token_ids": [151644, 872, 198, 9707]},
    )

    assert proxy.prompt_token_ids == [151644, 872, 198, 9707]


def test_proxy_keeps_stage_prompt_ids_when_already_chat_formatted():
    output = SimpleNamespace(outputs=["audio"], prompt_token_ids=[151644, 872, 198])
    proxy = _Proxy(
        engine_outputs=[output],
        original_prompt={"prompt_token_ids": [151644, 8948, 198, 151644, 872, 198]},
    )

    assert proxy.prompt_token_ids == [151644, 872, 198]


def test_load_omni_stage_configs_uses_v020_model_loader(tmp_path):
    stage_configs_path = tmp_path / "deploy.yaml"
    stage_configs_path.write_text("stages:\n  - stage_id: 0\n", encoding="utf-8")
    expected = [object()]

    with patch(
        "dynamo.vllm.omni.stage_worker.load_stage_configs_from_model",
        return_value=expected,
    ) as deploy_loader:
        assert (
            load_omni_stage_configs("zai-org/GLM-Image", str(stage_configs_path))
            is expected
        )

    deploy_loader.assert_called_once_with(
        "zai-org/GLM-Image", deploy_config_path=str(stage_configs_path)
    )


def test_ensure_omni_stage_connectors_adds_missing_engine_input_edges():
    existing = {("0", "1"): object()}
    created = object()
    stage_configs = [
        _make_stage_config(stage_id=0),
        _make_stage_config(stage_id=1, engine_input_source=[0]),
        _make_stage_config(stage_id=2, engine_input_source=[1]),
    ]

    with patch(
        "dynamo.vllm.omni.stage_worker.OmniConnectorFactory.create_connector",
        return_value=created,
    ) as create_connector:
        connectors = ensure_omni_stage_connectors(stage_configs, existing)

    assert connectors[("0", "1")] is existing[("0", "1")]
    assert connectors[("1", "2")] is created
    create_connector.assert_called_once()


def test_single_stage_config_disables_async_chunk_without_mutating_source():
    engine_args = SimpleNamespace(
        model_stage="thinker",
        async_chunk=True,
        engine_output_type="latent",
    )
    stage_config = _make_stage_config(
        engine_args=engine_args,
        runtime=SimpleNamespace(devices="0"),
        default_sampling_params={"max_tokens": 16},
    )

    result = _stage_config_to_dict(stage_config, "llm")

    assert result["engine_args"]["async_chunk"] is False
    assert result["engine_args"]["engine_output_type"] == "latent"
    assert engine_args.async_chunk is True


@pytest.mark.asyncio
async def test_direct_input_path():
    """Stage 0 direct path: engine receives the full request dict as prompt."""
    engine = _MockEngine()
    worker = _make_worker(engine=engine)
    request = {"engine_inputs": {"prompt": "hello"}, "sampling_params_list": None}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    # Direct path (no request_id, no stage_connector_refs) passes the whole request as prompt.
    assert engine.received_prompt == request
    assert any("shm_meta" in c for c in chunks)


@pytest.mark.asyncio
async def test_stage_connector_refs_input_path():
    """Stage N>0: engine receives output fetched from connector via stage_connector_refs."""
    engine = _MockEngine()
    fetched_prompt = {"prior_token_ids": [1, 2, 3]}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_prompt}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1", "size": 10})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    request = {
        "request_id": "req-1",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0", "size": 5}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    in_connector.get.assert_called_once_with(
        "0", "1", "req-1", metadata={"name": "ref0", "size": 5}
    )
    assert engine.received_prompt == fetched_prompt
    assert len(chunks) == 1
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1", "size": 10}
    assert chunks[0]["stage_connector_refs"]["0"] == {"name": "ref0", "size": 5}
    assert chunks[0]["original_prompt"] == {"prompt": "hello"}
    assert "final_stage_id" not in chunks[0]


@pytest.mark.asyncio
async def test_stage_connector_refs_builds_engine_core_request():
    """Stage N>0 without processor: upstream with .outputs builds OmniEngineCoreRequest."""
    engine = _MockEngine()

    # Mock upstream output that looks like a real RequestOutput (has .outputs[0].token_ids)
    mock_output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[100, 200, 300])],
        prompt_token_ids=[1, 2],
    )

    in_connector = MagicMock()
    in_connector.get.return_value = (
        mock_output  # raw object, not {"engine_inputs": ...}
    )

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
        stage_config=_make_stage_config(
            default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        ),
    )
    # Mock the engine's output_processors for registration
    engine.engine = MagicMock()
    request = {
        "request_id": "req-ecr",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    # The engine should receive an OmniEngineCoreRequest (not the raw dict)
    assert hasattr(engine.received_prompt, "prompt_token_ids")
    assert engine.received_prompt.prompt_token_ids == [100, 200, 300]
    # Output processor should have been registered
    engine.engine.output_processors[0].add_request.assert_called_once()


@pytest.mark.asyncio
async def test_stage_connector_refs_tags_engine_core_request_with_final_stage_id():
    engine = _MockEngine()
    engine.engine = MagicMock()
    mock_output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[100, 200, 300])],
        prompt_token_ids=[1, 2],
    )
    in_connector = MagicMock()
    in_connector.get.return_value = mock_output
    tagged_prompt = object()

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector},
        stage_id=1,
        stage_config=_make_stage_config(
            default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        ),
    )
    request = {
        "request_id": "req-ecr-final",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
        "final_stage_id": 2,
    }

    with patch(
        "dynamo.vllm.omni.stage_worker._apply_omni_final_stage_metadata",
        return_value=tagged_prompt,
    ) as apply_final_stage:
        _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt is tagged_prompt
    apply_final_stage.assert_called_once()
    assert apply_final_stage.call_args.args[1] == 2


@pytest.mark.asyncio
async def test_stage_connector_refs_with_processor():
    """Stage N>0 with processor: v0.20 processor receives source outputs."""
    engine = _MockEngine()
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"diffusion_input": True}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    processor_calls = []

    def mock_processor(
        source_outputs,
        original_prompt,
        requires_multimodal_data,
        streaming_context,
    ):
        processor_calls.append(
            {
                "source_outputs": source_outputs,
                "original_prompt": original_prompt,
                "requires_multimodal_data": requires_multimodal_data,
                "streaming_context": streaming_context,
            }
        )
        return [processed_prompt]

    cfg = _make_stage_config(
        stage_type="llm",
        final_output=False,
        custom_process_input_func=None,
        engine_input_source=[0],
        requires_multimodal_data=False,
    )
    worker = OmniStageWorker(
        engine=engine,
        stage_config=cfg,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    worker._processor = mock_processor

    request = {
        "request_id": "req-proc",
        "original_prompt": {"prompt": "hi", "height": 480},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert len(processor_calls) == 1
    assert processor_calls[0]["source_outputs"][0].engine_outputs == [fetched_output]
    assert processor_calls[0]["original_prompt"] == {"prompt": "hi", "height": 480}
    assert processor_calls[0]["streaming_context"] is None
    assert engine.received_prompt == processed_prompt
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1"}


@pytest.mark.asyncio
async def test_stage0_prebuilds_prompt_with_global_final_stage_id():
    engine = _MockEngine()
    prebuilt_prompt = object()
    engine.engine = MagicMock()
    engine.engine._build_add_request_message.return_value = {"prompt": prebuilt_prompt}

    worker = _make_worker(
        engine=engine,
        stage_id=0,
        stage_config=_make_stage_config(
            default_sampling_params={"temperature": 0.7, "max_tokens": 16},
        ),
    )
    worker._output_modalities = ["text", "audio"]
    request = {
        "request_id": "req-prebuild",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text", "audio"],
        "final_stage_id": 2,
    }

    _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt is prebuilt_prompt
    engine.engine._build_add_request_message.assert_called_once()
    assert (
        engine.engine._build_add_request_message.call_args.kwargs["final_stage_id"] == 2
    )


@pytest.mark.asyncio
async def test_stage0_forwards_prebuilt_prompt_token_ids_to_downstream_stage():
    engine = _MockEngine()
    prebuilt_prompt = SimpleNamespace(prompt_token_ids=[151644, 872, 198, 9707])
    engine.engine = MagicMock()
    engine.engine._build_add_request_message.return_value = {"prompt": prebuilt_prompt}
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref0"})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): out_connector},
        stage_id=0,
        stage_config=_make_stage_config(
            default_sampling_params={"temperature": 0.7, "max_tokens": 16},
        ),
    )
    worker._output_modalities = ["text", "audio"]
    request = {
        "request_id": "req-prebuild-token-ids",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text", "audio"],
        "final_stage_id": 2,
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt is prebuilt_prompt
    assert chunks[0]["original_prompt"]["prompt_token_ids"] == [
        151644,
        872,
        198,
        9707,
    ]


@pytest.mark.asyncio
async def test_processor_empty_output_yields_error_chunk():
    engine = _MockEngine()
    upstream = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11])])
    in_connector = MagicMock()
    in_connector.get.return_value = upstream

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector},
        stage_id=1,
        stage_config=_make_stage_config(engine_input_source=[0]),
    )
    worker._processor = lambda *_args: []
    worker._engine_input_source = [0]
    request = {
        "request_id": "req-empty-processor",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert chunks == [
        {
            "error": (
                "Stage 1: processor produced no engine inputs from upstream "
                "stage output"
            ),
            "finished": True,
        }
    ]
    assert engine.received_prompt is None


@pytest.mark.asyncio
async def test_connector_accumulates_multimodal_chunks_for_stage_handoff():
    import torch

    first_latent_chunk = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[10],
                multimodal_output={
                    "hidden_states": {
                        "layers": {
                            "0": torch.tensor([[1.0], [2.0]]),
                            "24": torch.tensor([[11.0], [12.0]]),
                        }
                    },
                    "embed": {"tts_bos": torch.tensor([[101.0]])},
                },
            )
        ]
    )
    second_latent_chunk = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[10, 11],
                multimodal_output={
                    "hidden_states": {
                        "layers": {
                            "0": torch.tensor([[3.0]]),
                            "24": torch.tensor([[13.0]]),
                        }
                    },
                    "embed": {"tts_bos": torch.tensor([[202.0]])},
                },
            )
        ]
    )
    final_chunk = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[10, 11, 12], text="visible text")]
    )
    engine = _MultiChunkEngine([first_latent_chunk, second_latent_chunk, final_chunk])
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref0"})
    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): out_connector},
        stage_id=0,
        stage_config=_make_stage_config(final_output=True, final_output_type="text"),
    )
    worker._output_modalities = ["text"]
    request = {
        "request_id": "req-text-audio",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch(
        "dynamo.vllm.omni.stage_worker.serialize_obj",
        return_value=b"serialized",
    ) as serialize_obj, patch(
        "dynamo.vllm.omni.stage_worker.shm_write_bytes",
        return_value={"name": "text-shm"},
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    out_connector.put.assert_called_once()
    connector_result = out_connector.put.call_args.args[3]
    connector_output = connector_result.outputs[0]
    assert connector_output.token_ids == [10, 11, 12]
    assert torch.equal(
        connector_output.multimodal_output["hidden_states"]["layers"]["0"],
        torch.tensor([[1.0], [2.0], [3.0]]),
    )
    assert torch.equal(
        connector_output.multimodal_output["hidden_states"]["layers"]["24"],
        torch.tensor([[11.0], [12.0], [13.0]]),
    )
    assert torch.equal(
        connector_output.multimodal_output["embed"]["tts_bos"],
        torch.tensor([[101.0]]),
    )
    assert not hasattr(final_chunk.outputs[0], "multimodal_output")
    serialize_obj.assert_called_once_with(final_chunk)
    assert chunks[0]["stage_connector_refs"]["0"] == {"name": "ref0"}
    assert chunks[0]["shm_meta"] == {"name": "text-shm"}


@pytest.mark.asyncio
async def test_engine_error_yields_error_chunk():
    """Engine raises → yields {error: ..., finished: True}, no crash."""
    worker = _make_worker(engine=_ErrorEngine())
    request = {"engine_inputs": {"prompt": "hello"}}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert any("error" in c for c in chunks)
    assert any(c.get("finished") for c in chunks)


@pytest.mark.asyncio
async def test_connector_put_failure_yields_error():
    """connector.put() returning ok=False → yields error, stops."""
    mock_connector = MagicMock()
    mock_connector.get.return_value = {"engine_inputs": {"x": 1}}
    mock_connector.put.return_value = (False, 0, {})

    worker = _make_worker(
        connectors={("1", "2"): mock_connector},
        stage_id=1,
    )
    request = {
        "request_id": "req-fail",
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }
    with patch.object(
        worker, "_fetch_stage_inputs", return_value=[_Proxy(engine_outputs=[{"x": 1}])]
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert chunks == [{"error": "connector.put() failed", "finished": True}]


# ── _fetch_stage_inputs method unit tests ──────────────────


def _make_worker_at_stage(stage_id, connectors, engine_input_source=None):
    cfg = _make_stage_config(engine_input_source=engine_input_source or [stage_id - 1])
    return OmniStageWorker(
        engine=_MockEngine(),
        stage_config=cfg,
        connectors=connectors,
        stage_id=stage_id,
    )


def test_fetch_stage_inputs_calls_correct_connector():
    meta0 = {"name": "ref0"}
    connector = MagicMock()
    connector.get.return_value = {"engine_inputs": {"tok": [1, 2]}}

    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): connector}, engine_input_source=[0]
    )
    result = worker._fetch_stage_inputs({0: meta0}, "r1")

    connector.get.assert_called_once_with("0", "1", "r1", metadata=meta0)
    assert result is not None
    assert result[0].engine_outputs == [{"tok": [1, 2]}]


def test_fetch_stage_inputs_raises_on_missing_connector():
    worker = _make_worker_at_stage(1, connectors={}, engine_input_source=[0])
    with pytest.raises(RuntimeError, match="no connector for edge"):
        worker._fetch_stage_inputs({0: {"name": "ref0"}}, "r1")


def test_fetch_stage_inputs_raises_on_missing_ref():
    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): MagicMock()}, engine_input_source=[0]
    )
    with pytest.raises(RuntimeError, match="no connector ref"):
        worker._fetch_stage_inputs({}, "r1")  # ref for stage 0 missing


def test_build_sampling_params_user_overrides_yaml_defaults():
    """User overrides applied on top of YAML defaults via setattr; unspecified keys preserved."""
    stage_config = SimpleNamespace(
        stage_type="diffusion",
        default_sampling_params={
            "num_inference_steps": 20,
            "guidance_scale": 5.0,
            "height": 480,
            "width": 832,
        },
    )
    result = _build_sampling_params(
        stage_config,
        {"num_inference_steps": 50},
    )
    assert result is not None
    sp = result[0]
    assert sp.num_inference_steps == 50  # user override wins
    assert sp.guidance_scale == 5.0  # YAML default preserved


def test_build_sampling_params_no_defaults_returns_none():
    """No default_sampling_params on stage_config -> returns None."""
    stage_config = SimpleNamespace(stage_type="llm")
    assert _build_sampling_params(stage_config, None) is None
    assert _build_sampling_params(stage_config, {}) is None


@pytest.mark.asyncio
async def test_image_request_with_default_sampling_params():
    """Image stage with default_sampling_params builds typed params from YAML defaults + overrides."""
    engine = _MockEngine()
    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(
            stage_type="diffusion",
            final_output=True,
            default_sampling_params={
                "num_inference_steps": 20,
                "guidance_scale": 1.5,
                "height": 1024,
                "width": 1024,
            },
        ),
        connectors={},
        stage_id=0,
        output_modalities=["image"],
    )
    request = {
        "request_id": "img-req-1",
        "prompt": "a red apple",
        "size": "1024x1024",
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert not any("error" in c for c in chunks)
    assert engine.received_sampling_params_list is not None


@pytest.mark.asyncio
async def test_chat_sampling_params_do_not_propagate_to_downstream_stages():
    engine = _MockEngine()
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})
    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(
            default_sampling_params={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 2048,
            },
        ),
        connectors={("0", "1"): out_connector},
        stage_id=0,
        output_modalities=["text", "audio"],
    )
    request = {
        "request_id": "chat-req-1",
        "messages": [{"role": "user", "content": "Say hi."}],
        "modalities": ["text", "audio"],
        "max_completion_tokens": 32,
        "temperature": 0.2,
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert not any("error" in c for c in chunks)
    assert engine.received_sampling_params_list[0].max_tokens == 32
    assert engine.received_sampling_params_list[0].temperature == 0.2
    assert "sampling_params_list" not in chunks[0]


@pytest.mark.asyncio
async def test_sampling_params_propagate_in_stage_output():
    """Non-final stage must include sampling_params_list in its output for downstream stages."""
    engine = _MockEngine()
    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": {"latents": [1, 2]}}
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    # Stage 1: non-final, receives stage_connector_refs from stage 0
    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
        stage_config=_make_stage_config(final_output=False),
    )
    request = {
        "request_id": "req-sp",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
        "sampling_params_list": {
            "num_inference_steps": 42,
            "height": 480,
            "width": 832,
        },
        "final_stage_id": 2,
    }

    with patch(
        "dynamo.vllm.omni.stage_worker._build_sampling_params", return_value=None
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert len(chunks) == 1
    assert chunks[0].get("sampling_params_list") == {
        "num_inference_steps": 42,
        "height": 480,
        "width": 832,
    }
    assert chunks[0].get("final_stage_id") == 2
