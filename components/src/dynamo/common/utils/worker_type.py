# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive ``(worker_type, needs)`` from ``DisaggregationMode`` + modifiers.

The shared helper for DGH-706 topology readiness. Each backend (vLLM, trtllm,
sglang) calls :func:`derive_worker_type_and_needs` during args processing to
compute the bitflag pair that is passed to :func:`dynamo.llm.register_model`.

See ``docs/proposals/health-disagg-readiness.md``. The derivation is uniform
across backends:

    base_needs = {
        AGGREGATED: empty,
        PREFILL:    Decode,
        DECODE:     Prefill,
        ENCODE:     Prefill | Decode,
    }
    needs = base_needs[mode]
    if route_to_encoder:
        needs |= Encode

``--route-to-encoder`` is a modifier flag, not a worker type.
``Encode + --route-to-encoder`` is rejected as nonsensical.
"""

from typing import Tuple

from dynamo._core import WorkerType
from dynamo.common.constants import DisaggregationMode


def derive_worker_type_and_needs(
    mode: DisaggregationMode,
    route_to_encoder: bool = False,
) -> Tuple[WorkerType, WorkerType]:
    """Compute ``(worker_type, needs)`` for the given configuration.

    Args:
        mode: The worker's disaggregation mode.
        route_to_encoder: True if ``--route-to-encoder`` (or equivalent) is set
            — the worker routes multimodal inputs to a separate encode worker.

    Raises:
        ValueError: if ``mode == ENCODE`` and ``route_to_encoder`` is True
            (an encode worker routing to another encoder is nonsensical).

    Returns:
        Tuple of ``(worker_type, needs)`` bitflags ready to pass to
        :func:`dynamo.llm.register_model`.
    """
    if mode == DisaggregationMode.ENCODE and route_to_encoder:
        raise ValueError(
            "--route-to-encoder is not valid with --disaggregation-mode=encode"
        )

    if mode == DisaggregationMode.AGGREGATED:
        worker_type = WorkerType.Aggregated
        needs = WorkerType.empty()
    elif mode == DisaggregationMode.PREFILL:
        worker_type = WorkerType.Prefill
        needs = WorkerType.Decode
    elif mode == DisaggregationMode.DECODE:
        worker_type = WorkerType.Decode
        needs = WorkerType.Prefill
    elif mode == DisaggregationMode.ENCODE:
        worker_type = WorkerType.Encode
        needs = WorkerType.Prefill | WorkerType.Decode
    else:
        raise ValueError(f"unrecognized DisaggregationMode: {mode!r}")

    if route_to_encoder:
        needs = needs | WorkerType.Encode

    return worker_type, needs
