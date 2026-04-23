# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the worker_type derivation helper.

Verifies the uniform mapping of (DisaggregationMode, route_to_encoder) to
(WorkerType, needs), including the Aggregated = Prefill | Decode alias
semantics and the rejection of nonsensical combinations. Also exercises
the WorkerType pyclass exposed by `dynamo._core` so the binding-level
equality contract is locked in.
"""

import pytest

from dynamo._core import WorkerType
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.worker_type import derive_worker_type_and_needs

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


def test_aggregated_no_encode():
    wt, needs = derive_worker_type_and_needs(DisaggregationMode.AGGREGATED)
    assert wt == WorkerType.Aggregated
    assert needs.is_empty()


def test_aggregated_with_encode():
    # The E-PD deployment pattern: aggregated PD worker + separate encode worker.
    wt, needs = derive_worker_type_and_needs(
        DisaggregationMode.AGGREGATED, route_to_encoder=True
    )
    assert wt == WorkerType.Aggregated
    assert needs == WorkerType.Encode


def test_prefill_no_encode():
    wt, needs = derive_worker_type_and_needs(DisaggregationMode.PREFILL)
    assert wt == WorkerType.Prefill
    assert needs == WorkerType.Decode


def test_prefill_with_encode():
    wt, needs = derive_worker_type_and_needs(
        DisaggregationMode.PREFILL, route_to_encoder=True
    )
    assert wt == WorkerType.Prefill
    assert needs == WorkerType.Decode | WorkerType.Encode


def test_decode_no_encode():
    wt, needs = derive_worker_type_and_needs(DisaggregationMode.DECODE)
    assert wt == WorkerType.Decode
    assert needs == WorkerType.Prefill


def test_decode_with_encode():
    # Code supports it; no shipping example uses it today.
    wt, needs = derive_worker_type_and_needs(
        DisaggregationMode.DECODE, route_to_encoder=True
    )
    assert wt == WorkerType.Decode
    assert needs == WorkerType.Prefill | WorkerType.Encode


def test_encode_no_encode():
    # Encode needs both Prefill and Decode (satisfied by Aggregated alias in E-PD).
    wt, needs = derive_worker_type_and_needs(DisaggregationMode.ENCODE)
    assert wt == WorkerType.Encode
    assert needs == WorkerType.Prefill | WorkerType.Decode


def test_encode_with_route_to_encoder_rejected():
    # An encoder routing to another encoder is nonsensical.
    with pytest.raises(ValueError, match="route-to-encoder"):
        derive_worker_type_and_needs(DisaggregationMode.ENCODE, route_to_encoder=True)


def test_aggregated_alias_matches_prefill_or_decode():
    # Load-bearing invariant: Aggregated == Prefill | Decode at the bit level.
    # This is what makes Encode.needs = Prefill | Decode satisfiable by a single
    # aggregated worker (E-PD pattern) via plain bitwise AND.
    wt_agg, _ = derive_worker_type_and_needs(DisaggregationMode.AGGREGATED)
    assert wt_agg == WorkerType.Prefill | WorkerType.Decode
    assert wt_agg.is_aggregated()
    assert wt_agg.contains_prefill()
    assert wt_agg.contains_decode()


# -- WorkerType pyclass equality guards --
#
# Without `#[pyclass(eq)]` on the binding, `==` between distinct Python
# instances of the same bitflag value falls back to identity comparison and
# silently returns False, breaking the Aggregated = Prefill | Decode alias.
# These tests assert the equality contract at the binding level so any
# regression is caught in CI rather than at live-integration time.


def test_worker_type_aggregated_alias_holds_in_python():
    assert WorkerType.Aggregated == WorkerType.Prefill | WorkerType.Decode
    assert WorkerType.Prefill | WorkerType.Decode == WorkerType.Aggregated


def test_worker_type_equality_and_helpers():
    assert WorkerType.Prefill == WorkerType.Prefill
    assert WorkerType.Prefill != WorkerType.Decode
    assert WorkerType.empty().is_empty()
    assert WorkerType.Aggregated.is_aggregated()
    assert WorkerType.Prefill.is_canonical()
    assert WorkerType.Aggregated.is_canonical()
    # Non-canonical combinations are rejected by is_canonical.
    assert not (WorkerType.Prefill | WorkerType.Encode).is_canonical()
    assert not WorkerType.empty().is_canonical()
