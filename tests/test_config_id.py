"""A declared ``AxisArray.config_id`` replaces the per-message layout hash; an undeclared one falls back to it."""

import asyncio
import dataclasses
import logging

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, chain_config_id, compute_config_id, replace

from ezmsg.baseproc import BaseStatefulTransformer

WARN_LOGGER = "ezmsg.baseproc.stateful"


class Settings(ez.Settings):
    pass


@dataclasses.dataclass
class State:
    resets: int = 0
    segment_resets: int = 0


class Counting(BaseStatefulTransformer[Settings, AxisArray, AxisArray, State]):
    hashes = 0

    def _hash_message(self, message) -> int:
        self.hashes += 1
        return super()._hash_message(message)

    def _reset_state(self, message) -> None:
        self._state.resets += 1

    def _reset_segment(self, message) -> None:
        self._state.segment_resets += 1

    def _process(self, message):
        return message


class IgnoresChannels(Counting):
    def _hash_message(self, message) -> int:
        self.hashes += 1
        return self._message_hash(message, exclude_dims=["ch"])


def msg(i: int = 0, labels=("a", "b"), segment: int = 0, declared: bool = True) -> AxisArray:
    m = AxisArray(
        np.zeros((10, len(labels))),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=i * 0.1),
            "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key="k",
        stream_dim="time",
        segment=segment,
    )
    if declared:
        m.config_id = compute_config_id(m)
    return m


@pytest.fixture(params=["sync", "async"])
def run(request):
    def _run(proc, message):
        return proc(message) if request.param == "sync" else asyncio.run(proc.__acall__(message))

    return _run


def counts(proc) -> tuple[int, int]:
    return proc.state.resets, proc.state.segment_resets


def test_declared_unchanged_is_never_hashed(run):
    proc = Counting(Settings())
    for i in range(20):
        run(proc, msg(i))
    assert proc.hashes == 1
    assert counts(proc) == (1, 0)


def test_config_change_with_a_layout_change_resets(run):
    proc = Counting(Settings())
    run(proc, msg(0))
    run(proc, msg(1, labels=("a", "c")))
    run(proc, msg(2, labels=("a", "c")))
    assert counts(proc) == (2, 0)
    assert proc.hashes == 2


def test_config_change_the_hash_ignores_does_not_reset(run):
    proc = IgnoresChannels(Settings())
    run(proc, msg(0))
    run(proc, msg(1, labels=("a", "c")))
    run(proc, msg(2, labels=("a", "c")))
    assert counts(proc) == (1, 0)
    assert proc.hashes == 2  # hashed once per distinct config_id


def test_config_change_the_hash_ignores_still_honours_a_new_segment(run):
    proc = IgnoresChannels(Settings())
    run(proc, msg(0))
    run(proc, msg(1, labels=("a", "c"), segment=1))
    assert counts(proc) == (1, 1)


@pytest.mark.parametrize("declared", [True, False])
def test_a_new_segment_resets_the_segment(run, declared):
    proc = Counting(Settings())
    for i, segment in enumerate([0, 0, 1, 1, 2]):
        run(proc, msg(i, segment=segment, declared=declared))
    assert counts(proc) == (1, 2)


def test_request_reset_forces_a_reset_on_an_unchanged_stream(run):
    proc = Counting(Settings())
    run(proc, msg(0))
    proc._request_reset()
    run(proc, msg(1))
    proc._hash = -1  # what other code assigns directly
    run(proc, msg(2))
    run(proc, msg(3))
    assert counts(proc) == (3, 0)


def test_restored_state_is_checked_against_the_stream(run):
    a, b = Counting(Settings()), Counting(Settings())
    a(msg(0))
    b(msg(0, labels=("x", "y")))
    b_resets = b.state.resets
    b.stateful_op((b.state, a._hash), msg(1, labels=("x", "y")))
    assert b.state.resets == b_resets + 1


def test_undeclared_hashes_every_message_and_resets_on_a_layout_change(run):
    proc = Counting(Settings())
    for i in range(3):
        run(proc, msg(i, declared=False))
    run(proc, msg(3, labels=("a", "c"), declared=False))
    assert proc.hashes == 4
    assert counts(proc) == (2, 0)


def test_undeclared_warns_once_per_class(run, caplog):
    class Fresh(Counting):
        pass

    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        for proc in (Fresh(Settings()), Fresh(Settings())):
            for i in range(5):
                run(proc, msg(i, declared=False))
        run(Counting(Settings()), msg(0))  # declared: no warning
    warnings = [r for r in caplog.records if r.name == WARN_LOGGER]
    assert len(warnings) == 1
    assert "Fresh" in warnings[0].getMessage() and "key='k'" in warnings[0].getMessage()


def test_non_axisarray_does_not_warn(run, caplog):
    class Plain(Counting):
        pass

    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        proc = Plain(Settings())
        for i in range(3):
            run(proc, {"x": i})
    assert not [r for r in caplog.records if r.name == WARN_LOGGER]
    assert counts(proc) == (1, 0)


def test_undeclared_warning_can_be_switched_off_per_class(run, caplog):
    class Stamper(Counting):
        WARN_UNDECLARED_CONFIG = False

    class Fresh(Counting):
        pass

    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        for i in range(3):
            run(Stamper(Settings()), msg(i, declared=False))
        run(Fresh(Settings()), msg(0, declared=False))
    warnings = [r.getMessage() for r in caplog.records if r.name == WARN_LOGGER]
    assert len(warnings) == 1 and "Fresh" in warnings[0]


class OtherBackendArray(np.ndarray):
    """Stands in for an array from another library (mlx, torch, ...); only its module name matters."""

    __module__ = "otherlib.core"


def test_undeclared_resets_on_a_backend_change(run):
    proc = Counting(Settings())
    run(proc, msg(0, declared=False))
    other = msg(1, declared=False)
    run(proc, AxisArray(other.data.view(OtherBackendArray), other.dims, other.axes, key="k", stream_dim="time"))
    assert counts(proc) == (2, 0)


def test_undeclared_resets_on_a_dtype_change(run):
    proc = Counting(Settings())
    run(proc, msg(0, declared=False))
    run(proc, msg(1, declared=False))
    other = msg(2, declared=False)
    run(proc, replace(other, data=other.data.astype(np.float32)))
    assert counts(proc) == (2, 0)


def test_switching_between_declared_and_undeclared_is_a_change(run):
    proc = Counting(Settings())
    run(proc, msg(0))
    run(proc, msg(1, labels=("a", "c"), declared=False))  # undeclared: hashed, layout changed
    assert counts(proc) == (2, 0)
    run(proc, msg(2))  # declared again with the first id: hashed, layout changed back
    assert counts(proc) == (3, 0)
    run(proc, msg(3, declared=False))  # same layout, now undeclared: id -> layout hash, resets
    run(proc, msg(4))  # declared again: layout hash -> id, resets
    assert counts(proc) == (5, 0)
    assert proc.hashes == 5


def test_default_resets_on_a_dtype_only_config_change(run):
    proc = Counting(Settings())
    a = msg(0)
    b = msg(1)
    b = replace(b, data=b.data.astype(np.complex128))
    b.config_id = compute_config_id(b)
    assert a.config_id != b.config_id
    run(proc, a)
    run(proc, b)
    assert counts(proc) == (2, 0)


def test_default_resets_on_a_folded_in_value(run):
    proc = Counting(Settings())
    a = msg(0)
    b = msg(1)
    b.config_id = chain_config_id(b.config_id, 2.0)  # e.g. a source's value scaling
    run(proc, a)
    run(proc, b)
    assert counts(proc) == (2, 0)


def test_a_narrowing_override_vetoes_those_resets(run):
    proc = IgnoresChannels(Settings())
    a = msg(0)
    b = msg(1, labels=("x", "y"))  # a relabel: IgnoresChannels excludes "ch"
    c = msg(2)
    c.config_id = chain_config_id(c.config_id, 2.0)
    for m in (a, b, c):
        run(proc, m)
    assert counts(proc) == (1, 0)
    assert proc.hashes == 3


def test_default_unchanged_id_does_not_reset_or_hash(run):
    proc = Counting(Settings())
    for i in range(5):
        m = msg(i)
        m.config_id = chain_config_id(m.config_id, 2.0)
        run(proc, m)
    assert counts(proc) == (1, 0)
    assert proc.hashes == 1
