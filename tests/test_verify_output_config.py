"""A stage whose output layout changes without a new ``config_id`` gets one from baseproc, with a warning."""

import asyncio
import dataclasses
import logging

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, chain_config_id, compute_config_id, replace

from ezmsg.baseproc import BaseAsyncTransformer, BaseStatefulTransformer

WARN_LOGGER = "ezmsg.baseproc.stateful"


class Settings(ez.Settings):
    labels: tuple = ("a", "b")


@dataclasses.dataclass
class State:
    pass


class Relabels(BaseStatefulTransformer[Settings, AxisArray, AxisArray, State]):
    """Forgets to set a new config_id when its settings relabel the output channels."""

    def _reset_state(self, message) -> None:
        pass

    def _process(self, message):
        self.last = replace(
            message, axes={**message.axes, "ch": AxisArray.CoordinateAxis(np.array(self.settings.labels), dims=["ch"])}
        )
        return self.last


class WellBehaved(Relabels):
    def _process(self, message):
        out = super()._process(message)
        self.last = replace(out, config_id=chain_config_id(message.config_id, self.settings.labels))
        return self.last


class AsyncRelabels(BaseAsyncTransformer[Settings, AxisArray, AxisArray, State]):
    def _reset_state(self, message) -> None:
        pass

    async def _aprocess(self, message):
        return Relabels._process(self, message)


def msg(i: int = 0, declared: bool = True) -> AxisArray:
    m = AxisArray(
        np.zeros((10, 2)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=i * 0.1),
            "ch": AxisArray.CoordinateAxis(data=np.array(["x", "y"]), dims=["ch"]),
        },
        key="k",
        stream_dim="time",
    )
    if declared:
        m.config_id = compute_config_id(m)
    return m


@pytest.fixture(params=["sync", "async"])
def run(request):
    def _run(proc, message):
        return proc(message) if request.param == "sync" else asyncio.run(proc.__acall__(message))

    return _run


def relabel_ids(run, proc, labels_seq, declared=True) -> list[int]:
    """config_id of the output after each settings change in ``labels_seq`` (two messages each)."""
    ids = []
    for n, labels in enumerate(labels_seq):
        proc.update_settings(Settings(labels=labels))
        first = run(proc, msg(2 * n, declared))
        assert run(proc, msg(2 * n + 1, declared)).config_id == first.config_id
        ids.append(first.config_id)
    return ids


def unannounced(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == WARN_LOGGER and "without a new config_id" in r.getMessage()]


def test_unannounced_change_is_corrected_and_warned_once(run, caplog):
    class Fresh(Relabels):
        pass

    labels_seq = [("a", "b"), ("c", "d"), ("e", "f"), ("a", "b")]
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        ids = relabel_ids(run, Fresh(Settings()), labels_seq)
        ids2 = relabel_ids(run, Fresh(Settings()), labels_seq)
    assert ids[0] == msg().config_id
    assert len(set(ids)) == len(ids)  # every change, even back to the original labels, gets a new id
    assert ids2 == ids
    records = unannounced(caplog)
    assert len(records) == 1
    text = records[0].getMessage()
    assert "Fresh" in text and "key='k'" in text and "chain_config_id" in text


def test_well_behaved_is_left_alone(run, caplog):
    proc = WellBehaved(Settings())
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        for n, labels in enumerate([("a", "b"), ("c", "d"), ("a", "b")]):
            proc.update_settings(Settings(labels=labels))
            out = run(proc, msg(n))
            assert out is proc.last
            assert out.config_id == chain_config_id(msg().config_id, labels)
    assert not unannounced(caplog)
    assert proc._out_changes == 0


def test_upstream_change_is_passed_on(run):
    proc = Relabels(Settings())
    up = msg()
    assert run(proc, up).config_id == up.config_id
    up2 = replace(msg(1), config_id=up.config_id + 1)
    assert run(proc, up2).config_id == up2.config_id

    proc.update_settings(Settings(labels=("c", "d")))
    corrected = run(proc, up2).config_id
    assert corrected not in (up.config_id, up2.config_id)
    up3 = replace(msg(2), config_id=up.config_id + 2)
    after = run(proc, up3).config_id
    assert after not in (up.config_id, up2.config_id, up3.config_id, corrected)


def test_undeclared_is_untouched(run, caplog):
    proc = Relabels(Settings())
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        assert relabel_ids(run, proc, [("a", "b"), ("c", "d")], declared=False) == [0, 0]
    assert not unannounced(caplog)
    assert proc._out_hash is None


def test_opt_out(run, caplog):
    class Trusted(Relabels):
        VERIFY_OUTPUT_CONFIG = False

    proc = Trusted(Settings())
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        assert relabel_ids(run, proc, [("a", "b"), ("c", "d")]) == [msg().config_id] * 2
    assert not unannounced(caplog)
    assert proc._out_hash is None


def test_survives_state_restore(run):
    proc = Relabels(Settings())
    before = run(proc, msg()).config_id
    proc.update_settings(Settings(labels=("c", "d")))
    proc.state = State()
    assert run(proc, msg(1)).config_id != before


class OtherBackendArray(np.ndarray):
    """Stands in for an array from another library (mlx, torch, ...); only its module name matters."""

    __module__ = "otherlib.core"


class SwitchesBackend(Relabels):
    """Forgets to set a new config_id when it starts emitting another array library's arrays."""

    other = False

    def _process(self, message):
        return replace(message, data=message.data.view(OtherBackendArray)) if self.other else message


def test_unannounced_backend_change_is_corrected(run, caplog):
    class Fresh(SwitchesBackend):  # the warning is once per class
        pass

    proc = Fresh(Settings())
    before = run(proc, msg()).config_id
    proc.other = True
    proc._request_reset()  # the output is checked after a reset
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        assert run(proc, msg(1)).config_id != before
    assert len(unannounced(caplog)) == 1


class SwitchesDtype(Relabels):
    """Forgets to set a new config_id when it starts emitting float32."""

    narrow = False

    def _process(self, message):
        return replace(message, data=message.data.astype(np.float32)) if self.narrow else message


def test_unannounced_dtype_change_is_corrected_and_warned_once(run, caplog):
    class Fresh(SwitchesDtype):  # the warning is once per class
        pass

    proc = Fresh(Settings())
    before = run(proc, msg()).config_id
    proc.narrow = True
    proc._request_reset()  # the output is checked after a reset
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        after = [run(proc, msg(i)).config_id for i in range(1, 4)]
    assert after[0] != before and len(set(after)) == 1
    assert len(unannounced(caplog)) == 1


class ToFloat(Relabels):
    """Always emits float64, whatever its input's dtype: a constant change, not an unannounced one."""

    def _process(self, message):
        return replace(message, data=message.data.astype(np.float64))


def test_constant_dtype_conversion_is_left_alone(run, caplog):
    proc = ToFloat(Settings())
    up = msg()
    up = replace(up, data=up.data.astype(np.int16))
    up.config_id = compute_config_id(up)
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        for _ in range(3):
            out = run(proc, up)
            assert out.data.dtype == np.float64 and out.config_id == up.config_id
    assert not unannounced(caplog)
    assert proc._out_changes == 0


@pytest.mark.parametrize("cls", [Relabels, AsyncRelabels])
def test_sync_and_async_agree(cls):
    labels_seq = [("a", "b"), ("c", "d"), ("e", "f")]
    sync_ids = relabel_ids(lambda p, m: p(m), cls(Settings()), labels_seq)
    async_ids = relabel_ids(lambda p, m: asyncio.run(p.__acall__(m)), cls(Settings()), labels_seq)
    assert sync_ids == async_ids
    assert len(set(sync_ids)) == 3


class Counts(Relabels):
    """Counts how often the output layout is hashed."""

    hashes = 0

    def _message_hash(self, message, *args, **kwargs):
        if message is getattr(self, "last", None):
            self.hashes += 1
        return super()._message_hash(message, *args, **kwargs)


def test_output_is_hashed_only_after_a_reset_or_new_config_id(run):
    proc = Counts(Settings())
    run(proc, msg())
    assert proc.hashes == 1
    for i in range(1, 5):
        run(proc, msg(i))
    assert proc.hashes == 1  # steady state: no hashing

    up = replace(msg(5), config_id=msg().config_id + 1)
    run(proc, up)  # new input id -> reset, and a new output id
    assert proc.hashes == 2
    run(proc, replace(msg(6), config_id=up.config_id))
    assert proc.hashes == 2

    run(proc, replace(msg(7), config_id=up.config_id, segment=1))  # segment reset leaves the layout alone
    assert proc.hashes == 2


def test_unannounced_change_after_request_reset_is_corrected(run, caplog):
    class Fresh(Counts):
        pass

    proc = Fresh(Settings())
    before = run(proc, msg()).config_id
    proc.settings = Settings(labels=("c", "d"))  # bypass update_settings to show _request_reset alone suffices
    proc._request_reset()
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        after = [run(proc, msg(i)).config_id for i in range(1, 4)]
    assert after[0] != before and len(set(after)) == 1
    assert len(unannounced(caplog)) == 1
    assert proc.hashes == 2


class SkipsFirst(Relabels):
    """Emits nothing for the first message after a reset."""

    def _reset_state(self, message) -> None:
        self.skip = True

    def _process(self, message):
        if self.skip:
            self.skip = False
            return None
        return super()._process(message)


def test_reset_with_no_output_checks_the_next_output(run, caplog):
    class Fresh(SkipsFirst):
        pass

    proc = Fresh(Settings())
    assert run(proc, msg()) is None
    before = run(proc, msg(1)).config_id
    proc.settings = Settings(labels=("c", "d"))
    proc._request_reset()
    with caplog.at_level(logging.WARNING, logger=WARN_LOGGER):
        assert run(proc, msg(2)) is None
        assert run(proc, msg(3)).config_id != before
    assert len(unannounced(caplog)) == 1
