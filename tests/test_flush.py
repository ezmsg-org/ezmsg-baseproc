"""``flush`` releases what a processor holds; transformer units call it when the segment changes."""

import asyncio
import dataclasses
import typing

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    BaseStatefulTransformer,
    BaseTransformer,
    BaseTransformerUnit,
    CompositeProcessor,
)

FS = 100.0


class HoldSettings(ez.Settings):
    n: int = 3


@dataclasses.dataclass
class HoldState:
    held: AxisArray | None = None


def msg(start: int, n: int = 10, segment: int = 0, n_ch: int = 1) -> AxisArray:
    """Samples numbered ``start, start + 1, ...`` so stream order can be read off the data."""
    return AxisArray(
        np.repeat(np.arange(start, start + n, dtype=float)[:, None], n_ch, axis=1),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS, offset=start / FS)},
        stream_dim="time",
        segment=segment,
    )


def at(message: AxisArray, data: np.ndarray, offset: float) -> AxisArray:
    return replace(message, data=data, axes={**message.axes, "time": replace(message.axes["time"], offset=offset)})


def samples(message: AxisArray | None) -> list[float]:
    return [] if message is None else message.data[:, 0].tolist()


class HoldBack(BaseStatefulTransformer[HoldSettings, AxisArray, AxisArray, HoldState]):
    """Emits all but the last ``n`` samples it has seen; ``flush`` releases those too."""

    def _reset_state(self, message: AxisArray) -> None:
        self._state.held = None

    def _process(self, message: AxisArray) -> AxisArray:
        held = self._state.held
        data = message.data if held is None else np.concatenate([held.data, message.data])
        offset = (message if held is None else held).axes["time"].offset
        n_out = max(len(data) - self.settings.n, 0)
        self._state.held = at(message, data[n_out:], offset + n_out / FS)
        return at(message, data[:n_out], offset)

    def flush(self, message: AxisArray | None = None) -> AxisArray | None:
        out = None if message is None else self(message)
        held, self._state.held = self._state.held, None
        if out is None or held is None:
            return held if out is None else out
        return replace(out, data=np.concatenate([out.data, held.data]))


class Double(BaseTransformer[HoldSettings, AxisArray, AxisArray]):
    """Stateless and without a ``flush`` override."""

    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data * 2)


class PlainDouble:
    """Not a BaseProcessor: has no ``flush`` at all."""

    def __call__(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data * 2)

    async def asend(self, message: AxisArray) -> AxisArray:
        return self(message)


def legacy_double() -> typing.Generator[AxisArray, AxisArray, None]:
    out = None
    while True:
        message = yield out
        out = replace(message, data=message.data * 2)


class TwoHolds(CompositeProcessor[HoldSettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(settings: HoldSettings) -> dict[str, typing.Any]:
        return {"a": HoldBack(n=3), "b": HoldBack(n=2)}


class HoldThenDouble(CompositeProcessor[HoldSettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(settings: HoldSettings) -> dict[str, typing.Any]:
        return {"hold": HoldBack(n=3), "double": Double(), "plain": PlainDouble()}


class HoldThenLegacy(CompositeProcessor[HoldSettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(settings: HoldSettings) -> dict[str, typing.Any]:
        return {"hold": HoldBack(n=3), "legacy": legacy_double()}


def call(proc, mode: str, message: AxisArray) -> AxisArray | None:
    return proc(message) if mode == "sync" else asyncio.run(proc.__acall__(message))


def flush(proc, mode: str, message: AxisArray | None = None) -> AxisArray | None:
    return proc.flush(message) if mode == "sync" else asyncio.run(proc.aflush(message))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_default_flush_holds_nothing(mode):
    proc = Double()
    assert flush(proc, mode) is None
    assert samples(flush(proc, mode, msg(0, n=4))) == samples(proc(msg(0, n=4))) == [0, 2, 4, 6]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_flush_releases_the_held_tail_with_the_old_segment_and_layout(mode):
    proc = HoldBack(n=3)
    assert samples(call(proc, mode, msg(0, n_ch=2))) == list(range(7))
    assert samples(call(proc, mode, msg(10, n_ch=2))) == list(range(7, 17))
    tail = flush(proc, mode)
    assert samples(tail) == [17, 18, 19]
    assert tail.segment == 0 and tail.data.shape == (3, 2)
    assert tail.axes["time"].offset == pytest.approx(17 / FS)
    assert flush(proc, mode) is None  # nothing left


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_composite_flush_releases_every_stage_in_stream_order(mode):
    proc = TwoHolds()
    assert samples(call(proc, mode, msg(0))) == list(range(5))
    # "b" processes the 3 samples "a" releases, then releases its own 2 behind them.
    assert samples(flush(proc, mode)) == list(range(5, 10))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_composite_flush_passes_the_tail_through_stages_without_flush(mode):
    proc = HoldThenDouble()
    assert samples(call(proc, mode, msg(0))) == [4 * i for i in range(7)]
    assert samples(flush(proc, mode)) == [4 * i for i in (7, 8, 9)]
    assert flush(proc, mode) is None


def test_composite_aflush_passes_the_tail_through_a_legacy_generator():
    proc = HoldThenLegacy()
    asyncio.run(proc.__acall__(msg(0)))
    assert samples(asyncio.run(proc.aflush())) == [14, 16, 18]
    assert asyncio.run(proc.aflush()) is None


class HoldUnit(BaseTransformerUnit[HoldSettings, AxisArray, AxisArray, HoldBack]):
    SETTINGS = HoldSettings


class DoubleUnit(BaseTransformerUnit[HoldSettings, AxisArray, AxisArray, Double]):
    SETTINGS = HoldSettings


class AdaptiveHold(BaseAdaptiveTransformer[HoldSettings, AxisArray, AxisArray, HoldState]):
    _reset_state = HoldBack._reset_state
    _process = HoldBack._process
    flush = HoldBack.flush

    def partial_fit(self, message: AxisArray) -> None: ...


class AdaptiveHoldUnit(BaseAdaptiveTransformerUnit[HoldSettings, AxisArray, AxisArray, AdaptiveHold]):
    SETTINGS = HoldSettings


def drive_raw(unit_type, messages: list[AxisArray]) -> list[AxisArray]:
    unit = unit_type(HoldSettings())
    unit.create_processor()

    async def run():
        return [obj for m in messages async for _, obj in unit.on_signal(m)]

    return asyncio.run(run())


def drive(unit_type, messages: list[AxisArray]) -> list[list[float]]:
    return [samples(obj) for obj in drive_raw(unit_type, messages)]


@pytest.mark.parametrize("unit_type", [HoldUnit, AdaptiveHoldUnit])
def test_unit_flushes_before_a_new_segment(unit_type):
    out = drive(unit_type, [msg(0), msg(10), msg(100, segment=1)])
    assert out == [list(range(7)), list(range(7, 17)), [17, 18, 19], list(range(100, 107))]


@pytest.mark.parametrize("unit_type", [HoldUnit, AdaptiveHoldUnit])
def test_unit_flushes_before_a_new_config_id(unit_type):
    out = drive_raw(
        unit_type, [replace(msg(0), config_id=7), replace(msg(10), config_id=7), replace(msg(20), config_id=8)]
    )
    assert [samples(o) for o in out] == [list(range(7)), list(range(7, 17)), [17, 18, 19], list(range(20, 27))]
    assert [o.config_id for o in out] == [7, 7, 7, 8]  # the tail carries the old config


@pytest.mark.parametrize("unit_type", [HoldUnit, AdaptiveHoldUnit])
def test_unit_flushes_once_when_segment_and_config_id_both_change(unit_type):
    out = drive(unit_type, [replace(msg(0), config_id=7), replace(msg(100, segment=1), config_id=8)])
    assert out == [list(range(7)), [7, 8, 9], list(range(100, 107))]


def test_unit_does_not_flush_while_segment_and_config_id_are_unchanged():
    out = drive(HoldUnit, [replace(msg(i), config_id=7) for i in (0, 10, 20)])
    assert out == [list(range(7)), list(range(7, 17)), list(range(17, 27))]


def test_unit_does_not_flush_before_the_first_message():
    assert drive(HoldUnit, [replace(msg(0, segment=3), config_id=7)]) == [list(range(7))]


def test_unit_yields_once_per_message_when_nothing_is_held():
    out = drive(DoubleUnit, [msg(0, n=2), msg(2, n=2), msg(100, n=2, segment=1)])
    assert out == [[0, 2], [4, 6], [200, 202]]
