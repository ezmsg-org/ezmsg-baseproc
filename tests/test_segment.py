"""A change of ``AxisArray.segment`` resets stateful processors; sources number segments from offsets."""

import asyncio
import dataclasses

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id

from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAsyncTransformer,
    BaseClockDrivenProducer,
    BaseClockDrivenUnit,
    BaseProducer,
    BaseProducerUnit,
    BaseStatefulTransformer,
    ClockDrivenSettings,
    ClockDrivenState,
    SegmentTracker,
    end_of_stream,
    find_gaps,
)

FS = 100.0


class Settings(ez.Settings):
    pass


@dataclasses.dataclass
class State:
    resets: int = 0
    segment_resets: int = 0
    slow: int = 0


class Counting(BaseStatefulTransformer[Settings, AxisArray, AxisArray, State]):
    def _reset_state(self, message: AxisArray) -> None:
        self._state.resets += 1

    def _process(self, message: AxisArray) -> AxisArray:
        self._state.slow += 1
        return message


class KeepsSlowState(Counting):
    def _reset_segment(self, message: AxisArray) -> None:
        self._state.segment_resets += 1


class AsyncCounting(BaseAsyncTransformer[Settings, AxisArray, AxisArray, State]):
    def _reset_state(self, message: AxisArray) -> None:
        self._state.resets += 1

    async def _aprocess(self, message: AxisArray) -> AxisArray:
        return message


class AsyncKeepsSlowState(AsyncCounting):
    def _reset_segment(self, message: AxisArray) -> None:
        self._state.segment_resets += 1


class Adaptive(BaseAdaptiveTransformer[Settings, AxisArray, AxisArray, State]):
    def _reset_state(self, message: AxisArray) -> None:
        self._state.resets += 1

    def partial_fit(self, message: AxisArray) -> None: ...

    def _process(self, message: AxisArray) -> AxisArray:
        return message


def msg(segment: int = 0, n_ch: int = 2, i: int = 0) -> AxisArray:
    return AxisArray(
        np.zeros((10, n_ch)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS, offset=i * 10 / FS)},
        stream_dim="time",
        segment=segment,
    )


def test_a_new_segment_resets():
    proc = Counting(Settings())
    for i, segment in enumerate([0, 0, 1, 1, 1, 5, 4]):
        proc(msg(segment, i=i))
    # First message, then 0 -> 1, 1 -> 5 and 5 -> 4: any change counts, not only an increase.
    assert proc.state.resets == 4


def test_a_stream_that_never_breaks_resets_once():
    proc = Counting(Settings())
    for i in range(5):
        proc(msg(i=i))
    assert proc.state.resets == 1


def test_a_layout_change_and_a_new_segment_together_reset_once():
    proc = Counting(Settings())
    proc(msg(0, n_ch=2))
    proc(msg(1, n_ch=3))
    assert proc.state.resets == 2
    proc(msg(1, n_ch=3))  # the segment that arrived with the layout change is not new again
    assert proc.state.resets == 2


def test_an_override_keeps_state_across_a_segment():
    proc = KeepsSlowState(Settings())
    for i, segment in enumerate([0, 0, 1, 2]):
        proc(msg(segment, i=i))
    assert (proc.state.resets, proc.state.segment_resets, proc.state.slow) == (1, 2, 4)


@pytest.mark.parametrize("call", ["sync", "async"])
def test_async_paths_reset_and_honour_a_sync_override(call):
    def run(proc, message):
        return proc(message) if call == "sync" else asyncio.run(proc.__acall__(message))

    plain, keeps = AsyncCounting(Settings()), AsyncKeepsSlowState(Settings())
    for i, segment in enumerate([0, 1, 1]):
        run(plain, msg(segment, i=i))
        run(keeps, msg(segment, i=i))
    assert plain.state.resets == 2
    assert (keeps.state.resets, keeps.state.segment_resets) == (1, 1)


def test_partial_fit_transform_resets_on_a_new_segment():
    proc = Adaptive(Settings())
    for i, segment in enumerate([0, 1]):
        proc.partial_fit_transform(msg(segment, i=i))
    asyncio.run(proc.apartial_fit_transform(msg(2, i=2)))
    assert proc.state.resets == 3


def test_messages_without_a_segment_are_unaffected():
    class Plain(BaseStatefulTransformer[Settings, object, object, State]):
        def _reset_state(self, message) -> None:
            self._state.resets += 1

        def _process(self, message):
            return message

    proc = Plain(Settings())
    for _ in range(3):
        proc("not an AxisArray")
    assert proc.state.resets == 1


class TestFindGaps:
    def test_continuous(self):
        assert find_gaps(np.arange(50) / FS, 1 / FS, 0.5 / FS).size == 0

    def test_forward_jump_and_backward_step(self):
        t = np.arange(50) / FS
        t[20:] += 0.3  # dropped samples
        t[40:] -= 1.0  # clock stepped back
        np.testing.assert_array_equal(find_gaps(t, 1 / FS, 0.5 / FS), [20, 40])

    def test_jitter_within_tolerance_is_not_a_gap(self):
        t = np.arange(500) / FS + np.random.default_rng(0).uniform(-0.002, 0.002, 500)
        assert find_gaps(t, 1 / FS, 0.005).size == 0

    def test_splits_a_chunk(self):
        t = np.r_[np.arange(5), np.arange(5) + 100] / FS
        parts = np.split(np.arange(10), find_gaps(t, 1 / FS, 0.5 / FS))
        assert [p.tolist() for p in parts] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    @pytest.mark.parametrize("n", [0, 1])
    def test_too_short_to_have_a_gap(self, n):
        assert find_gaps(np.zeros(n), 1 / FS, 0.0).size == 0


class TestSegmentTracker:
    def test_contiguous_chunks_stay_in_one_segment(self):
        tracker = SegmentTracker(tolerance=0.5 / FS)
        assert len({tracker.update(i * 10 / FS, 10, 1 / FS) for i in range(5)}) == 1

    def test_a_jump_or_a_step_back_starts_a_segment(self):
        tracker = SegmentTracker(tolerance=0.5 / FS)
        offsets = [0.0, 0.1, 5.0, 5.1, 1.0]
        assert [tracker.update(o, 10, 1 / FS) for o in offsets] == [0, 0, 1, 1, 2]

    def test_jitter_and_drift_do_not_accumulate(self):
        """Chunks stamped by a clock that runs 1 % fast and jitters by 5 ms never trip a 15 ms tolerance."""
        rng = np.random.default_rng(0)
        tracker = SegmentTracker(tolerance=0.015)
        for i in range(2000):
            tracker.update(i * 0.1 * 1.01 + rng.uniform(-0.005, 0.005), 10, 1 / FS)
        assert tracker.segment == 0

    def test_bump_starts_a_segment_without_a_gap(self):
        tracker = SegmentTracker(tolerance=0.5 / FS)
        assert tracker.update(0.0, 10, 1 / FS) == 0
        assert tracker.bump() == 1
        # Whatever the next offset is, it belongs to the segment bump() started.
        assert tracker.update(50.0, 10, 1 / FS) == 1
        assert tracker.update(50.1, 10, 1 / FS) == 1

    def test_a_marker_at_a_known_gap_then_the_data_after_it(self):
        """(a) A reader ends a section before a gap it knows about; the next chunk's offset shows the gap too."""
        tracker = SegmentTracker(tolerance=0.5 / FS)
        before = msg(tracker.update(0.0, 10, 1 / FS))
        eos = tracker.end_of_stream(before)
        assert (before.segment, eos.segment, tracker.update(5.0, 10, 1 / FS)) == (0, 1, 2)

    def test_a_marker_at_the_end_of_the_file(self):
        """(b) Nothing follows the marker."""
        tracker = SegmentTracker(tolerance=0.5 / FS)
        for i in range(3):
            last = msg(tracker.update(i * 10 / FS, 10, 1 / FS), i=i)
        assert (last.segment, tracker.end_of_stream(last).segment, tracker.segment) == (0, 1, 1)

    def test_data_continuing_after_a_marker_is_not_in_its_segment(self):
        """(c) The marker ended its segment: data after it starts the next one even without a gap."""
        tracker = SegmentTracker(tolerance=0.5 / FS)
        before = msg(tracker.update(0.0, 10, 1 / FS))
        eos = tracker.end_of_stream(before)
        after = tracker.update(eos.axes["time"].offset, 10, 1 / FS)
        assert (before.segment, eos.segment, after) == (0, 1, 2)
        assert tracker.update(eos.axes["time"].offset + 10 / FS, 10, 1 / FS) == 2

    def test_bump_after_a_marker_starts_one_segment(self):
        tracker = SegmentTracker(tolerance=0.5 / FS)
        eos = tracker.end_of_stream(msg(tracker.update(0.0, 10, 1 / FS)))
        assert (eos.segment, tracker.bump(), tracker.update(9.0, 10, 1 / FS)) == (1, 2, 2)

    def test_successive_gaps_never_reuse_a_number(self):
        tracker = SegmentTracker(tolerance=0.5 / FS)
        out = []
        for start in (0.0, 5.0, 9.0):
            chunks = [msg(tracker.update(start + i * 10 / FS, 10, 1 / FS), i=i) for i in range(2)]
            out += [m.segment for m in chunks] + [tracker.end_of_stream(chunks[-1]).segment]
        assert out == [0, 0, 1, 2, 2, 3, 4, 4, 5]


class TestEndOfStream:
    def test_ends_a_linear_stream_just_after_its_last_sample(self):
        last = AxisArray(
            np.arange(20.0).reshape(2, 10),
            dims=["ch", "time"],
            axes={
                "ch": AxisArray.CoordinateAxis(np.array(["a", "b"]), dims=["ch"]),
                "time": AxisArray.TimeAxis(fs=FS, offset=1.0),
            },
            attrs={"subject": 1},
            key="src",
            stream_dim="time",
            segment=7,
        )
        last.config_id = compute_config_id(last)
        eos = end_of_stream(last)
        assert eos.data.shape == (2, 0) and eos.data.dtype == last.data.dtype
        assert eos.data.base is None  # its own array, not a view keeping the last buffer alive
        assert eos.axes["time"].offset == pytest.approx(1.0 + 10 / FS)
        assert eos.axes["time"].gain == last.axes["time"].gain and eos.axes["ch"] is last.axes["ch"]
        assert (eos.dims, eos.key, eos.attrs, eos.stream_dim) == (last.dims, last.key, last.attrs, last.stream_dim)
        assert eos.config_id == last.config_id == compute_config_id(eos)
        assert eos.segment == 8

    def test_empties_a_coordinate_stream_axis(self):
        # No stream_dim declared: falls back to "time", as consumers do.
        last = AxisArray(
            np.zeros((3, 2), dtype=np.int16),
            dims=["time", "ch"],
            axes={"time": AxisArray.CoordinateAxis(np.array([0.0, 0.5, 7.0]), dims=["time"], unit="s")},
        )
        eos = end_of_stream(last)
        assert eos.data.shape == (0, 2) and eos.data.dtype == np.int16
        assert eos.axes["time"].data.shape == (0,) and eos.axes["time"].unit == "s"
        assert eos.segment == last.segment + 1

    def test_the_data_after_a_gap_does_not_continue_it(self):
        """A reader keeping its own count continues it from the marker's segment."""
        before = msg(4)
        eos = end_of_stream(before)
        after = msg(eos.segment + 1, i=50)
        assert (before.segment, eos.segment, after.segment) == (4, 5, 6)

    def test_resets_a_stateful_processor_without_a_layout_change(self):
        proc = Counting(Settings())
        last = msg(3)
        proc(last)
        proc(end_of_stream(last))
        assert proc.state.resets == 2


class _Src(BaseProducer[Settings, AxisArray]):
    async def _produce(self) -> AxisArray:
        return end_of_stream(msg(3))


class _SrcUnit(BaseProducerUnit[Settings, AxisArray, _Src]):
    SETTINGS = Settings


class _Clocked(BaseClockDrivenProducer[ClockDrivenSettings, ClockDrivenState]):
    def _reset_state(self, time_axis) -> None: ...

    def _produce(self, n_samples: int, time_axis) -> AxisArray:
        return end_of_stream(msg(3))


class _ClockedUnit(BaseClockDrivenUnit[ClockDrivenSettings, _Clocked]):
    SETTINGS = ClockDrivenSettings


def test_producer_units_publish_the_end_of_stream_message():
    src = _SrcUnit(Settings())
    src.create_producer()
    clocked = _ClockedUnit(ClockDrivenSettings(fs=FS, n_time=10))
    clocked.create_processor()

    async def first(gen):
        out = await gen.__anext__()
        await gen.aclose()
        return out

    for gen in (src.produce(), clocked.on_clock(AxisArray.LinearAxis(gain=0.1, offset=0.0))):
        _, out = asyncio.run(first(gen))
        assert out.data.shape == (0, 2)
