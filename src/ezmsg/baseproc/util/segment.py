"""Detect breaks in a stream's time base and number its segments.

A source stamps every message with
:attr:`~ezmsg.util.messages.axisarray.AxisArray.segment`, and changes the value on
the first message that does not continue the previous one. Stateful processors
reset when they see it change (``BaseStatefulProcessor._reset_segment``), so
gap detection happens once, at the source, instead of in every processor.
"""

import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis, replace

from .streamdim import resolve_stream_dim


def _empty_along(arr, axis: int):
    # zeros_like of a zero-length slice: a new array of the same dtype and device, not a view of arr's buffer.
    return get_namespace(arr).zeros_like(arr[(slice(None),) * axis + (slice(0, 0),)])


def end_of_stream(message: AxisArray) -> AxisArray:
    """The message a source emits after ``message``, its last, to say its stream has ended.

    For a source that knows: a file reader at the end of the file, or at the end of a section
    before a gap it knows about. Downstream units see the new ``segment`` and flush what they
    hold now, instead of when (or if) more data arrives.

    The result has ``message``'s layout, ``config_id``, key and attrs, but no samples along its
    stream dimension. A ``LinearAxis`` there is moved to just after ``message``'s last sample; a
    ``CoordinateAxis`` there is emptied. Its ``segment`` is ``message.segment + 1``, and nothing
    else may share it: a source keeping its own count continues it from the result, so that its
    next message, even data that continues in time, takes ``result.segment + 1`` or more. A
    source using :class:`SegmentTracker` calls :meth:`SegmentTracker.end_of_stream` instead.

    Args:
        message: The last message the source emitted, or a template whose stream axis
            already starts where the next sample would have.
    """
    dim = resolve_stream_dim(message)
    idx = message.get_axis_idx(dim)
    axes = message.axes
    axis = axes.get(dim)
    if isinstance(axis, LinearAxis):
        axes = {**axes, dim: replace(axis, offset=axis.offset + message.data.shape[idx] * axis.gain)}
    elif isinstance(axis, CoordinateAxis) and dim in axis.dims:
        axes = {**axes, dim: replace(axis, data=_empty_along(axis.data, axis.dims.index(dim)))}
    return replace(message, data=_empty_along(message.data, idx), axes=axes, segment=message.segment + 1)


def find_gaps(timestamps: npt.ArrayLike, period: float, tolerance: float) -> npt.NDArray[np.intp]:
    """Indices of the samples that do not follow the sample before them.

    Sample ``i`` follows sample ``i - 1`` when ``timestamps[i] - timestamps[i - 1]`` is within
    ``tolerance`` of ``period``. A step backward breaks the run just as a jump forward does.
    Use the result to split a chunk before it is given a ``LinearAxis``, which cannot
    describe a gap: ``np.split(data, find_gaps(...))``.

    Args:
        timestamps: Per-sample times, in seconds.
        period: Expected time between samples, in seconds (``1 / fs``).
        tolerance: Largest deviation from ``period``, in seconds, that is still continuous.
            Size it to the jitter of the timestamps, not to the sample period: a slip of a
            sample or two costs a filter far less than the reset a reported gap causes.

    Returns:
        Sorted indices in ``[1, len(timestamps))``; empty when the chunk is continuous.
    """
    steps = np.diff(np.asarray(timestamps, dtype=float))
    return np.flatnonzero(np.abs(steps - period) > tolerance) + 1


class SegmentTracker:
    """Numbers the segments of one stream from the offsets of its chunks.

    Counting starts at 0 and goes up by one per break.

    Call :meth:`update` once per chunk, in order, and put the result in the message's
    ``segment``. Each chunk is compared with the one before it only, so neither clock drift
    nor one late chunk accumulates into a false gap. Call :meth:`bump` for a break the
    offsets cannot show, such as a device reopened with known data loss. A changed
    configuration alone is not a break: it changes ``config_id``, not the segment.
    End a segment with :meth:`end_of_stream`, not the module function, so that
    the next chunk is numbered after the marker.

    Args:
        tolerance: Largest difference, in seconds, between a chunk's offset and where the
            previous chunk ended that is still continuous. Two clock-stamped chunks can
            each be off by the source's jitter, so allow twice that.
    """

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.segment = 0
        self._expected: float | None = None
        self._ended = False

    def update(self, offset: float, n_samples: int, gain: float) -> int:
        """Account for a chunk of ``n_samples`` starting at ``offset``; return its segment."""
        if self._ended or (self._expected is not None and abs(offset - self._expected) > self.tolerance):
            self.segment += 1
        self._ended = False
        self._expected = offset + n_samples * gain
        return self.segment

    def bump(self) -> int:
        """Start a new segment regardless of the next chunk's offset; return it."""
        self.segment += 1
        self._ended = False
        self._expected = None
        return self.segment

    def end_of_stream(self, message: AxisArray) -> AxisArray:
        """:func:`end_of_stream` of ``message``, numbered by this tracker.

        The marker takes the next segment, and the next chunk (or :meth:`bump`) the one after
        it, whatever its offset: the marker says its segment has ended.
        """
        self.segment += 1
        self._ended = True
        self._expected = None
        return replace(end_of_stream(message), segment=self.segment)
