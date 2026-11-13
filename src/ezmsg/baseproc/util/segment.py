"""Detect breaks in a stream's time base and number its segments.

A source stamps every message with
:attr:`~ezmsg.util.messages.axisarray.AxisArray.segment`, and changes the value on
the first message that does not continue the previous one. Stateful processors
reset when they see it change (``BaseStatefulProcessor._reset_segment``), so
gap detection happens once, at the source, instead of in every processor.
"""

import numpy as np
import numpy.typing as npt


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

    Call :meth:`update` once per chunk, in order, and put the result in the message's
    ``segment``. Each chunk is compared with the one before it only, so neither clock drift
    nor one late chunk accumulates into a false gap. Call :meth:`bump` for a break the
    offsets cannot show, such as a reconnect to a different device or a changed
    configuration.

    Args:
        tolerance: Largest difference, in seconds, between a chunk's offset and where the
            previous chunk ended that is still continuous. Two clock-stamped chunks can
            each be off by the source's jitter, so allow twice that.
    """

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.segment = 0
        self._expected: float | None = None

    def update(self, offset: float, n_samples: int, gain: float) -> int:
        """Account for a chunk of ``n_samples`` starting at ``offset``; return its segment."""
        if self._expected is not None and abs(offset - self._expected) > self.tolerance:
            self.segment += 1
        self._expected = offset + n_samples * gain
        return self.segment

    def bump(self) -> int:
        """Start a new segment regardless of the next chunk's offset; return it."""
        self.segment += 1
        self._expected = None
        return self.segment
