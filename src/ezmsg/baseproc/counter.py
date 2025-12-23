"""Counter generator for sample counting and timing."""

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, replace

from .protocols import processor_state
from .stateful import BaseStatefulTransformer
from .units import BaseTransformerUnit


class CounterSettings(ez.Settings):
    """Settings for :obj:`Counter` and :obj:`CounterTransformer`."""

    fs: float
    """Sampling rate in Hz."""

    n_time: int | None = None
    """
    Samples per block.
    - If specified: fixed chunk size (clock gain is ignored)
    - If None: derived from clock gain (fs * clock.gain), with fractional sample tracking
    """

    mod: int | None = None
    """If set, counter values rollover at this modulus."""


@processor_state
class CounterTransformerState:
    """State for :obj:`CounterTransformer`."""

    counter: int = 0
    """Current counter value (next sample index)."""

    fractional_samples: float = 0.0
    """Accumulated fractional samples for variable chunk mode."""

    template: AxisArray | None = None


class CounterTransformer(
    BaseStatefulTransformer[CounterSettings, AxisArray.LinearAxis, AxisArray, CounterTransformerState]
):
    """
    Transforms clock ticks (LinearAxis) into AxisArray counter values.

    Each clock tick produces a block of counter values. The block size is either
    fixed (n_time setting) or derived from the clock's gain (fs * gain).
    """

    def _reset_state(self, message: AxisArray.LinearAxis) -> None:
        """Reset state - counter transformer state is simple, just reset values."""
        self._state.counter = 0
        self._state.fractional_samples = 0.0
        self._state.template = AxisArray(
            data=np.array([], dtype=int),
            dims=["time"],
            axes={
                "time": AxisArray.TimeAxis(fs=self.settings.fs, offset=message.offset),
            },
            key="counter",
        )

    def _hash_message(self, message: AxisArray.LinearAxis) -> int:
        # Return constant hash - counter state should never reset based on message content.
        # The counter maintains continuity regardless of clock rate changes.
        return 0

    def _process(self, clock_tick: AxisArray.LinearAxis) -> AxisArray | None:
        """Transform a clock tick into counter AxisArray."""
        # Determine number of samples for this block
        if self.settings.n_time is not None:
            # Fixed chunk size mode
            n_samples = self.settings.n_time
            # Use wall clock or synthetic offset based on clock gain
            if clock_tick.gain == 0.0:
                # AFAP mode - synthetic offset
                offset = self.state.counter / self.settings.fs
            else:
                # Use clock's timestamp
                offset = clock_tick.offset
        else:
            # Variable chunk size mode - derive from clock gain
            if clock_tick.gain == 0.0:
                # AFAP with no fixed n_time - this is an error
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")

            # Calculate samples including fractional accumulation
            # Add small epsilon to avoid floating point truncation errors (e.g., 0.9999999 -> 0)
            samples_float = self.settings.fs * clock_tick.gain + self.state.fractional_samples
            n_samples = int(samples_float + 1e-9)
            self.state.fractional_samples = samples_float - n_samples

            if n_samples == 0:
                # Not enough samples accumulated yet
                # TODO: Return empty array. What should offset be?
                return None

            # Use clock's timestamp for offset
            offset = clock_tick.offset

        # Generate counter data
        block_samp = np.arange(self.state.counter, self.state.counter + n_samples)
        if self.settings.mod is not None:
            block_samp = block_samp % self.settings.mod

        # Create output AxisArray
        result = replace(
            self._state.template,
            data=block_samp,
            axes={"time": replace(self._state.template.axes["time"], offset=offset)},
        )

        # Update state
        self.state.counter += n_samples

        return result


class Counter(BaseTransformerUnit[CounterSettings, AxisArray.LinearAxis, AxisArray, CounterTransformer]):
    """
    Transforms clock ticks into monotonically increasing counter values as AxisArray.

    Receives timing from INPUT_SIGNAL (LinearAxis from Clock) and outputs AxisArray.
    """

    SETTINGS = CounterSettings
