"""Counter generator for sample counting and timing."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, replace

from .clockdriven import (
    BaseClockDrivenProducer,
    ClockDrivenSettings,
    ClockDrivenState,
)
from .protocols import processor_state
from .units import BaseClockDrivenUnit


class CounterSettings(ClockDrivenSettings):
    """Settings for :obj:`Counter` and :obj:`CounterTransformer`."""

    mod: int | None = None
    """If set, counter values rollover at this modulus."""


@processor_state
class CounterTransformerState(ClockDrivenState):
    """State for :obj:`CounterTransformer`."""

    template: AxisArray | None = None


class CounterTransformer(BaseClockDrivenProducer[CounterSettings, CounterTransformerState]):
    """
    Transforms clock ticks (LinearAxis) into AxisArray counter values.

    Each clock tick produces a block of counter values. The block size is either
    fixed (n_time setting) or derived from the clock's gain (fs * gain).
    """

    def _reset_state(self, time_axis: LinearAxis) -> None:
        """Reset state - initialize template for counter output."""
        self._state.template = AxisArray(
            data=np.array([], dtype=int),
            dims=["time"],
            axes={"time": time_axis},
            key="counter",
        )

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        """Generate counter values for this chunk."""
        # Generate counter data (using pre-increment counter value)
        block_samp = np.arange(self._state.counter, self._state.counter + n_samples)
        if self.settings.mod is not None:
            block_samp = block_samp % self.settings.mod

        return replace(
            self._state.template,
            data=block_samp,
            axes={"time": time_axis},
        )


class Counter(BaseClockDrivenUnit[CounterSettings, CounterTransformer]):
    """
    Transforms clock ticks into monotonically increasing counter values as AxisArray.

    Receives timing from INPUT_CLOCK (LinearAxis from Clock) and outputs AxisArray.
    """

    SETTINGS = CounterSettings
