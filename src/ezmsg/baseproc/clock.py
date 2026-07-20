"""Clock generator for timing control."""

import asyncio
import math
import time
from dataclasses import field

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .protocols import processor_state
from .stateful import BaseStatefulProducer
from .units import BaseProducerUnit


class ClockSettings(ez.Settings):
    """Settings for :obj:`ClockProducer`."""

    dispatch_rate: float | None = None
    """
    Dispatch rate in Hz.
    - Finite value (e.g., 100.0): Dispatch 100 times per second.
    - None (default): Dispatch as fast as possible (no sleep). ``math.inf`` is
      also accepted as an "unthrottled" sentinel for backward compatibility, but
      None is preferred (math.inf is not JSON-serializable, which breaks tools
      that reflect the settings schema, e.g. the ezmsg dashboard).
    """


@processor_state
class ClockState:
    """State for :obj:`ClockProducer`."""

    t_0: float = field(default_factory=time.monotonic)
    """Start time (monotonic)."""

    n_dispatch: int = 0
    """Number of dispatches since reset."""


class ClockProducer(BaseStatefulProducer[ClockSettings, AxisArray.LinearAxis, ClockState]):
    """
    Produces clock ticks at a specified rate.

    Each tick outputs a :obj:`AxisArray.LinearAxis` containing:
    - ``gain``: 1/dispatch_rate (seconds per tick), or 0.0 if dispatch_rate is infinite
    - ``offset``: Wall clock timestamp (time.monotonic)

    This output type allows downstream components (like Counter) to know both
    the timing of the tick and the nominal dispatch rate.
    """

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._state.t_0 = time.monotonic()
        self._state.n_dispatch = 0

    @property
    def _throttled(self) -> bool:
        """Whether a finite dispatch rate is set (None / inf mean unthrottled)."""
        rate = self.settings.dispatch_rate
        return rate is not None and math.isfinite(rate)

    def _make_output(self, timestamp: float) -> AxisArray.LinearAxis:
        """Create LinearAxis output with gain and offset."""
        gain = 1.0 / self.settings.dispatch_rate if self._throttled else 0.0
        return AxisArray.LinearAxis(gain=gain, offset=timestamp)

    def __call__(self) -> AxisArray.LinearAxis:
        """Synchronous clock production."""
        if self._hash == -1:
            self._reset_state()
            self._hash = 0

        now = time.monotonic()
        if self._throttled:
            target_time = self.state.t_0 + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            if target_time > now:
                time.sleep(target_time - now)
        else:
            target_time = now

        self.state.n_dispatch += 1
        return self._make_output(target_time)

    async def _produce(self) -> AxisArray.LinearAxis:
        """Generate next clock tick."""
        now = time.monotonic()
        if self._throttled:
            target_time = self.state.t_0 + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            if target_time > now:
                await asyncio.sleep(target_time - now)
        else:
            target_time = now

        self.state.n_dispatch += 1
        return self._make_output(target_time)


class Clock(
    BaseProducerUnit[
        ClockSettings,
        AxisArray.LinearAxis,
        ClockProducer,
    ]
):
    """
    Clock unit that produces ticks at a specified rate.

    Output is a :obj:`AxisArray.LinearAxis` with:
    - ``gain``: 1/dispatch_rate (seconds per tick)
    - ``offset``: Wall clock timestamp
    """

    SETTINGS = ClockSettings
