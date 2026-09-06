import time
import typing
import warnings
from dataclasses import dataclass, field

from ezmsg.util.messages.axisarray import AxisArray


@dataclass(unsafe_hash=True)
class SampleTriggerMessage:
    timestamp: float = field(default_factory=time.monotonic)
    """Time of the trigger, in seconds. The Clock depends on the input but defaults to time.monotonic"""

    period: tuple[float, float] | None = None
    """The period around the timestamp, in seconds"""

    value: typing.Any = None
    """A value or 'label' associated with the trigger."""


@dataclass
class SampleMessage:
    """
    .. deprecated::
        ``SampleMessage`` is deprecated. Use ``AxisArray`` with
        ``attrs={"trigger": SampleTriggerMessage(...)}`` instead.
    """

    trigger: SampleTriggerMessage
    """The time, window, and value (if any) associated with the trigger."""

    sample: AxisArray
    """The data sampled around the trigger."""

    def __post_init__(self):
        warnings.warn(
            "SampleMessage is deprecated. Use AxisArray with attrs={'trigger': SampleTriggerMessage(...)} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


def is_sample_message(message: typing.Any) -> bool:
    """Detect old SampleMessage OR new AxisArray-with-trigger."""
    if isinstance(message, SampleMessage):
        return True
    return isinstance(message, AxisArray) and "trigger" in getattr(message, "attrs", {})
