"""Protocol definitions and type variables for ezmsg processors."""

import functools
import typing
from dataclasses import dataclass

import ezmsg.core as ez

from .util.message import SampleMessage

# --- Processor state decorator ---
processor_state = functools.partial(dataclass, unsafe_hash=True, frozen=False, init=False)

# --- Type variables for protocols and processors ---
MessageInType = typing.TypeVar("MessageInType")
MessageOutType = typing.TypeVar("MessageOutType")
SettingsType = typing.TypeVar("SettingsType", bound=ez.Settings)
StateType = typing.TypeVar("StateType")


# --- Protocols for processors ---
class Processor(typing.Protocol[SettingsType, MessageInType, MessageOutType]):
    """
    Protocol for processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous Consumer and Transformer protocols, and the base classes
    in this module which implement them.

    Note: In Python 3.12+, we can invoke `__acall__` directly using `await obj(message)`,
     but to support earlier versions we need to use `await obj.__acall__(message)`.
    """

    def __call__(self, message: typing.Any) -> typing.Any: ...
    async def __acall__(self, message: typing.Any) -> typing.Any: ...


class Producer(typing.Protocol[SettingsType, MessageOutType]):
    """
    Protocol for producers that generate messages.
    """

    def __call__(self) -> MessageOutType: ...
    async def __acall__(self) -> MessageOutType: ...


class Consumer(Processor[SettingsType, MessageInType, None], typing.Protocol):
    """
    Protocol for consumers that receive messages but do not return a result.
    """

    def __call__(self, message: MessageInType) -> None: ...
    async def __acall__(self, message: MessageInType) -> None: ...


class Transformer(Processor[SettingsType, MessageInType, MessageOutType], typing.Protocol):
    """Protocol for transformers that receive messages and return a result of the same class."""

    def __call__(self, message: MessageInType) -> MessageOutType: ...
    async def __acall__(self, message: MessageInType) -> MessageOutType: ...


class StatefulProcessor(typing.Protocol[SettingsType, MessageInType, MessageOutType, StateType]):
    """
    Base protocol for _stateful_ message processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous StatefulConsumer and StatefulTransformer protocols.
    """

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self, message: typing.Any) -> typing.Any: ...
    async def __acall__(self, message: typing.Any) -> typing.Any: ...

    def stateful_op(
        self,
        state: typing.Any,
        message: typing.Any,
    ) -> tuple[typing.Any, typing.Any]: ...


class StatefulProducer(typing.Protocol[SettingsType, MessageOutType, StateType]):
    """Protocol for producers that generate messages without consuming inputs."""

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self) -> MessageOutType: ...
    async def __acall__(self) -> MessageOutType: ...

    def stateful_op(
        self,
        state: typing.Any,
    ) -> tuple[typing.Any, typing.Any]: ...


class StatefulConsumer(StatefulProcessor[SettingsType, MessageInType, None, StateType], typing.Protocol):
    """Protocol specifically for processors that consume messages without producing output."""

    def __call__(self, message: MessageInType) -> None: ...
    async def __acall__(self, message: MessageInType) -> None: ...

    def stateful_op(
        self,
        state: tuple[StateType, int],
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], None]: ...

    """
    Note: The return type is still a tuple even though the second entry is always None.
    This is intentional so we can use the same protocol for both consumers and transformers,
    and chain them together in a pipeline (e.g., `CompositeProcessor`).
    """


class StatefulTransformer(
    StatefulProcessor[SettingsType, MessageInType, MessageOutType, StateType],
    typing.Protocol,
):
    """
    Protocol specifically for processors that transform messages.
    """

    def __call__(self, message: MessageInType) -> MessageOutType: ...
    async def __acall__(self, message: MessageInType) -> MessageOutType: ...

    def stateful_op(
        self,
        state: tuple[StateType, int],
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], MessageOutType]: ...


class AdaptiveTransformer(StatefulTransformer, typing.Protocol):
    def partial_fit(self, message: SampleMessage) -> None:
        """Update transformer state using labeled training data.

        This method should update the internal state/parameters of the transformer
        based on the provided labeled samples, without performing any transformation.
        """
        ...

    async def apartial_fit(self, message: SampleMessage) -> None: ...
