"""Stateful processor base classes for ezmsg."""

import pickle
import typing
from abc import ABC, abstractmethod

from .processor import (
    BaseProcessor,
    BaseProducer,
    _get_base_processor_message_in_type,
)
from .protocols import MessageInType, MessageOutType, SettingsType, StateType
from .util.asio import run_coroutine_sync
from .util.message import SampleMessage, is_sample_message
from .util.typeresolution import resolve_typevar


def _get_base_processor_state_type(cls: type) -> type:
    try:
        return resolve_typevar(cls, StateType)
    except TypeError as e:
        raise TypeError(
            f"Could not resolve state type for {cls}. Ensure that the class is properly annotated with a StateType."
        ) from e


class Stateful(ABC, typing.Generic[StateType]):
    """
    Mixin class for stateful processors. DO NOT use this class directly.
    Used to enforce that the processor/producer has a state attribute and stateful_op method.
    """

    _state: StateType

    @classmethod
    def get_state_type(cls) -> type[StateType]:
        return _get_base_processor_state_type(cls)

    @property
    def state(self) -> StateType:
        return self._state

    @state.setter
    def state(self, state: StateType | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state  # type: ignore

    def _hash_message(self, message: typing.Any) -> int:
        """
        Check if the message metadata indicates a need for state reset.

        This method is not abstract because there are some processors that might only
        need to reset once but are otherwise insensitive to the message structure.

        For example, an activation function that benefits greatly from pre-computed values should
        do this computation in `_reset_state` and attach those values to the processor state,
        but if it e.g. operates elementwise on the input then it doesn't care if the incoming
        data changes shape or sample rate so you don't need to reset again.

        All processors' initial state should have `.hash = -1` then by returning `0` here
        we force an update on the first message.
        """
        return 0

    @abstractmethod
    def _reset_state(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Reset internal state based on
            - new message metadata (processors), or
            - after first call (producers).
        """
        ...

    @abstractmethod
    def stateful_op(self, *args: typing.Any, **kwargs: typing.Any) -> tuple: ...


class BaseStatefulProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    Stateful[StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful processor functionality.
    You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
    Use BaseStatefulConsumer for operations that do not return a result,
    or BaseStatefulTransformer for operations that do return a result.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hash = -1
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()
        # TODO: Enforce that StateType has .hash: int field.

    @abstractmethod
    def _reset_state(self, message: typing.Any) -> None:
        """
        Reset internal state based on new message metadata.
        This method will only be called when there is a significant change in the message metadata,
        such as sample rate or shape (criteria defined by `_hash_message`), and not for every message,
        so use it to do all the expensive pre-allocation and caching of variables that can speed up
        the processing of subsequent messages in `_process`.
        """
        ...

    @abstractmethod
    def _process(self, message: typing.Any) -> typing.Any: ...

    def __call__(self, message: typing.Any) -> typing.Any:
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return self._process(message)

    async def __acall__(self, message: typing.Any) -> typing.Any:
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return await self._aprocess(message)

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
        message: typing.Any,
    ) -> tuple[tuple[StateType, int], typing.Any]:
        if state is not None:
            self.state, self._hash = state
        result = self(message)
        return (self.state, self._hash), result


class BaseStatefulProducer(
    BaseProducer[SettingsType, MessageOutType],
    Stateful[StateType],
    ABC,
    typing.Generic[SettingsType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful producer functionality.
      Examples of stateful producers are things that require counters, clocks,
      or to cycle through a set of values.

    Unlike BaseStatefulProcessor, this class does not message hashing because there
      are no input messages. We still use self._hash to simply track the transition from
      initialization (.hash == -1) to state reset (.hash == 0).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._hash = -1
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()

    @abstractmethod
    def _reset_state(self) -> None:
        """
        Reset internal state upon first call.
        """
        ...

    async def __acall__(self) -> MessageOutType:
        if self._hash == -1:
            self._reset_state()
            self._hash = 0
        return await self._produce()

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
    ) -> tuple[tuple[StateType, int], MessageOutType]:
        if state is not None:
            self.state, self._hash = state  # Update state via setter
        result = self()  # Uses synchronous call
        return (self.state, self._hash), result


class BaseStatefulConsumer(
    BaseStatefulProcessor[SettingsType, MessageInType, None, StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, StateType],
):
    """
    Base class for stateful message consumers that don't produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @classmethod
    def get_message_type(cls, dir: str) -> type[MessageInType] | None:
        if dir == "in":
            return _get_base_processor_message_in_type(cls)
        elif dir == "out":
            return None
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    @abstractmethod
    def _process(self, message: MessageInType) -> None: ...

    async def _aprocess(self, message: MessageInType) -> None:
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> None:
        return await super().__acall__(message)

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], None]:
        state, _ = super().stateful_op(state, message)
        return state, None


class BaseStatefulTransformer(
    BaseStatefulProcessor[SettingsType, MessageInType, MessageOutType, StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    Base class for stateful message transformers that produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageInType) -> MessageOutType: ...

    async def _aprocess(self, message: MessageInType) -> MessageOutType:
        return self._process(message)

    def __call__(self, message: MessageInType) -> MessageOutType:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        return await super().__acall__(message)

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], MessageOutType]:
        return super().stateful_op(state, message)


class BaseAdaptiveTransformer(
    BaseStatefulTransformer[
        SettingsType,
        MessageInType | SampleMessage,
        MessageOutType | None,
        StateType,
    ],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    @abstractmethod
    def partial_fit(self, message: SampleMessage) -> None: ...

    async def apartial_fit(self, message: SampleMessage) -> None:
        """Override me if you need async partial fitting."""
        return self.partial_fit(message)

    def __call__(self, message: MessageInType | SampleMessage) -> MessageOutType | None:
        """
        Adapt transformer with training data (and optionally labels)
        in SampleMessage

        Args:
            message: An instance of SampleMessage with optional
             labels (y) in message.trigger.value.data and
             data (X) in message.sample.data

        Returns: None
        """
        if is_sample_message(message):
            return self.partial_fit(message)
        return super().__call__(message)

    async def __acall__(self, message: MessageInType | SampleMessage) -> MessageOutType | None:
        if is_sample_message(message):
            return await self.apartial_fit(message)
        return await super().__acall__(message)


class BaseAsyncTransformer(
    BaseStatefulTransformer[SettingsType, MessageInType, MessageOutType, StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    This reverses the priority of async and sync methods from :obj:`BaseStatefulTransformer`.
    Whereas in :obj:`BaseStatefulTransformer`, the async methods simply called the sync methods,
    here the sync methods call the async methods, more similar to :obj:`BaseStatefulProducer`.
    """

    def _process(self, message: MessageInType) -> MessageOutType:
        return run_coroutine_sync(self._aprocess(message))

    @abstractmethod
    async def _aprocess(self, message: MessageInType) -> MessageOutType: ...

    def __call__(self, message: MessageInType) -> MessageOutType:
        # Override (synchronous) __call__ to run coroutine `aprocess`.
        return run_coroutine_sync(self.__acall__(message))

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        # Note: In Python 3.12, we can invoke this with `await obj(message)`
        # Earlier versions must be explicit: `await obj.__acall__(message)`
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return await self._aprocess(message)
