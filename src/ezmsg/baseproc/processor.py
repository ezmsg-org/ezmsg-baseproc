"""Base processor classes for ezmsg (non-stateful)."""

import typing
from abc import ABC, abstractmethod

from .protocols import MessageInType, MessageOutType, SettingsType
from .util.asio import run_coroutine_sync
from .util.typeresolution import resolve_typevar


def _get_base_processor_settings_type(cls: type) -> type:
    try:
        return resolve_typevar(cls, SettingsType)
    except TypeError as e:
        raise TypeError(
            f"Could not resolve settings type for {cls}. "
            f"Ensure that the class is properly annotated with a SettingsType."
        ) from e


def _get_base_processor_message_in_type(cls: type) -> type:
    return resolve_typevar(cls, MessageInType)


def _get_base_processor_message_out_type(cls: type) -> type:
    return resolve_typevar(cls, MessageOutType)


def _unify_settings(obj: typing.Any, settings: object | None, *args, **kwargs) -> typing.Any:
    """Helper function to unify settings for processor initialization."""
    settings_type = _get_base_processor_settings_type(obj.__class__)

    if settings is None:
        if len(args) > 0 and isinstance(args[0], settings_type):
            settings = args[0]
        elif len(args) > 0 or len(kwargs) > 0:
            settings = settings_type(*args, **kwargs)
        else:
            settings = settings_type()
    assert isinstance(settings, settings_type), "Settings must be of type " + str(settings_type)
    return settings


class BaseProcessor(ABC, typing.Generic[SettingsType, MessageInType, MessageOutType]):
    """
    Base class for processors. You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
      * Use :obj:`BaseConsumer` or :obj:`BaseTransformer` for ops that return a result or not, respectively.
      * Use :obj:`BaseStatefulProcessor` and its children for operations that require state.

    Note that `BaseProcessor` and its children are sync by default. If you need async by defualt, then
    override the async methods and call them from the sync methods. Look to `BaseProducer` for examples of
    calling async methods from sync methods.
    """

    settings: SettingsType

    @classmethod
    def get_settings_type(cls) -> type[SettingsType]:
        return _get_base_processor_settings_type(cls)

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Any:
        if dir == "in":
            return _get_base_processor_message_in_type(cls)
        elif dir == "out":
            return _get_base_processor_message_out_type(cls)
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, settings: SettingsType | None = None, **kwargs) -> None:
        self.settings = _unify_settings(self, settings, *args, **kwargs)

    @abstractmethod
    def _process(self, message: typing.Any) -> typing.Any: ...

    async def _aprocess(self, message: typing.Any) -> typing.Any:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: typing.Any) -> typing.Any:
        # Note: We use the indirection to `_process` because this allows us to
        #  modify __call__ in derived classes with common functionality while
        #  minimizing the boilerplate code in derived classes as they only need to
        #  implement `_process`.
        return self._process(message)

    async def __acall__(self, message: typing.Any) -> typing.Any:
        """
        In Python 3.12+, we can invoke this method simply with `await obj(message)`,
        but earlier versions require direct syntax: `await obj.__acall__(message)`.
        """
        return await self._aprocess(message)

    def send(self, message: typing.Any) -> typing.Any:
        """Alias for __call__."""
        return self(message)

    async def asend(self, message: typing.Any) -> typing.Any:
        """Alias for __acall__."""
        return await self.__acall__(message)


class BaseProducer(ABC, typing.Generic[SettingsType, MessageOutType]):
    """
    Base class for producers -- processors that generate messages without consuming inputs.

    Note that `BaseProducer` and its children are async by default, and the sync methods simply wrap
      the async methods. This is the opposite of :obj:`BaseProcessor` and its children which are sync by default.
      These classes are designed this way because it is highly likely that a producer, which (probably) does not
      receive inputs, will require some sort of IO which will benefit from being async.
    """

    @classmethod
    def get_settings_type(cls) -> type[SettingsType]:
        return _get_base_processor_settings_type(cls)

    @classmethod
    def get_message_type(cls, dir: str) -> type[MessageOutType] | None:
        if dir == "out":
            return _get_base_processor_message_out_type(cls)
        elif dir == "in":
            return None
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, settings: SettingsType | None = None, **kwargs) -> None:
        self.settings = _unify_settings(self, settings, *args, **kwargs)

    @abstractmethod
    async def _produce(self) -> MessageOutType: ...

    async def __acall__(self) -> MessageOutType:
        return await self._produce()

    def __call__(self) -> MessageOutType:
        # Warning: This is a bit slow. Override this method in derived classes if performance is critical.
        return run_coroutine_sync(self.__acall__())

    def __iter__(self) -> typing.Iterator[MessageOutType]:
        # Make self an iterator
        return self

    async def __anext__(self) -> MessageOutType:
        # So this can be used as an async generator.
        return await self.__acall__()

    def __next__(self) -> MessageOutType:
        # So this can be used as a generator.
        return self()


class BaseConsumer(
    BaseProcessor[SettingsType, MessageInType, None],
    ABC,
    typing.Generic[SettingsType, MessageInType],
):
    """
    Base class for consumers -- processors that receive messages but don't produce output.
    This base simply overrides type annotations of BaseProcessor to remove the outputs.
    (We don't bother overriding `send` and `asend` because those are deprecated.)
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
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> None:
        return await super().__acall__(message)


class BaseTransformer(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType],
):
    """
    Base class for transformers -- processors which receive messages and produce output.
    This base simply overrides type annotations of :obj:`BaseProcessor` to indicate that outputs are not optional.
    (We don't bother overriding `send` and `asend` because those are deprecated.)
    """

    @abstractmethod
    def _process(self, message: MessageInType) -> MessageOutType: ...

    async def _aprocess(self, message: MessageInType) -> MessageOutType:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> MessageOutType:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        return await super().__acall__(message)
