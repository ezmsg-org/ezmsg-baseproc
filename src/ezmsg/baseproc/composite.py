"""Composite processor classes for building pipelines."""

import inspect
import logging
import pickle
import typing
from abc import ABC, abstractmethod
from types import GeneratorType

from .processor import BaseProcessor, BaseProducer, _changed_settings_fields
from .protocols import MessageInType, MessageOutType, SettingsType
from .stateful import Stateful
from .util.asio import SyncToAsyncGeneratorWrapper
from .util.typeresolution import check_message_type_compatibility

logger = logging.getLogger(__name__)


def _close_child(child: typing.Any) -> None:
    """Call ``close()`` on a composite child if it exposes one.

    Safe against missing or failing ``close``; exceptions are logged and
    swallowed so a misbehaving child cannot block a settings-driven rebuild.
    """
    if child is None:
        return
    close = getattr(child, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.exception("close() raised on composite child; ignoring")


def _apply_composite_settings(composite: typing.Any, new_settings: typing.Any) -> None:
    """Default ``update_settings`` logic shared by Composite{Processor,Producer}.

    Rebinds ``composite.settings`` and, if any changed field is outside
    ``NONRESET_SETTINGS_FIELDS``, rebuilds the child pipeline from
    ``_initialize_processors(new_settings)``. Children whose key *and* exact
    type survive the rebuild keep their state via a delegated
    ``update_settings`` call; children whose key disappears or whose type
    changes are closed and replaced.
    """
    changed = _changed_settings_fields(composite.settings, new_settings)
    composite.settings = new_settings
    if not (changed - composite.NONRESET_SETTINGS_FIELDS):
        return
    new_procs = composite._initialize_processors(new_settings)
    for key, new_proc in new_procs.items():
        old_proc = composite._procs.get(key)
        if (
            old_proc is not None
            and type(old_proc) is type(new_proc)
            and hasattr(old_proc, "update_settings")
            and hasattr(new_proc, "settings")
        ):
            old_proc.update_settings(new_proc.settings)
        else:
            _close_child(old_proc)
            composite._procs[key] = new_proc
    for key in list(composite._procs):
        if key not in new_procs:
            _close_child(composite._procs.pop(key))
    composite._validate_processor_chain()


def _get_processor_message_type(
    proc: BaseProcessor | BaseProducer | GeneratorType | SyncToAsyncGeneratorWrapper,
    dir: str,
) -> type | None:
    """Extract the input or output message type from a processor.

    For objects without ``get_message_type``, falls back to inspecting
    ``__call__`` type hints (return annotation for "out", first parameter
    annotation for "in"). Returns ``typing.Any`` if no type info is available.
    """
    if isinstance(proc, GeneratorType) or isinstance(proc, SyncToAsyncGeneratorWrapper):
        gen_func = proc.gi_frame.f_globals[proc.gi_frame.f_code.co_name]
        args = typing.get_args(gen_func.__annotations__.get("return"))
        return args[0] if dir == "out" else args[1]  # yield type / send type
    if hasattr(proc.__class__, "get_message_type"):
        return proc.__class__.get_message_type(dir)
    # Fall back to __call__ type hints
    if callable(proc):
        try:
            hints = typing.get_type_hints(proc.__call__)
        except Exception:
            return typing.Any
        if dir == "out":
            return hints.get("return", typing.Any)
        # "in": first parameter after 'self'
        params = [k for k in hints if k != "return"]
        if params:
            return hints[params[0]]
    return typing.Any


def _has_stateful_op(proc: typing.Any) -> typing.TypeGuard[Stateful]:
    """
    Check if the processor has a stateful_op method.
    This is used to determine if the processor is stateful or not.
    """
    return hasattr(proc, "stateful_op")


class CompositeStateful(Stateful[dict[str, typing.Any]], ABC, typing.Generic[SettingsType, MessageOutType]):
    """
    Mixin class for composite processor/producer chains. DO NOT use this class directly.
    Used to enforce statefulness of the composite processor/producer chain and provide
    initialization and validation methods.
    """

    _procs: dict[str, BaseProducer | BaseProcessor | GeneratorType | SyncToAsyncGeneratorWrapper]
    _processor_type: typing.Literal["producer", "processor"]

    def _validate_processor_chain(self) -> None:
        """Validate the composite chain types at runtime."""
        if not self._procs:
            raise ValueError(f"Composite {self._processor_type} requires at least one processor")

        expected_in_type = _get_processor_message_type(self, "in")
        expected_out_type = _get_processor_message_type(self, "out")

        procs = [p for p in self._procs.items() if p[1] is not None]
        in_type = _get_processor_message_type(procs[0][1], "in")
        if not check_message_type_compatibility(expected_in_type, in_type):
            raise TypeError(
                f"Input type mismatch: Composite {self._processor_type} expects {expected_in_type}, "
                f"but its first processor (name: {procs[0][0]}, type: {procs[0][1].__class__.__name__}) "
                f"accepts {in_type}"
            )

        out_type = _get_processor_message_type(procs[-1][1], "out")
        if not check_message_type_compatibility(out_type, expected_out_type):
            raise TypeError(
                f"Output type mismatch: Composite {self._processor_type} wants to return {expected_out_type}, "
                f"but its last processor (name: {procs[-1][0]}, type: {procs[-1][1].__class__.__name__})  "
                f"returns {out_type}"
            )

        # Check intermediate connections
        for i in range(len(procs) - 1):
            current_out_type = _get_processor_message_type(procs[i][1], "out")
            next_in_type = _get_processor_message_type(procs[i + 1][1], "in")

            if current_out_type is None or current_out_type is type(None):
                raise TypeError(
                    f"Processor {i} (name: {procs[i][0]}, type: {procs[i][1].__class__.__name__}) is a consumer "
                    "or returns None. Consumers can only be the last processor of a "
                    f"composite {self._processor_type} chain."
                )
            if next_in_type is None or next_in_type is type(None):
                raise TypeError(
                    f"Processor {i + 1} (name: {procs[i + 1][0]}, type: {procs[i + 1][1].__class__.__name__}) "
                    f"is a producer or receives only None. Producers can only be the first processor of a composite "
                    f"producer chain."
                )
            if not check_message_type_compatibility(current_out_type, next_in_type):
                raise TypeError(
                    f"Message type mismatch between processors {i} (name: {procs[i][0]}, "
                    f"type: {procs[i][1].__class__.__name__}) "
                    f"and {i + 1} (name: {procs[i + 1][0]}, type: {procs[i + 1][1].__class__.__name__}): "
                    f"{procs[i][1].__class__.__name__} outputs {current_out_type}, "
                    f"but {procs[i + 1][1].__class__.__name__} expects {next_in_type}"
                )
            if inspect.isgenerator(procs[i][1]) and hasattr(procs[i][1], "send"):
                # If the processor is a generator, wrap it in a SyncToAsyncGeneratorWrapper
                procs[i] = (procs[i][0], SyncToAsyncGeneratorWrapper(procs[i][1]))
        if inspect.isgenerator(procs[-1][1]) and hasattr(procs[-1][1], "send"):
            # If the last processor is a generator, wrap it in a SyncToAsyncGeneratorWrapper
            procs[-1] = (procs[-1][0], SyncToAsyncGeneratorWrapper(procs[-1][1]))
        self._procs = {k: v for (k, v) in procs}

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, typing.Any]: ...

    @property
    def state(self) -> dict[str, typing.Any]:
        return {k: getattr(proc, "state") for k, proc in self._procs.items() if hasattr(proc, "state")}

    @state.setter
    def state(self, state: dict[str, typing.Any] | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                state = pickle.loads(state)
            for k, v in state.items():  # type: ignore
                if k not in self._procs:
                    raise KeyError(
                        f"Processor (name: {k}) in provided state not found in composite {self._processor_type} chain. "
                        f"Available keys: {list(self._procs.keys())}"
                    )
                if hasattr(self._procs[k], "state"):
                    setattr(self._procs[k], "state", v)

    def _reset_state(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # By default, we don't expect to change the state of a composite processor/producer
        pass

    def _post_process(self, result: MessageOutType | None) -> MessageOutType | None:
        """Hook called after the processor chain completes. Override to add post-processing."""
        return result

    @abstractmethod
    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]: ...


class CompositeProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    CompositeStateful[SettingsType, MessageOutType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType],
):
    """
    A processor that chains multiple processors together in a feedforward non-branching graph.
    The individual processors may be stateless or stateful. The last processor may be a consumer,
    otherwise processors must be transformers. Use ``CompositeProducer`` if you want the first
    processor to be a producer. Concrete subclasses must implement ``_initialize_processors``.
    Optionally override ``_reset_state`` if you want adaptive state behaviour.

    Example implementation::

        class CustomCompositeProcessor(CompositeProcessor[CustomSettings, AxisArray, AxisArray]):
            @staticmethod
            def _initialize_processors(settings: CustomSettings) -> dict[str, BaseProcessor]:
                return {
                    "stateful_transformer": CustomStatefulProducer(**settings),
                    "transformer": CustomTransformer(**settings),
                }

    Where ``**settings`` should be replaced with initialisation arguments for each processor.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._processor_type = "processor"
        self._procs = self._initialize_processors(self.settings)
        self._validate_processor_chain()
        first_proc = next(iter(self._procs.items()))
        first_proc_in_type = _get_processor_message_type(first_proc[1], "in")
        if first_proc_in_type is None or first_proc_in_type is type(None):
            raise TypeError(
                f"First processor (name: {first_proc[0]}, type: {first_proc[1].__class__.__name__}) "
                f"is a producer or receives only None. Please use CompositeProducer, not "
                f"CompositeProcessor for this composite chain."
            )
        self._hash = -1

    @staticmethod
    @abstractmethod
    def _initialize_processors(settings: SettingsType) -> dict[str, typing.Any]: ...

    def update_settings(self, new_settings: SettingsType) -> None:
        """Rebind settings and reconcile the child pipeline.

        See :func:`_apply_composite_settings` for the reconciliation rules.
        Override for finer-grained control than the class-level
        ``NONRESET_SETTINGS_FIELDS`` allow-list provides.
        """
        _apply_composite_settings(self, new_settings)

    def _process(self, message: MessageInType | None = None) -> MessageOutType | None:
        """
        Process a message through the pipeline of processors. If the message is None, or no message is provided,
        then it will be assumed that the first processor is a producer and will be called without arguments.
        This will be invoked via `__call__` or `send`.

        Warning: All processors will be called using their synchronous API, which may invoke a slow sync->async wrapper
        for processors that are async-first (i.e., children of BaseProducer or BaseAsyncTransformer).
        If you are in an async context, please use instead this object's `asend` or `__acall__`,
        which is much faster for async processors and does not incur penalty on sync processors.
        """
        result = message
        for proc in self._procs.values():
            result = proc(result) if callable(proc) else proc.send(result)
        return self._post_process(result)

    async def _aprocess(self, message: MessageInType | None = None) -> MessageOutType | None:
        """
        Process a message through the pipeline of processors using their async APIs.
        If the message is None, or no message is provided, then it will be assumed that the first processor
        is a producer and will be called without arguments.
        We use `__anext__` and `asend` to allow using legacy generators that have yet to be converted to transformers.
        """
        result = message
        for proc in self._procs.values():
            result = await proc.asend(result)
        return self._post_process(result)

    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
        message: MessageInType | None,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]:
        result = message
        state = state or {}
        try:
            state_keys = list(state.keys())
        except AttributeError as e:
            raise AttributeError("state provided to stateful_op must be a dict or None") from e
        for key in state_keys:
            if key not in self._procs:
                raise KeyError(
                    f"Processor (name: {key}) in provided state not found in composite processor chain. "
                    f"Available keys: {list(self._procs.keys())}"
                )
        for k, proc in self._procs.items():
            if _has_stateful_op(proc):
                state[k], result = proc.stateful_op(state.get(k, None), result)
            else:
                result = proc(result) if callable(proc) else proc.send(result)
        return state, self._post_process(result)


class CompositeProducer(
    BaseProducer[SettingsType, MessageOutType],
    CompositeStateful[SettingsType, MessageOutType],
    ABC,
    typing.Generic[SettingsType, MessageOutType],
):
    """
    A producer that chains multiple processors (starting with a producer) together in a feedforward
    non-branching graph. The individual processors may be stateless or stateful.
    The first processor must be a producer, the last processor may be a consumer, otherwise
    processors must be transformers.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._processor_type = "producer"
        self._procs = self._initialize_processors(self.settings)
        self._validate_processor_chain()
        first_proc = next(iter(self._procs.items()))
        first_proc_in_type = _get_processor_message_type(first_proc[1], "in")
        if first_proc_in_type is not None and first_proc_in_type is not type(None):
            raise TypeError(
                f"First processor (name: {first_proc[0]}, type: {first_proc[1].__class__.__name__}) "
                f"is not a producer. Please use CompositeProcessor, not "
                f"CompositeProducer for this composite chain."
            )
        self._hash = -1

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, typing.Any]: ...

    def update_settings(self, new_settings: SettingsType) -> None:
        """Rebind settings and reconcile the child pipeline.

        See :func:`_apply_composite_settings` for the reconciliation rules.
        Override for finer-grained control than the class-level
        ``NONRESET_SETTINGS_FIELDS`` allow-list provides.
        """
        _apply_composite_settings(self, new_settings)

    async def _produce(self) -> MessageOutType:
        """
        Process a message through the pipeline of processors. If the message is None, or no message is provided,
        then it will be assumed that the first processor is a producer and will be called without arguments.
        This will be invoked via `__call__` or `send`.
        We use `__next__` and `send` to allow using legacy generators that have yet to be converted to transformers.

        Warning: All processors will be called using their asynchronous API, which is much faster for async
        processors and does not incur penalty on sync processors.
        """
        procs = list(self._procs.values())
        result = await procs[0].__anext__()
        for proc in procs[1:]:
            result = await proc.asend(result)
        return self._post_process(result)

    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]:
        state = state or {}
        try:
            state_keys = list(state.keys())
        except AttributeError as e:
            raise AttributeError("state provided to stateful_op must be a dict or None") from e
        for key in state_keys:
            if key not in self._procs:
                raise KeyError(
                    f"Processor (name: {key}) in provided state not found in composite producer chain. "
                    f"Available keys: {list(self._procs.keys())}"
                )
        labeled_procs = list(self._procs.items())
        prod_name, prod = labeled_procs[0]
        if _has_stateful_op(prod):
            state[prod_name], result = prod.stateful_op(state.get(prod_name, None))
        else:
            result = prod.__next__()
        for k, proc in labeled_procs[1:]:
            if _has_stateful_op(proc):
                state[k], result = proc.stateful_op(state.get(k, None), result)
            else:
                result = proc(result) if callable(proc) else proc.send(result)
        return state, self._post_process(result)
