"""Stateful processor base classes for ezmsg."""

import logging
import pickle
import typing
import warnings
from abc import ABC, abstractmethod

from ezmsg.util.messages.axisarray import AxisArray, chain_config_id, replace

from .processor import (
    BaseProcessor,
    BaseProducer,
    _get_base_processor_message_in_type,
)
from .protocols import MessageInType, MessageOutType, SettingsType, StateType
from .util.asio import run_coroutine_sync
from .util.message import is_sample_message
from .util.streamdim import STREAMING_DIMS as _STREAMING_DIMS
from .util.typeresolution import resolve_typevar

logger = logging.getLogger(__name__)

# Processor classes already warned about receiving AxisArrays without a config_id.
_warned_undeclared: set[type] = set()
# Processor classes already warned about changing their output layout without a new config_id.
_warned_unannounced: set[type] = set()


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

    _config_id: int | None = None
    """The previous message's :attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id`.

    While messages declare the same non-zero id, their configuration is unchanged and
    :meth:`_hash_message` is not consulted. ``None`` matches no message, so the
    next one is hashed.
    """

    STREAMING_DIMS: typing.ClassVar[tuple[str, ...]] = _STREAMING_DIMS
    """Fallback stream dimension for messages that do not declare one.

    Consulted only when :attr:`~ezmsg.util.messages.axisarray.AxisArray.stream_dim`
    is ``None``. ``("time",)`` is right for a raw signal and wrong downstream of
    a windowing stage, where the message is ``(win, time, ch)`` and ``win`` is
    what grows; such a processor sets ``("win",)``.

    Prefer teaching the producer to declare ``stream_dim``. That puts the answer
    in the one place that knows it, rather than asking each consumer to guess
    about a message it did not create.
    """

    @classmethod
    def get_state_type(cls) -> type[StateType]:
        return _get_base_processor_state_type(cls)

    @property
    def state(self) -> StateType:
        return self._state

    @state.setter
    def state(self, state: StateType | bytes | None) -> None:
        if state is not None:
            # State restored from elsewhere may not belong to the stream's current
            # config_id: hash the next message and compare it with the restored hash.
            self._config_id = None
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state  # type: ignore

    def _hash_message(self, message: typing.Any) -> int:
        """
        Check if the message metadata indicates a need for state reset.

        The default is the message's declared
        :attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id`, so a
        processor resets on *every* change of it: layout, dtype, array backend,
        axis units, and anything a source folded into the id (e.g. value scaling).

        For a message that does not declare one (``config_id == 0``, or not an
        ``AxisArray``) it falls back to the layout hash, :meth:`_message_hash`,
        computed on every message. That keys on the stream's *shape and
        identity* but not its per-message extent: the message key, its dims, the
        array backend and dtype, the length of every dimension except the one it streams
        along, the coordinate values on those dimensions, and the gain and
        offset of any linear axis among them. For a message that does not
        declare :attr:`~ezmsg.util.messages.axisarray.AxisArray.stream_dim` the
        streaming dimension falls back to :attr:`STREAMING_DIMS`.

        Override only to *narrow* the default: to veto resets on id changes the
        state genuinely does not depend on, e.g. a processor whose state does
        not depend on channel identity. Prefer returning :meth:`_message_hash`
        with the appropriate arguments over rebuilding the hash from scratch, so
        that the axis-value coverage is not silently lost. An override replaces
        the id, so it must itself cover every change the state *does* depend on.

        Processors whose state is insensitive to everything may return a
        constant. All processors' initial state has ``.hash = -1``, so any
        constant forces exactly one reset on the first message.

        On a stream that declares ``config_id`` this is called only when the id
        changes; on a stream that does not, it is called on every message.
        """
        # Ids are positive 63-bit, so they never collide with the 0 / -1 sentinels.
        if isinstance(message, AxisArray) and message.config_id:
            return message.config_id
        return self._message_hash(message)

    def _message_hash(
        self,
        message: typing.Any,
        *,
        exclude_dims: typing.Iterable[str] | None = None,
        include_key: bool = True,
        extra: typing.Iterable[typing.Any] = (),
    ) -> int:
        """
        Hash the parts of an ``AxisArray`` that a cached state can depend on.

        Folds in, in dimension order:

        * ``message.key`` (unless *include_key* is False), ``message.dims``, the
          array backend (top-level module of ``type(message.data)``: numpy, mlx, ...)
          and the data's ``dtype`` (``None`` for data that has none)
        * for each dimension other than the stream dimension: its length, plus
          either the coordinate axis's
          :attr:`~ezmsg.util.messages.axisarray.CoordinateAxis.fingerprint` or a
          linear axis's ``gain`` **and** ``offset``
        * for the stream dimension: only the ``gain``

        ``offset`` is dropped for the stream dimension alone, where it simply
        counts off elapsed samples. Everywhere else it locates the axis and a
        change in it is a configuration change: a spectrum whose ``freq`` axis
        moves from 5-25 Hz to 70-90 Hz keeps the same gain and the same length,
        and is only distinguishable by its offset.

        The fingerprint is what makes a channel *relabel* at a fixed channel
        count visible. Without it a filter keeps per-channel state belonging to
        channels that are no longer there, and the first samples of the new ones
        come out dominated by the old ones' history.

        The stream dimension is ``message.stream_dim`` when declared, else
        :attr:`STREAMING_DIMS`. Naming a dimension the message does not have is
        harmless -- nothing matches, so nothing is excluded.

        Non-``AxisArray`` messages hash to a constant, giving the same
        reset-once-then-never behaviour those processors had before.

        :param exclude_dims: Further dimensions to leave out, *in addition to*
            the stream dimension. Use for a processor whose state genuinely does
            not depend on a dimension's identity.
        :param include_key: Set False for a processor whose state depends only
            on shape, so that switching streams does not force a reset.
        :param extra: Additional hashable values to fold in.
        """
        if not isinstance(message, AxisArray):
            return 0

        # The producer renamed the dims and so is the only party that reliably
        # knows which one grows; fall back to the class default when it is silent.
        stream_dim = message.stream_dim
        if stream_dim is None:
            exclude = self.STREAMING_DIMS if exclude_dims is None else (*self.STREAMING_DIMS, *exclude_dims)
        elif exclude_dims is None:
            exclude = (stream_dim,)
        else:
            exclude = (stream_dim, *exclude_dims)

        # Hoisted out of the loop: this runs on every message of a stream that
        # does not declare config_id, so the repeated attribute lookups are
        # worth removing. A tuple rather
        # than a set for `exclude` -- it holds one or two entries in practice,
        # where a linear scan beats building a set.
        dims = message.dims
        axes = message.axes
        data = message.data
        shape = data.shape
        parts: list[typing.Any] = [message.key] if include_key else []
        # The array backend (numpy, mlx, torch, ...) and dtype, as ezmsg's compute_config_id reads them.
        parts.append((tuple(dims), type(data).__module__.partition(".")[0], getattr(data, "dtype", None)))

        for idx, dim in enumerate(dims):
            axis = axes.get(dim)
            if dim in exclude:
                gain = getattr(axis, "gain", None)
                if gain is not None:
                    parts.append((dim, gain))
                continue
            parts.append((dim, shape[idx]))
            # A CoordinateAxis identifies itself by its values; a LinearAxis by
            # gain *and* offset, which together say where the axis starts and
            # how far it steps; a dimension with no axis, only by its length.
            # Asked for in that order so a coordinate axis costs one lookup:
            # fetching `gain` first made it pay a failed one it never used.
            fingerprint = getattr(axis, "fingerprint", None)
            if fingerprint is not None:
                parts.append(fingerprint)
            else:
                gain = getattr(axis, "gain", None)
                if gain is not None:
                    parts.append((gain, axis.offset))

        parts.extend(extra)
        return hash(tuple(parts))

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

    VERIFY_OUTPUT_CONFIG: typing.ClassVar[bool] = True
    """Check that the output's ``config_id`` changes whenever its layout does.

    Downstream processors trust :attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id`
    and keep their state while it is unchanged, so a processor whose output layout
    changes (e.g. a settings change selects different channels) must set a new one.
    When enabled, the first declared ``AxisArray`` output after a reset (`_reset_state`,
    not `_reset_segment`) and any output whose ``config_id`` changes are hashed
    (:meth:`_message_hash`); a layout change under an unchanged ``config_id`` is warned
    about once per class, and from then on the outputs carry a corrected ``config_id``.
    Other outputs cost only a comparison of ``config_id``, so classes do not need to
    opt out for speed. Set False to turn the check off.

    Known limit: a processor that changes its output layout with no reset (e.g. from a
    field in ``NONRESET_SETTINGS_FIELDS``, or from adaptive state) is not caught.
    """

    WARN_UNDECLARED_CONFIG: typing.ClassVar[bool] = True
    """Warn, once per class, on receiving an ``AxisArray`` with no ``config_id``.

    Set False only on a processor whose job is to receive such messages, such as
    the adapter that stamps a ``config_id`` on them. Read only for an undeclared message.
    """

    _out_hash: int | None = None
    _out_config_id: int = 0
    _out_changes: int = 0
    """How many output layout changes arrived without a new ``config_id``."""
    _verify_pending: bool = True
    """Hash the next declared ``AxisArray`` output; set by a full reset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hash = -1
        self._segment: int | None = None
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()
        # TODO: Enforce that StateType has .hash: int field.

    def _request_reset(self) -> None:
        # Invalidate the hash so the next __call__ / __acall__ triggers
        # _reset_state(message) even if the message metadata hasn't changed.
        self._hash = -1

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

    async def _areset_state(self, message: typing.Any) -> None:
        """
        Async variant of `_reset_state`. Override this if reset requires async work;
        in that case `_reset_state` should bridge via `run_coroutine_sync(self._areset_state(message))`.
        """
        return self._reset_state(message)

    def _reset_segment(self, message: typing.Any) -> None:
        """
        Start again because ``message`` does not continue the previous one.

        Called when :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment` differs from the
        previous message's while the layout (`_hash_message`) is unchanged: the source reported
        a gap in time, or a reconfiguration this processor cannot otherwise see. Anything carried
        over from before the break -- filter memory, a partly filled window -- no longer describes
        the samples that precede this message.

        The default is a full `_reset_state`. Override it to keep state that is still valid
        across a break, such as long-run statistics or a tracked frequency, and drop the rest.
        """
        self._reset_state(message)

    async def _areset_segment(self, message: typing.Any) -> None:
        """Async variant of `_reset_segment`. A `_reset_segment` override is honoured here too."""
        if type(self)._reset_segment is not BaseStatefulProcessor._reset_segment:
            return self._reset_segment(message)
        await self._areset_state(message)

    def _changed_hash(self, message: typing.Any, config_id: int) -> int | None:
        """``_hash_message(message)`` if it differs from the current hash, else None."""
        if (
            not config_id
            and self.WARN_UNDECLARED_CONFIG
            and type(self) not in _warned_undeclared
            and isinstance(message, AxisArray)
        ):
            _warned_undeclared.add(type(self))
            logger.warning(
                f"{type(self).__name__} is receiving AxisArray messages with no config_id "
                f"(key={message.key!r}), so it is falling back to hashing the layout on every message. "
                "Fix this by having the source set config_id "
                "(ezmsg.util.messages.axisarray.compute_config_id) "
                "or by inserting a StreamStamp node (ezmsg-sigproc) after the source."
            )
        msg_hash = self._hash_message(message)
        self._config_id = config_id
        return None if msg_hash == self._hash else msg_hash

    def _prepare_state(self, message: typing.Any) -> None:
        """Reset before processing ``message`` if its layout or its segment changed.

        A message whose ``config_id`` is declared and unchanged is not hashed; a
        hash of -1 (set by `_request_reset`, or directly) still forces a reset.
        """
        config_id = getattr(message, "config_id", 0)
        segment = getattr(message, "segment", None)
        msg_hash = None
        if config_id != self._config_id or not config_id or self._hash == -1:
            msg_hash = self._changed_hash(message, config_id)
        if msg_hash is not None:
            self._reset_state(message)
            self._hash = msg_hash
            self._verify_pending = True
        elif segment != self._segment:
            self._reset_segment(message)
        self._segment = segment

    async def _aprepare_state(self, message: typing.Any) -> None:
        config_id = getattr(message, "config_id", 0)
        segment = getattr(message, "segment", None)
        msg_hash = None
        if config_id != self._config_id or not config_id or self._hash == -1:
            msg_hash = self._changed_hash(message, config_id)
        if msg_hash is not None:
            await self._areset_state(message)
            self._hash = msg_hash
            self._verify_pending = True
        elif segment != self._segment:
            await self._areset_segment(message)
        self._segment = segment

    def _verify_output(self, output: typing.Any) -> typing.Any:
        """Return ``output``, with a corrected ``config_id`` once its layout has changed unannounced.

        See :attr:`VERIFY_OUTPUT_CONFIG`. The output is hashed only when its layout can
        have changed: it is the first declared output after a full reset, or its
        ``config_id`` differs from the previous output's. A processor that changes its
        output layout with no reset is not caught. The remembered output hash and id
        survive `_request_reset` and a state restore: they describe what downstream last
        received, and a settings change is exactly when an unannounced change happens.
        """
        if not isinstance(output, AxisArray):
            return output
        config_id = output.config_id
        if not config_id:
            return output
        if self._verify_pending or config_id != self._out_config_id:
            self._verify_pending = False
            out_hash = self._message_hash(output)
            if out_hash != self._out_hash:
                if config_id == self._out_config_id:
                    self._out_changes += 1
                    if type(self) not in _warned_unannounced:
                        _warned_unannounced.add(type(self))
                        logger.warning(
                            f"{type(self).__name__}'s output layout changed without a new config_id "
                            f"(key={output.key!r}), so downstream processors would keep stale state. "
                            "baseproc is correcting the config_id on its outputs. Fix this by having "
                            f"{type(self).__name__} set a new config_id on its outputs whenever its output "
                            "layout changes (ezmsg.util.messages.axisarray.chain_config_id)."
                        )
                self._out_hash = out_hash
            self._out_config_id = config_id
        if self._out_changes:
            return replace(output, config_id=chain_config_id(config_id, self._out_changes))
        return output

    @abstractmethod
    def _process(self, message: typing.Any) -> typing.Any: ...

    def __call__(self, message: typing.Any) -> typing.Any:
        self._prepare_state(message)
        output = self._process(message)
        return self._verify_output(output) if self.VERIFY_OUTPUT_CONFIG else output

    async def __acall__(self, message: typing.Any) -> typing.Any:
        await self._aprepare_state(message)
        output = await self._aprocess(message)
        return self._verify_output(output) if self.VERIFY_OUTPUT_CONFIG else output

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

    def _request_reset(self) -> None:
        # Force the next __acall__ back into the uninitialized branch.
        self._hash = -1

    @abstractmethod
    def _reset_state(self) -> None:
        """
        Reset internal state upon first call.
        """
        ...

    async def _areset_state(self) -> None:
        """
        Async variant of `_reset_state`. Override this if reset requires async work;
        in that case `_reset_state` should bridge via `run_coroutine_sync(self._areset_state())`.
        """
        return self._reset_state()

    async def __acall__(self) -> MessageOutType:
        if self._hash == -1:
            await self._areset_state()
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

    VERIFY_OUTPUT_CONFIG = False  # No output to verify.

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
        MessageInType,
        MessageOutType | None,
        StateType,
    ],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    @abstractmethod
    def partial_fit(self, message: AxisArray) -> None: ...

    async def apartial_fit(self, message: AxisArray) -> None:
        """Override me if you need async partial fitting."""
        return self.partial_fit(message)

    def __call__(self, message: MessageInType) -> MessageOutType | None:
        if is_sample_message(message):
            warnings.warn(
                f"{self.__class__.__name__}.__call__() received a sample message "
                "(AxisArray with 'trigger' in attrs). Auto-routing to partial_fit "
                "has been removed. Use partial_fit() for training only, or "
                "partial_fit_transform() for training + inference.",
                UserWarning,
                stacklevel=2,
            )
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> MessageOutType | None:
        if is_sample_message(message):
            warnings.warn(
                f"{self.__class__.__name__}.__acall__() received a sample message "
                "(AxisArray with 'trigger' in attrs). Auto-routing to partial_fit "
                "has been removed. Use apartial_fit() for training only, or "
                "apartial_fit_transform() for training + inference.",
                UserWarning,
                stacklevel=2,
            )
        return await super().__acall__(message)

    def partial_fit_transform(self, message: AxisArray) -> MessageOutType:
        """Train on the message, then run inference and return the result."""
        self._prepare_state(message)
        self.partial_fit(message)
        output = self._process(message)
        return self._verify_output(output) if self.VERIFY_OUTPUT_CONFIG else output

    async def apartial_fit_transform(self, message: AxisArray) -> MessageOutType:
        """Async variant of partial_fit_transform."""
        await self._aprepare_state(message)
        await self.apartial_fit(message)
        output = await self._aprocess(message)
        return self._verify_output(output) if self.VERIFY_OUTPUT_CONFIG else output


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
        await self._aprepare_state(message)
        output = await self._aprocess(message)
        return self._verify_output(output) if self.VERIFY_OUTPUT_CONFIG else output
