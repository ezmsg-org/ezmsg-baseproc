"""Stateful processor base classes for ezmsg."""

import pickle
import typing
import warnings
from abc import ABC, abstractmethod

from ezmsg.util.messages.axisarray import AxisArray

from .processor import (
    BaseProcessor,
    BaseProducer,
    _get_base_processor_message_in_type,
)
from .protocols import MessageInType, MessageOutType, SettingsType, StateType
from .util.asio import run_coroutine_sync
from .util.message import is_sample_message
from .util.typeresolution import resolve_typevar


def _get_base_processor_state_type(cls: type) -> type:
    try:
        return resolve_typevar(cls, StateType)
    except TypeError as e:
        raise TypeError(
            f"Could not resolve state type for {cls}. Ensure that the class is properly annotated with a StateType."
        ) from e


def _shape_slice(dims: list[str], exclude: tuple[str, ...]) -> slice | None:
    """A slice selecting the dimensions whose *length* feeds the hash.

    Only worth having when the excluded dimensions sit at one end, which is the
    case for every layout in practice -- the chunk dimension leads (``time, ch``;
    ``win, time, ch``) or, after a transpose, trails. Anything else returns
    ``None`` and simply declines the fast path rather than paying a comprehension
    per message to reproduce ``shape``.
    """
    dropped = [ix for ix, dim in enumerate(dims) if dim in exclude]
    if not dropped:
        return slice(None)
    if dropped == list(range(len(dropped))):
        return slice(len(dropped), None)
    if dropped == list(range(len(dims) - len(dropped), len(dims))):
        return slice(None, -len(dropped))
    return None


def _axis_value(axis: typing.Any) -> typing.Any:
    """What the hash reads off an axis, for comparing two distinct objects.

    ``None`` means "nothing comparable" -- a dimension with no axis, or one whose
    axis is neither coordinate nor linear. Callers treat that as a mismatch and
    fall back to recomputing, which is the conservative direction.
    """
    fingerprint = getattr(axis, "fingerprint", None)
    if fingerprint is not None:
        return fingerprint
    gain = getattr(axis, "gain", None)
    return None if gain is None else (gain, axis.offset)


def _build_witness(
    message: typing.Any,
    dims: list[str],
    shape: tuple[int, ...],
    exclude: tuple[str, ...],
    exclude_dims: typing.Iterable[str] | None,
    include_key: bool,
    extra: typing.Iterable[typing.Any],
    result: int,
) -> tuple | None:
    """Compile a validator for the message this hash was derived from.

    The layout is fixed for the life of a witness, so the decisions that depend
    on it -- which dimensions to skip, where the kept lengths sit, whether the
    key matters -- are made once here and baked into a closure's defaults rather
    than re-derived per message. Unpacking a witness tuple and branching on it
    cost more than the comparisons it was guarding.

    Returns ``None`` for layouts the fast path declines to handle; the caller
    then simply always recomputes.
    """
    sl = _shape_slice(dims, exclude)
    if sl is None:
        return None
    axes = message.axes
    # Each kept dimension is recorded twice: the axis *object*, which settles it
    # in one pointer comparison when the producer reuses its per-stream axes, and
    # the axis *value*, for when it cannot -- most importantly on the far side of
    # a process boundary, where unpickling hands out a new object per message but
    # the fingerprint rides along already computed.
    kept = tuple((dim, axes.get(dim), _axis_value(axes.get(dim))) for dim in dims if dim not in exclude)
    # The chunk axis is a new object every message on any path -- its offset
    # advances -- so it is compared by value always.
    chunked = tuple((dim, getattr(axes.get(dim), "gain", None)) for dim in dims if dim in exclude)
    w_dims, w_key, w_chunk = list(dims), message.key, message.chunk_dim

    if len(kept) == 1 and len(chunked) == 1 and chunked[0][1] is not None and kept[0][2] is not None:
        # One coordinate axis to pin down and one chunk axis carrying the sample
        # rate. This is `(time, ch)`, and `(win, time, ch)` once `time` is also
        # excluded -- between them, nearly every message in a graph.
        (kept_dim, kept_axis, kept_value), (chunk_dim, chunk_gain) = kept[0], chunked[0]
        kept_ix = dims.index(kept_dim)

        def validate(
            msg: typing.Any,
            _kd: str = kept_dim,
            _ka: typing.Any = kept_axis,
            _kv: typing.Any = kept_value,
            _cd: str = chunk_dim,
            _cg: float = chunk_gain,
            _kix: int = kept_ix,
            _klen: int = shape[kept_ix],
            _dims: list[str] = w_dims,
            _key: str = w_key,
            _chunk: str | None = w_chunk,
            _check_key: bool = include_key,
        ) -> bool:
            axes = msg.axes
            try:
                axis = axes[_kd]
                if axis is not _ka and _axis_value(axis) != _kv:
                    return False
                return (
                    axes[_cd].gain == _cg
                    and msg.data.shape[_kix] == _klen
                    and msg.chunk_dim == _chunk
                    and msg.dims == _dims
                    and (not _check_key or msg.key == _key)
                )
            except (AttributeError, KeyError, IndexError):
                # The layout shifted out from under the specialisation: the chunk
                # axis stopped being linear (an irregular-rate stream switches to
                # a CoordinateAxis), a dimension lost its axis, or the data lost a
                # dimension. Decline and let the full hash sort it out. Costs
                # nothing while it does not fire, which is always in a steady
                # stream, and `_build_witness` re-specialises on the next change.
                return False
    else:

        def validate(
            msg: typing.Any,
            _kept: tuple = kept,
            _chunked: tuple = chunked,
            _sl: slice = sl,
            _ks: tuple = shape[sl],
            _dims: list[str] = w_dims,
            _key: str = w_key,
            _chunk: str | None = w_chunk,
            _check_key: bool = include_key,
        ) -> bool:
            axes = msg.axes
            for dim, axis, value in _kept:
                incoming = axes.get(dim)
                if incoming is not axis and (value is None or _axis_value(incoming) != value):
                    return False
            for dim, gain in _chunked:
                # No `is not None` shortcut on the axis: an excluded dimension
                # *losing* its axis drops a term from the hash, so absence has to
                # compare unequal to a gain rather than be skipped.
                if getattr(axes.get(dim), "gain", None) != gain:
                    return False
            return (
                msg.data.shape[_sl] == _ks
                and msg.chunk_dim == _chunk
                and msg.dims == _dims
                and (not _check_key or msg.key == _key)
            )

    return (validate, None if exclude_dims is None else tuple(exclude_dims), include_key, tuple(extra), result)


class Stateful(ABC, typing.Generic[StateType]):
    """
    Mixin class for stateful processors. DO NOT use this class directly.
    Used to enforce that the processor/producer has a state attribute and stateful_op method.
    """

    _state: StateType

    _hash_witness: typing.ClassVar[tuple | None] = None
    """The objects the last :meth:`_message_hash` result was derived from.

    Recomputing the hash means walking the dims, reaching into the axes and
    building a tuple to hash -- and in a steady stream the answer is the same
    every time. A producer that builds its per-stream axes once and replaces only
    the chunk axis per message (the template idiom every ezmsg source uses) hands
    every consumer the *same coordinate axis object* for the life of the stream,
    so identity is enough to prove the hash cannot have changed.

    Shadowed by an instance attribute once set. ``None`` means "no witness" and
    is the safe state: it costs a full recomputation, never a wrong answer.
    """

    STREAMING_DIMS: typing.ClassVar[tuple[str, ...]] = ("time",)
    """Fallback chunk dimension for messages that do not declare one.

    Consulted only when :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`
    is ``None``. ``("time",)`` is right for a raw signal and wrong downstream of
    a windowing stage, where the message is ``(win, time, ch)`` and ``win`` is
    what grows; such a processor sets ``("win",)``.

    Prefer teaching the producer to declare ``chunk_dim``. That puts the answer
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
            # The witness describes the message the *previous* state was built
            # from. Restoring state from elsewhere leaves it describing nothing,
            # and a match against it would return a hash for state that is gone.
            self._hash_witness = None
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state  # type: ignore

    def _hash_message(self, message: typing.Any) -> int:
        """
        Check if the message metadata indicates a need for state reset.

        For a message that declares :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`,
        the default keys on everything describing the stream's *shape and
        identity* but not its per-chunk extent: the message key, its dims, the
        length of every dimension except the one it is a chunk along, the
        coordinate values on those dimensions, and the gain and offset of any
        linear axis among them. See :meth:`_message_hash`.

        A message that does not declare it falls back to
        :attr:`STREAMING_DIMS`. That fallback is a guess, and a wrong guess is
        not a small error: name a dimension that is actually stable and the
        processor stops noticing real changes to it; name one that grows and it
        resets on every message. It is right for the common ``(time, ch)``
        stream and wrong downstream of a windowing stage.

        Override to add something the default cannot know about -- a dtype the
        state depends on, a value derived from the processor's own state -- or
        to *narrow* it, for a processor whose state genuinely does not depend on
        channel identity. In either case prefer calling :meth:`_message_hash`
        with the appropriate arguments over rebuilding the hash from scratch, so
        that the axis-value coverage is not silently lost.

        Processors whose state is insensitive to everything may return a
        constant. All processors' initial state has ``.hash = -1``, so any
        constant forces exactly one reset on the first message.
        """
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

        * ``message.key`` (unless *include_key* is False) and ``message.dims``
        * for each dimension other than the chunk dimension: its length, plus
          either the coordinate axis's
          :attr:`~ezmsg.util.messages.axisarray.CoordinateAxis.fingerprint` or a
          linear axis's ``gain`` **and** ``offset``
        * for the chunk dimension: only the ``gain``

        ``offset`` is dropped for the chunk dimension alone, where it simply
        counts off elapsed samples. Everywhere else it locates the axis and a
        change in it is a configuration change: a spectrum whose ``freq`` axis
        moves from 5-25 Hz to 70-90 Hz keeps the same gain and the same length,
        and is only distinguishable by its offset.

        The fingerprint is what makes a channel *relabel* at a fixed channel
        count visible. Without it a filter keeps per-channel state belonging to
        channels that are no longer there, and the first samples of the new ones
        come out dominated by the old ones' history.

        The chunk dimension is ``message.chunk_dim`` when declared, else
        :attr:`STREAMING_DIMS`. Naming a dimension the message does not have is
        harmless -- nothing matches, so nothing is excluded.

        Non-``AxisArray`` messages hash to a constant, giving the same
        reset-once-then-never behaviour those processors had before.

        :param exclude_dims: Further dimensions to leave out, *in addition to*
            the chunk dimension. Use for a processor whose state genuinely does
            not depend on a dimension's identity.
        :param include_key: Set False for a processor whose state depends only
            on shape, so that switching streams does not force a reset.
        :param extra: Additional hashable values to fold in.
        """
        if not isinstance(message, AxisArray):
            return 0

        # The witness is checked before anything else is derived: if nothing it
        # was built from has changed identity, the answer cannot have changed.
        # Its validator runs first because it is the most discriminating -- a
        # producer that rebuilds its axes fails on one `is` rather than after the
        # bookkeeping comparisons.
        witness = self._hash_witness
        if (
            witness is not None
            and witness[0](message)
            and witness[2] is include_key
            and witness[3] == extra
            and (witness[1] is None if exclude_dims is None else witness[1] == tuple(exclude_dims))
        ):
            return witness[4]

        # The producer renamed the dims and so is the only party that reliably
        # knows which one grows; fall back to the class default when it is silent.
        chunk_dim = message.chunk_dim
        if chunk_dim is None:
            exclude = self.STREAMING_DIMS if exclude_dims is None else (*self.STREAMING_DIMS, *exclude_dims)
        elif exclude_dims is None:
            exclude = (chunk_dim,)
        else:
            exclude = (chunk_dim, *exclude_dims)

        # Hoisted out of the loop: this runs on every message of every stream,
        # so the repeated attribute lookups are worth removing. A tuple rather
        # than a set for `exclude` -- it holds one or two entries in practice,
        # where a linear scan beats building a set.
        dims = message.dims
        axes = message.axes
        shape = message.data.shape
        parts: list[typing.Any] = [message.key] if include_key else []
        parts.append(tuple(dims))

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
        result = hash(tuple(parts))

        # Rebuild the witness when the answer changed -- a reset is about to run,
        # so the cost lands where it is already expensive -- or when there is no
        # witness at all, which is how one is established after a state restore.
        if witness is None or result != getattr(self, "_hash", None):
            self._hash_witness = _build_witness(message, dims, shape, exclude, exclude_dims, include_key, extra, result)
        return result

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

    def _request_reset(self) -> None:
        # Invalidate the hash so the next __call__ / __acall__ triggers
        # _reset_state(message) even if the message metadata hasn't changed.
        # The witness has to go with it: it would otherwise answer with the hash
        # this line is trying to invalidate.
        self._hash = -1
        self._hash_witness = None

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
            await self._areset_state(message)
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
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        self.partial_fit(message)
        return self._process(message)

    async def apartial_fit_transform(self, message: AxisArray) -> MessageOutType:
        """Async variant of partial_fit_transform."""
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            await self._areset_state(message)
            self._hash = msg_hash
        await self.apartial_fit(message)
        return await self._aprocess(message)


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
            await self._areset_state(message)
            self._hash = msg_hash
        return await self._aprocess(message)
