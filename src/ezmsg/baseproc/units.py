"""Base Unit classes for ezmsg integration."""

import math
import traceback
import typing
from abc import ABC, abstractmethod

import ezmsg.core as ez
from ezmsg.util.generator import GenState
from ezmsg.util.messages.axisarray import AxisArray

from .composite import CompositeProcessor
from .processor import BaseConsumer, BaseProducer, BaseTransformer
from .protocols import MessageInType, MessageOutType, SettingsType
from .stateful import BaseAdaptiveTransformer, BaseStatefulConsumer, BaseStatefulTransformer
from .util.message import SampleMessage
from .util.profile import profile_subpub
from .util.typeresolution import resolve_typevar

# --- Type variables for Unit classes ---
ProducerType = typing.TypeVar("ProducerType", bound=BaseProducer)
ConsumerType = typing.TypeVar("ConsumerType", bound=BaseConsumer | BaseStatefulConsumer)
TransformerType = typing.TypeVar(
    "TransformerType",
    bound=BaseTransformer | BaseStatefulTransformer | CompositeProcessor,
)
AdaptiveTransformerType = typing.TypeVar("AdaptiveTransformerType", bound=BaseAdaptiveTransformer)


def get_base_producer_type(cls: type) -> type:
    return resolve_typevar(cls, ProducerType)


def get_base_consumer_type(cls: type) -> type:
    return resolve_typevar(cls, ConsumerType)


def get_base_transformer_type(cls: type) -> type:
    return resolve_typevar(cls, TransformerType)


def get_base_adaptive_transformer_type(cls: type) -> type:
    return resolve_typevar(cls, AdaptiveTransformerType)


# --- Base classes for ezmsg Unit with specific processing capabilities ---
class BaseProducerUnit(ez.Unit, ABC, typing.Generic[SettingsType, MessageOutType, ProducerType]):
    """
    Base class for producer units -- i.e. units that generate messages without consuming inputs.
    Implement a new Unit as follows:

    class CustomUnit(BaseProducerUnit[
        CustomProducerSettings,    # SettingsType
        AxisArray,                 # MessageOutType
        CustomProducer,            # ProducerType
    ]):
        SETTINGS = CustomProducerSettings

    ... that's all!

    Where CustomProducerSettings, and CustomProducer are custom implementations of ez.Settings,
    and BaseProducer or BaseStatefulProducer, respectively.
    """

    INPUT_SETTINGS = ez.InputStream(SettingsType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    async def initialize(self) -> None:
        self.create_producer()

    def create_producer(self) -> None:
        # self.producer: ProducerType
        """Create the producer instance from settings."""
        producer_type = get_base_producer_type(self.__class__)
        self.producer = producer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the producer.
        Child classes that wish to have fine-grained control over whether the
        core producer resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)  # type: ignore
        self.create_producer()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            out = await self.producer.__acall__()
            if out is not None:  # and math.prod(out.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, out


class BaseProcessorUnit(ez.Unit, ABC, typing.Generic[SettingsType]):
    """
    Base class for processor units -- i.e. units that process messages.
    This is an abstract base class that provides common functionality for consumer and transformer
    units. You probably do not want to inherit from this class directly as you would need to define
    a custom implementation of `create_processor`.
    Refer instead to BaseConsumerUnit or BaseTransformerUnit.
    """

    INPUT_SETTINGS = ez.InputStream(SettingsType)

    async def initialize(self) -> None:
        self.create_processor()

    @abstractmethod
    def create_processor(self) -> None: ...

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the processor.
        Child classes that wish to have fine-grained control over whether the
        core processor resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)  # type: ignore
        self.create_processor()


class BaseConsumerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[SettingsType, MessageInType, ConsumerType],
):
    """
    Base class for consumer units -- i.e. units that receive messages but do not return results.
    Implement a new Unit as follows:

    class CustomUnit(BaseConsumerUnit[
        CustomConsumerSettings,    # SettingsType
        AxisArray,                 # MessageInType
        CustomConsumer,            # ConsumerType
    ]):
        SETTINGS = CustomConsumerSettings

    ... that's all!

    Where CustomConsumerSettings and CustomConsumer are custom implementations of:
    - ez.Settings for settings
    - BaseConsumer or BaseStatefulConsumer for the consumer implementation
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)

    def create_processor(self):
        # self.processor: ConsumerType[SettingsType, MessageInType, StateType]
        """Create the consumer instance from settings."""
        consumer_type = get_base_consumer_type(self.__class__)
        self.processor = consumer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_signal(self, message: MessageInType):
        """
        Consume the message.
        Args:
            message: Input message to be consumed
        """
        await self.processor.__acall__(message)


class BaseTransformerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, TransformerType],
):
    """
    Base class for transformer units -- i.e. units that transform input messages into output messages.
    Implement a new Unit as follows:

    class CustomUnit(BaseTransformerUnit[
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageInType
        AxisArray,                    # MessageOutType
        CustomTransformer,            # TransformerType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerSettings and CustomTransformer are custom implementations of:
    - ez.Settings for settings
    - One of these transformer types:
      * BaseTransformer
      * BaseStatefulTransformer
      * CompositeProcessor
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    def create_processor(self):
        # self.processor: TransformerType[SettingsType, MessageInType, MessageOutType, StateType]
        """Create the transformer instance from settings."""
        transformer_type = get_base_transformer_type(self.__class__)
        self.processor = transformer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageInType) -> typing.AsyncGenerator:
        result = await self.processor.__acall__(message)
        if result is not None:  # and math.prod(result.data.shape) > 0:
            yield self.OUTPUT_SIGNAL, result


class BaseAdaptiveTransformerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, AdaptiveTransformerType],
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)
    INPUT_SIGNAL = ez.InputStream(MessageInType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    def create_processor(self) -> None:
        # self.processor: AdaptiveTransformerType[SettingsType, MessageInType, MessageOutType, StateType]
        """Create the adaptive transformer instance from settings."""
        adaptive_transformer_type = get_base_adaptive_transformer_type(self.__class__)
        self.processor = adaptive_transformer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageInType) -> typing.AsyncGenerator:
        result = await self.processor.__acall__(message)
        if result is not None:  # and math.prod(result.data.shape) > 0:
            yield self.OUTPUT_SIGNAL, result

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        await self.processor.apartial_fit(msg)


# Legacy class
class GenAxisArray(ez.Unit):
    STATE = GenState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    INPUT_SETTINGS = ez.InputStream(ez.Settings)

    async def initialize(self) -> None:
        self.construct_generator()

    # Method to be implemented by subclasses to construct the specific generator
    def construct_generator(self):
        raise NotImplementedError

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: ez.Settings) -> None:
        """
        Update unit settings and reset generator.
        Note: Not all units will require a full reset with new settings.
        Override this method to implement a selective reset.

        Args:
            msg: Instance of SETTINGS object.
        """
        self.apply_settings(msg)
        self.construct_generator()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        try:
            ret = self.STATE.gen.send(message)
            if math.prod(ret.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, ret
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Generator closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
