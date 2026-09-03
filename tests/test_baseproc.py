"""Tests for ezmsg-baseproc package."""

import dataclasses
import pickle
from types import NoneType
from typing import Any

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAsyncTransformer,
    BaseConsumer,
    BaseProcessor,
    BaseProducer,
    BaseStatefulConsumer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    BaseStatefulTransformer,
    BaseTransformer,
    CompositeProcessor,
    CompositeProducer,
    SampleTriggerMessage,
    _get_base_processor_message_in_type,
    _get_base_processor_message_out_type,
    _get_base_processor_settings_type,
    _get_base_processor_state_type,
    _get_processor_message_type,
    processor_state,
)
from ezmsg.baseproc.stateful import Stateful

# -- Mock Classes for Testing --


@dataclasses.dataclass
class MockSettings:
    param1: int = 10
    param2: str = "test"


@processor_state
class MockState:
    iterations: int = 0
    fit_count: int = 0
    hash: int = -1


# Simple mock message types for testing type compatibility
class MockMessageA:
    pass


class MockMessageB(MockMessageA):
    pass


class MockMessageC:
    pass


# Mock processors for testing
class MockProcessor(BaseProcessor[MockSettings, MockMessageA, MockMessageB]):
    def _process(self, message: MockMessageA) -> MockMessageB:
        return MockMessageB()


class DeepMockProcessor(MockProcessor):
    def _process(self, message: MockMessageA) -> MockMessageB:
        return MockMessageB()


class MockProducer(BaseProducer[MockSettings, MockMessageA]):
    async def _produce(self) -> MockMessageA:
        return MockMessageA()


class MockConsumer(BaseConsumer[MockSettings, MockMessageB]):
    def _process(self, message: MockMessageB) -> None:
        pass


class MockTransformer(BaseTransformer[MockSettings, MockMessageA, MockMessageC]):
    def _process(self, message: MockMessageA) -> MockMessageC:
        return MockMessageC()


class DeepMockTransformer(MockTransformer):
    pass


class DeeperMockTransformer(DeepMockTransformer):
    pass


class MockStatefulProcessor(BaseStatefulProcessor[MockSettings, MockMessageA, MockMessageB, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()


class DeepMockStatefulProcessor(MockStatefulProcessor):
    pass


class MockStatefulProducer(BaseStatefulProducer[MockSettings, MockMessageA, MockState]):
    def _reset_state(self) -> None:
        self._state.iterations = 0

    async def _produce(self) -> MockMessageA:
        self._state.iterations += 1
        return MockMessageA()


class MockStatefulConsumer(BaseStatefulConsumer[MockSettings, MockMessageA, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> None:
        self._state.iterations += 1


class MockStatefulTransformer(BaseStatefulTransformer[MockSettings, MockMessageA, MockMessageC, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> MockMessageC:
        self._state.iterations += 1
        return MockMessageC()


class MockAdaptiveTransformer(BaseAdaptiveTransformer[MockSettings, MockMessageA, MockMessageB, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()

    def partial_fit(self, message: AxisArray) -> None:
        self._state.fit_count += 1


class MockAsyncTransformer(BaseAsyncTransformer[MockSettings, MockMessageA, MockMessageB, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    async def _aprocess(self, message: MockMessageA) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()


# Mock CompositeProcessor examples
class ValidSingleCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"processor": MockProcessor(settings=settings)}


class ValidMultipleCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


class EmptyCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {}


class InvalidOutputCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"transformer": MockTransformer(settings=settings)}


class InvalidInputCompositeProcessor(CompositeProcessor[MockSettings, None, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"transformer": MockTransformer(settings=settings)}


class InvalidProducerNotFirstCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class InvalidConsumerNotLastCompositeProcessor(CompositeProcessor[MockSettings, MockMessageB, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "consumer": MockConsumer(settings=settings),
            "transformer": MockTransformer(settings=settings),
        }


class InvalidConsumerProducerCompositeProcessor(CompositeProcessor[MockSettings, MockMessageB, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "consumer": MockConsumer(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class TypeMismatchCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, None]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "transformer": MockTransformer(settings=settings),
            "consumer": MockConsumer(settings=settings),
        }


class InvalidProducerFirstCompositeProcessor(CompositeProcessor[MockSettings, NoneType, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "stateful_producer": MockStatefulProducer(settings=settings),
            "stateful_transformer": MockStatefulTransformer(settings=settings),
        }


class ChainedCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, None]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "stateful_consumer": MockStatefulConsumer(settings=settings),
        }


class ChainedCompositeProcessorWithDeepProcessors(CompositeProcessor[MockSettings, MockMessageA, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "deep_processor": DeepMockProcessor(settings=settings),
            "deeper_transformer": DeeperMockTransformer(settings=settings),
        }


class MockGeneratorTransformer(BaseTransformer[MockSettings, MockMessageA, MockMessageA]):
    """A mock transformer replacing the legacy @consumer generator."""

    def _process(self, message: MockMessageA) -> MockMessageA:
        return MockMessageA()


class MockGeneratorCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings):
        return {
            "generator": MockGeneratorTransformer(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


# Mock classes for _post_process testing


class AxisArrayScaleTransformer(BaseTransformer[MockSettings, AxisArray, AxisArray]):
    """Multiplies AxisArray.data by settings.param1."""

    def _process(self, message: AxisArray) -> AxisArray:
        return dataclasses.replace(message, data=message.data * self.settings.param1)


class AxisArrayOffsetTransformer(BaseTransformer[MockSettings, AxisArray, AxisArray]):
    """Adds settings.param1 to AxisArray.data."""

    def _process(self, message: AxisArray) -> AxisArray:
        return dataclasses.replace(message, data=message.data + self.settings.param1)


class PostProcessTrackingComposite(CompositeProcessor[MockSettings, MockMessageA, MockMessageB]):
    """CompositeProcessor that tracks _post_process calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_calls = 0

    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }

    def _post_process(self, result):
        self.post_process_calls += 1
        return result


# Mock CompositeProducer examples
class ValidSingleCompositeProducer(CompositeProducer[MockSettings, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"producer": MockProducer(settings=settings)}


class ValidMultipleCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


class EmptyCompositeProducer(CompositeProducer[MockSettings, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {}


class InvalidOutputCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"producer": MockProducer(settings=settings)}


class InvalidProducerNotFirstCompositeProducer(CompositeProducer[MockSettings, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class InvalidConsumerNotLastCompositeProducer(CompositeProducer[MockSettings, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "stateful_consumer": MockStatefulConsumer(settings=settings),
            "transformer": MockTransformer(settings=settings),
        }


class TypeMismatchCompositeProducer(CompositeProducer[MockSettings, None]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "consumer": MockConsumer(settings=settings),
        }


class ChainedCompositeProducerWithDeepProcessors(CompositeProducer[MockSettings, MockMessageC]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "stateful_producer": MockStatefulProducer(settings=settings),
            "deep_processor": DeepMockProcessor(settings=settings),
            "deeper_transformer": DeeperMockTransformer(settings=settings),
        }


class MockGeneratorProducer(BaseProducer[MockSettings, MockMessageA]):
    """A mock producer replacing the legacy @consumer producer generator."""

    async def _produce(self) -> MockMessageA:
        return MockMessageA()


class MockGeneratorPassthroughTransformer(BaseTransformer[MockSettings, MockMessageA, MockMessageA]):
    """A mock transformer replacing the legacy unprimed generator."""

    def _process(self, message: MockMessageA) -> MockMessageA:
        return message or MockMessageA()


class MockGeneratorCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings):
        return {
            "generator": MockGeneratorProducer(settings=settings),
            "mock_generator_passthrough": MockGeneratorPassthroughTransformer(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


# -- Tests --


def test_import():
    """Test that the package can be imported."""
    import ezmsg.baseproc

    assert hasattr(ezmsg.baseproc, "__version__")


def test_version():
    """Test that version is a string."""
    from ezmsg.baseproc import __version__

    assert isinstance(__version__, str)


# -- Tests for Helper Functions --


class TestHelperFunctions:
    def test_processor_state_decorator(self):
        # Test that processor_state creates a dataclass with the right properties
        state = MockState()
        assert state.iterations == 0
        assert state.hash == -1

        # Test that we can modify it (not frozen)
        state.iterations = 5
        assert state.iterations == 5

        # Test hash functionality
        state1 = MockState()
        state2 = MockState()
        assert hash(state1) == hash(state2)

        state1.iterations = 10
        assert hash(state1) != hash(state2)

    def test_get_base_processor_settings_type(self):
        # Test with regular processor classes
        assert _get_base_processor_settings_type(MockProcessor) == MockSettings
        assert _get_base_processor_settings_type(MockProducer) == MockSettings
        assert _get_base_processor_settings_type(MockConsumer) == MockSettings
        assert _get_base_processor_settings_type(MockTransformer) == MockSettings

        # Test with stateful processor classes
        assert _get_base_processor_settings_type(MockStatefulProcessor) == MockSettings
        assert _get_base_processor_settings_type(MockStatefulProducer) == MockSettings
        assert _get_base_processor_settings_type(MockStatefulConsumer) == MockSettings
        assert _get_base_processor_settings_type(MockStatefulTransformer) == MockSettings

        # Test with derived classes
        assert _get_base_processor_settings_type(DeepMockProcessor) == MockSettings
        assert _get_base_processor_settings_type(DeepMockTransformer) == MockSettings
        assert _get_base_processor_settings_type(DeeperMockTransformer) == MockSettings

        # Test with composite processor
        assert _get_base_processor_settings_type(ValidSingleCompositeProcessor) == MockSettings

        # Test with no settings type should raise exception
        class NoSettingsTypeClass:
            pass

        with pytest.raises(TypeError, match="Could not resolve settings type"):
            _get_base_processor_settings_type(NoSettingsTypeClass)

    def test_get_base_processor_message_in_type(self):
        # Test with processor classes that have input types
        assert _get_base_processor_message_in_type(MockProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(MockConsumer) == MockMessageB
        assert _get_base_processor_message_in_type(MockTransformer) == MockMessageA
        assert _get_base_processor_message_in_type(MockStatefulProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(MockStatefulConsumer) == MockMessageA
        assert _get_base_processor_message_in_type(MockStatefulTransformer) == MockMessageA
        assert _get_base_processor_message_in_type(MockAdaptiveTransformer) == MockMessageA
        assert _get_base_processor_message_in_type(MockAsyncTransformer) == MockMessageA

        # Test with derived classes
        assert _get_base_processor_message_in_type(DeepMockProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(DeepMockTransformer) == MockMessageA
        assert _get_base_processor_message_in_type(DeeperMockTransformer) == MockMessageA

        # Test with composite processor
        assert _get_base_processor_message_in_type(ValidSingleCompositeProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(ChainedCompositeProcessor) == MockMessageA

        # Test with producers (should throw)
        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(MockProducer)
        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(MockStatefulProducer)

        # Test with no message in type should raise exception
        class NoMessageInTypeClass:
            pass

        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(NoMessageInTypeClass)

    def test_get_base_processor_message_out_type(self):
        # Test with processor classes that have output types
        assert _get_base_processor_message_out_type(MockProcessor) == MockMessageB
        assert _get_base_processor_message_out_type(MockProducer) == MockMessageA
        assert _get_base_processor_message_out_type(MockTransformer) == MockMessageC
        assert _get_base_processor_message_out_type(MockStatefulProcessor) == MockMessageB
        assert _get_base_processor_message_out_type(MockStatefulProducer) == MockMessageA
        assert _get_base_processor_message_out_type(MockStatefulTransformer) == MockMessageC
        assert _get_base_processor_message_out_type(MockAdaptiveTransformer) == MockMessageB
        assert _get_base_processor_message_out_type(MockAsyncTransformer) == MockMessageB

        # Test with derived classes
        assert _get_base_processor_message_out_type(DeepMockProcessor) == MockMessageB
        assert _get_base_processor_message_out_type(DeepMockTransformer) == MockMessageC
        assert _get_base_processor_message_out_type(DeeperMockTransformer) == MockMessageC

        # Test with composite processor
        assert _get_base_processor_message_out_type(ValidSingleCompositeProcessor) == MockMessageB

        # Test with consumers (should be None)
        assert _get_base_processor_message_out_type(MockConsumer) is NoneType
        assert _get_base_processor_message_out_type(MockStatefulConsumer) is NoneType

        # Test with no message out type should raise exception
        class NoMessageOutTypeClass:
            pass

        with pytest.raises(TypeError, match="Could not resolve .*MessageOutType"):
            _get_base_processor_message_out_type(NoMessageOutTypeClass)

    # Test _unify_settings function through the MockProcessor class __init__
    def test_unify_settings_with_provided_settings(self):
        settings = MockSettings(param1=20)
        obj = MockProcessor(settings=settings)
        assert obj.settings.param1 == 20
        assert obj.settings.param2 == "test"

    def test_unify_settings_with_settings_arg(self):
        obj = MockProcessor(MockSettings(param1=30, param2="new"))
        assert obj.settings.param1 == 30
        assert obj.settings.param2 == "new"

    def test_unify_settings_with_args(self):
        obj = MockProcessor(40, "newer")
        assert obj.settings.param1 == 40
        assert obj.settings.param2 == "newer"

    def test_unify_settings_with_kwargs(self):
        obj = MockProcessor(param1=50)
        assert obj.settings.param1 == 50
        assert obj.settings.param2 == "test"

    def test_unify_settings_with_args_kwargs(self):
        obj = MockProcessor(60, param2="newest")
        assert obj.settings.param1 == 60
        assert obj.settings.param2 == "newest"

    def test_unify_settings_with_default(self):
        obj = MockProcessor()
        assert obj.settings.param1 == 10
        assert obj.settings.param2 == "test"

    def test_get_base_processor_state_type(self):
        # Test with class that has state type
        assert _get_base_processor_state_type(MockStatefulProcessor) == MockState
        assert _get_base_processor_state_type(DeepMockStatefulProcessor) == MockState
        assert _get_base_processor_state_type(MockStatefulProducer) == MockState
        assert _get_base_processor_state_type(MockStatefulConsumer) == MockState
        assert _get_base_processor_state_type(MockStatefulTransformer) == MockState
        assert _get_base_processor_state_type(MockAdaptiveTransformer) == MockState
        assert _get_base_processor_state_type(MockAsyncTransformer) == MockState
        assert _get_base_processor_state_type(ValidMultipleCompositeProcessor) == dict[str, Any]

        # Test with class that doesn't have state type
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockProcessor)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(DeepMockProcessor)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockProducer)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockConsumer)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockTransformer)

    def test_get_processor_message_type(self):
        processor = MockProcessor(MockSettings())
        producer = MockProducer(MockSettings())
        consumer = MockConsumer(MockSettings())
        deep_transformer = DeeperMockTransformer(MockSettings())

        # Test getting input type
        assert _get_processor_message_type(processor, "in") == MockMessageA

        # Test getting output type
        assert _get_processor_message_type(processor, "out") == MockMessageB

        # Test with producer
        assert _get_processor_message_type(producer, "in") is None
        assert _get_processor_message_type(producer, "out") == MockMessageA

        # Test with consumer
        assert _get_processor_message_type(consumer, "in") == MockMessageB
        assert _get_processor_message_type(consumer, "out") is None

        # Test with deep subclass (transformer)
        assert _get_processor_message_type(deep_transformer, "in") == MockMessageA
        assert _get_processor_message_type(deep_transformer, "out") == MockMessageC

        # Test with invalid direction
        with pytest.raises(ValueError, match="Invalid direction"):
            _get_processor_message_type(processor, "invalid")


# -- Tests for BaseProcessor and derived classes --


class TestBaseProcessor:
    def test_initialization(self):
        processor = MockProcessor(param1=20)
        assert processor.settings.param1 == 20
        assert processor.settings.param2 == "test"

    def test_get_settings_type(self):
        assert MockProcessor.get_settings_type() == MockSettings

    def test_get_message_type(self):
        assert MockProcessor.get_message_type("in") == MockMessageA
        assert MockProcessor.get_message_type("out") == MockMessageB

    def test_call_methods(self):
        processor = MockProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    @pytest.mark.asyncio
    async def test_acall_methods(self):
        processor = MockProcessor()
        result = await processor.__acall__(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_send_alias(self):
        processor = MockProcessor()
        result = processor.send(MockMessageA())
        assert isinstance(result, MockMessageB)

    @pytest.mark.asyncio
    async def test_asend_alias(self):
        processor = MockProcessor()
        result = await processor.asend(MockMessageA())
        assert isinstance(result, MockMessageB)


class TestBaseConsumer:
    def test_initialization(self):
        consumer = MockConsumer(param1=20)
        assert consumer.settings.param1 == 20
        assert consumer.settings.param2 == "test"

    def test_get_message_type(self):
        assert MockConsumer.get_message_type("in") == MockMessageB
        assert MockConsumer.get_message_type("out") is None

    def test_call_method(self):
        consumer = MockConsumer()
        # Should not raise an exception
        consumer(MockMessageB())

    @pytest.mark.asyncio
    async def test_acall_method(self):
        consumer = MockConsumer()
        # Should not raise an exception
        await consumer.__acall__(MockMessageB())


class TestBaseTransformer:
    def test_call_method(self):
        transformer = MockTransformer()
        result = transformer(MockMessageA())
        assert isinstance(result, MockMessageC)

    @pytest.mark.asyncio
    async def test_acall_method(self):
        transformer = MockTransformer()
        result = await transformer.__acall__(MockMessageA())
        assert isinstance(result, MockMessageC)


# -- Tests for BaseProducer --


class TestBaseProducer:
    def test_initialization(self):
        producer = MockProducer(param1=20)
        assert producer.settings.param1 == 20
        assert producer.settings.param2 == "test"

    def test_get_settings_type(self):
        assert MockProducer.get_settings_type() == MockSettings

    def test_get_message_type(self):
        assert MockProducer.get_message_type("in") is None
        assert MockProducer.get_message_type("out") == MockMessageA

    def test_call_method(self):
        producer = MockProducer()
        result = producer()
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_acall_method(self):
        producer = MockProducer()
        result = await producer.__acall__()
        assert isinstance(result, MockMessageA)

    def test_iterator_interface(self):
        producer = MockProducer()
        result = next(producer)
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_async_iterator_interface(self):
        producer = MockProducer()
        result = await producer.__anext__()
        assert isinstance(result, MockMessageA)


# -- Tests for BaseStatefulProcessor and derived classes --


class TestBaseStatefulProcessor:
    def test_initialization(self):
        processor = MockStatefulProcessor(param1=20)
        assert processor.settings.param1 == 20
        assert processor._hash == -1
        assert processor.state.iterations == 0

    def test_state_property(self):
        processor = MockStatefulProcessor()
        assert processor.state.iterations == 0

        # Modify state
        processor.state.iterations = 5
        assert processor.state.iterations == 5

    def test_state_setter_with_state_object(self):
        processor = MockStatefulProcessor()
        new_state = MockState()
        new_state.iterations = 10
        processor.state = new_state
        assert processor.state.iterations == 10

    def test_state_setter_with_bytes(self):
        processor = MockStatefulProcessor()
        new_state = MockState()
        new_state.iterations = 15
        serialized = pickle.dumps(new_state)
        processor.state = serialized
        assert processor.state.iterations == 15

    def test_hash_message_default(self):
        processor = MockStatefulProcessor()
        assert processor._hash_message(MockMessageA()) == 0

    def test_state_reset_on_first_call(self):
        processor = MockStatefulProcessor()
        assert processor._hash == -1

        # First call should trigger state reset
        processor(MockMessageA())
        assert processor._hash == 0
        assert processor.state.iterations == 1

    def test_stateful_op(self):
        processor = MockStatefulProcessor()
        state = (MockState(), -1)
        new_state, result = processor.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageB)
        assert new_state[0].iterations == 1
        assert new_state[1] == 0  # hash updated


class TestBaseStatefulConsumer:
    def test_stateful_op(self):
        consumer = MockStatefulConsumer()
        state = (MockState(), 0)
        new_state, result = consumer.stateful_op(state, MockMessageB())
        assert result is None
        assert new_state[0].iterations == 1


class TestBaseStatefulTransformer:
    def test_stateful_op(self):
        transformer = MockStatefulTransformer()
        state = (MockState(), 0)
        new_state, result = transformer.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageC)
        assert new_state[0].iterations == 1


# Helper to create an AxisArray with trigger in attrs for testing BaseAdaptiveTransformer
def mock_sample_message():
    return AxisArray(
        data=np.zeros((1, 1)),
        dims=["time", "ch"],
        attrs={"trigger": SampleTriggerMessage()},
    )


class TestBaseAdaptiveTransformer:
    def test_partial_fit(self):
        transformer = MockAdaptiveTransformer()
        transformer.partial_fit(mock_sample_message())
        assert transformer.state.fit_count == 1

    @pytest.mark.asyncio
    async def test_apartial_fit(self):
        transformer = MockAdaptiveTransformer()
        await transformer.apartial_fit(mock_sample_message())
        assert transformer.state.fit_count == 1

    def test_call_with_sample_message_warns(self):
        transformer = MockAdaptiveTransformer()
        sample_msg = mock_sample_message()
        with pytest.warns(UserWarning, match="Auto-routing to partial_fit"):
            result = transformer(sample_msg)
        assert isinstance(result, MockMessageB)  # inference, not partial_fit
        assert transformer.state.fit_count == 0  # partial_fit NOT called

    def test_partial_fit_transform(self):
        transformer = MockAdaptiveTransformer()
        sample_msg = mock_sample_message()
        result = transformer.partial_fit_transform(sample_msg)
        assert isinstance(result, MockMessageB)
        assert transformer.state.fit_count == 1
        assert transformer.state.iterations == 1  # _process was called


class TestBaseAsyncTransformer:
    @pytest.mark.asyncio
    async def test_acall_method(self):
        transformer = MockAsyncTransformer()
        result = await transformer.__acall__(MockMessageA())
        assert isinstance(result, MockMessageB)
        assert transformer.state.iterations == 1

    def test_call_method(self):
        transformer = MockAsyncTransformer()
        result = transformer(MockMessageA())
        assert isinstance(result, MockMessageB)
        assert transformer.state.iterations == 1


# -- Test for BaseStatefulProducer --


class TestBaseStatefulProducer:
    def test_initialization(self):
        producer = MockStatefulProducer(param1=20)
        assert producer.settings.param1 == 20
        assert producer._hash == -1
        assert producer.state.iterations == 0

    @pytest.mark.asyncio
    async def test_state_reset_on_first_call(self):
        producer = MockStatefulProducer()
        assert producer._hash == -1

        # First call should trigger state reset
        result = await producer.__acall__()
        assert producer._hash == 0
        assert producer.state.iterations == 1
        assert isinstance(result, MockMessageA)

    def test_stateful_op(self):
        producer = MockStatefulProducer()
        state = (MockState(), 0)
        new_state, result = producer.stateful_op(state)
        assert isinstance(result, MockMessageA)
        assert new_state[0].iterations == 1


# -- Tests for CompositeProcessor --


class TestCompositeProcessor:
    def test_valid_single_processor_chain(self):
        processor = ValidSingleCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_valid_multiple_processor_chain(self):
        processor = ValidMultipleCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_empty_processor_chain(self):
        with pytest.raises(ValueError, match="requires at least one processor"):
            EmptyCompositeProcessor()

    def test_invalid_output_type(self):
        with pytest.raises(TypeError, match="Output type mismatch"):
            InvalidOutputCompositeProcessor()

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="Input type mismatch"):
            InvalidInputCompositeProcessor()

    def test_invalid_producer_not_first(self):
        with pytest.raises(
            TypeError,
            match="Producers can only be the first processor of a composite producer chain.",
        ):
            InvalidProducerNotFirstCompositeProcessor()

    def test_invalid_consumer_not_last(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite processor chain.",
        ):
            InvalidConsumerNotLastCompositeProcessor()

    def test_invalid_consumer_to_producer(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite processor chain.",
        ):
            InvalidConsumerProducerCompositeProcessor()

    def test_type_mismatch_between_processors(self):
        with pytest.raises(TypeError, match="Message type mismatch between processors"):
            TypeMismatchCompositeProcessor()

    def test_composite_processor_with_producer_first(self):
        with pytest.raises(
            TypeError,
            match=r"First processor .* is a producer or receives only None. Please use CompositeProducer.*",
        ):
            InvalidProducerFirstCompositeProcessor()

    def test_chained_processors(self):
        processor = ChainedCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, NoneType)

    def test_chained_processors_with_deep_classes(self):
        processor = ChainedCompositeProcessorWithDeepProcessors()
        result = processor(None)
        assert isinstance(result, MockMessageC)

    def test_composite_processor_with_generator_sync(self):
        processor = MockGeneratorCompositeProcessor()
        # Attempt to use the sync processing interface and assert it raises an error
        result = processor._process(MockMessageA())
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_composite_processor_with_generator_async(self):
        processor = MockGeneratorCompositeProcessor()
        result = await processor._aprocess(MockMessageA())
        assert isinstance(result, MockMessageA)

    def test_state_property(self):
        processor1 = ValidMultipleCompositeProcessor()
        processor2 = ChainedCompositeProcessorWithDeepProcessors()
        processor3 = MockGeneratorCompositeProcessor()

        state1 = processor1.state
        assert "processor" not in state1
        assert "stateful_processor" in state1

        state2 = processor2.state
        assert "processor" not in state2
        assert "deep_processor" not in state2
        assert "deeper_transformer" not in state2

        state3 = processor3.state
        assert "generator" not in state3
        assert "stateful_processor" in state3

    def test_state_setter(self):
        processor = ChainedCompositeProcessor()

        # Create new states
        state1 = MockState()
        state1.iterations = 5
        state2 = MockState()
        state2.iterations = 10

        # Set composite state
        processor.state = {"processor": state1, "stateful_consumer": state2}

        assert not hasattr(processor._procs["processor"], "state")
        assert processor._procs["stateful_consumer"].state.iterations == 10

        # Attempt to set state for non-existent processor
        with pytest.raises(
            KeyError,
            match=r"Processor .* in provided state not found in composite processor chain.",
        ):
            processor.state = {"non_existent_processor": state1}

    def test_state_setter_with_bytes(self):
        processor = ChainedCompositeProcessor()

        # Create new states
        state1 = MockState()
        state1.iterations = 15
        state2 = MockState()
        state2.iterations = 20

        # Serialize composite state
        serialized = pickle.dumps({"processor": state1, "stateful_consumer": state2})
        processor.state = serialized
        assert not hasattr(processor._procs["processor"], "state")
        assert processor._procs["stateful_consumer"].state.iterations == 20

    @pytest.mark.asyncio
    async def test_aprocess_method(self):
        processor = ValidMultipleCompositeProcessor()
        result = await processor._aprocess(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_stateful_op(self):
        processor = ValidMultipleCompositeProcessor()
        state = {"processor": (MockState(), 2), "stateful_processor": (MockState(), 3)}
        new_state, result = processor.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageB)
        # Do we really want to keep the 'state' of a stateless processor?
        # State of processor 1 not changed from default values
        assert new_state["processor"][0].iterations == 0
        assert new_state["processor"][0].hash == -1
        assert hasattr(processor._procs["processor"], "_state") is False
        # State of processor 2 updated
        assert new_state["stateful_processor"][0].iterations == 1
        assert new_state["stateful_processor"][0].hash == -1
        assert processor._procs["stateful_processor"].state.iterations == 1
        # Hash not set to 3 as expected as processor is called after setting the hash
        # Hash set to 0 via the _reset_state method.
        assert processor._procs["stateful_processor"]._hash == 0


# -- Tests for _post_process hook --


class TestPostProcess:
    def test_post_process_called_on_process(self):
        proc = PostProcessTrackingComposite()
        assert proc.post_process_calls == 0
        proc._process(MockMessageA())
        assert proc.post_process_calls == 1
        proc._process(MockMessageA())
        assert proc.post_process_calls == 2

    @pytest.mark.asyncio
    async def test_post_process_called_on_aprocess(self):
        proc = PostProcessTrackingComposite()
        await proc._aprocess(MockMessageA())
        assert proc.post_process_calls == 1

    def test_post_process_called_on_stateful_op(self):
        proc = PostProcessTrackingComposite()
        proc.stateful_op(None, MockMessageA())
        assert proc.post_process_calls == 1

    def test_post_process_default_is_identity(self):
        """The default _post_process is a no-op."""
        proc = ValidSingleCompositeProcessor()
        result = proc(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_post_process_with_mlx(self):
        """Test _post_process with MLX eval on Apple Silicon."""
        mx = pytest.importorskip("mlx.core")

        class MLXEvalComposite(CompositeProcessor[MockSettings, AxisArray, AxisArray]):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.used_mlx_eval = False

            @staticmethod
            def _initialize_processors(settings):
                return {
                    "scale": AxisArrayScaleTransformer(settings=settings),
                    "offset": AxisArrayOffsetTransformer(settings=settings),
                }

            def _post_process(self, result):
                if result is not None:
                    import mlx.core as mx

                    if isinstance(result.data, mx.array):
                        mx.eval(result.data)
                        self.used_mlx_eval = True
                return result

        input_data = mx.array([1.0, 2.0, 3.0])
        msg = AxisArray(data=input_data, dims=["time"])

        # param1=3: scale(×3) then offset(+3) → [1,2,3] → [3,6,9] → [6,9,12]
        proc = MLXEvalComposite(settings=MockSettings(param1=3))
        result = proc(msg)

        # Verify MLX path was taken
        assert proc.used_mlx_eval, "Expected _post_process to detect and eval MLX array"
        # Verify result is still an MLX array (not converted to numpy)
        assert isinstance(result.data, mx.array)
        # Verify both processors ran: (data * 3) + 3
        np.testing.assert_array_equal(np.array(result.data), [6.0, 9.0, 12.0])


# -- Tests for CompositeProducer --


class TestCompositeProducer:
    def test_valid_single_producer_chain(self):
        producer = ValidSingleCompositeProducer()
        result = producer()
        assert isinstance(result, MockMessageA)

    def test_valid_multiple_producer_chain(self):
        producer = ValidMultipleCompositeProducer()
        result = producer()
        assert isinstance(result, MockMessageB)

    def test_empty_producer_chain(self):
        with pytest.raises(ValueError, match="requires at least one processor"):
            EmptyCompositeProducer()

    def test_invalid_output_type(self):
        with pytest.raises(TypeError, match="Output type mismatch"):
            InvalidOutputCompositeProducer()

    def test_invalid_producer_not_first(self):
        with pytest.raises(TypeError, match="Input type mismatch: Composite producer expects None"):
            InvalidProducerNotFirstCompositeProducer()

    def test_invalid_consumer_not_last(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite producer chain.",
        ):
            InvalidConsumerNotLastCompositeProducer()

    def test_type_mismatch_between_processors(self):
        with pytest.raises(TypeError, match="Message type mismatch between processors"):
            TypeMismatchCompositeProducer()

    def test_chained_producers_with_deep_classes(self):
        producer = ChainedCompositeProducerWithDeepProcessors()
        result = producer()
        assert isinstance(result, MockMessageC)

    @pytest.mark.asyncio
    async def test_composite_producer_with_generator(self):
        producer = MockGeneratorCompositeProducer()
        result = await producer._produce()
        assert isinstance(result, MockMessageA)

    def test_state_property(self):
        producer1 = ValidMultipleCompositeProducer()
        producer2 = ChainedCompositeProducerWithDeepProcessors()
        producer3 = MockGeneratorCompositeProducer()

        state1 = producer1.state
        assert "producer" not in state1
        assert "stateful_processor" in state1

        state2 = producer2.state
        assert "stateful_producer" in state2
        assert "deep_processor" not in state2
        assert "deeper_transformer" not in state2

        state3 = producer3.state
        assert "generator" not in state3
        assert "stateful_processor" in state3

    def test_state_setter(self):
        producer = ValidMultipleCompositeProducer()

        # Create new states
        state1 = MockState()
        state1.iterations = 5
        state2 = MockState()
        state2.iterations = 10

        # Set composite state
        producer.state = {"producer": state1, "stateful_processor": state2}

        assert not hasattr(producer._procs["producer"], "state")
        assert producer._procs["stateful_processor"].state.iterations == 10

        # Attempt to set state for non-existent processor
        with pytest.raises(
            KeyError,
            match=r"Processor .* in provided state not found in composite producer chain.",
        ):
            producer.state = {"non_existent_processor": state1}

    def test_state_setter_with_bytes(self):
        producer = ValidMultipleCompositeProducer()

        # Create new states
        state1 = MockState()
        state1.iterations = 15
        state2 = MockState()
        state2.iterations = 20

        # Serialize composite state
        serialized = pickle.dumps({"producer": state1, "stateful_processor": state2})
        producer.state = serialized
        assert not hasattr(producer._procs["producer"], "state")
        assert producer._procs["stateful_processor"].state.iterations == 20

    @pytest.mark.asyncio
    async def test_produce_method(self):
        producer = ValidMultipleCompositeProducer()
        result = await producer._produce()
        assert isinstance(result, MockMessageB)

    def test_stateful_op(self):
        composite_producer = ValidMultipleCompositeProducer()
        state = {"producer": (MockState(), 2), "stateful_processor": (MockState(), 3)}
        new_state, result = composite_producer.stateful_op(state)
        assert isinstance(result, MockMessageB)
        # Do we really want to keep the 'state' of a stateless processor?
        # State of producer not changed from default values
        assert new_state["producer"][0].iterations == 0
        assert new_state["producer"][0].hash == -1
        assert hasattr(composite_producer._procs["producer"], "_state") is False
        # State of stateful_processor updated
        assert new_state["stateful_processor"][0].iterations == 1
        assert new_state["stateful_processor"][0].hash == -1
        assert composite_producer._procs["stateful_processor"].state.iterations == 1
        # Hash not set to 3 as expected as processor is called after setting the hash
        # Hash set to 0 via the _reset_state method.
        assert composite_producer._procs["stateful_processor"]._hash == 0


class TestMessageHashDefault:
    """The default `_hash_message` keys on stream shape and channel identity.

    A processor that caches per-channel state -- a filter's `zi`, a scaler's
    running mean -- is only valid for the channels it was built for. Keying on
    shape alone cannot see a source that swaps channels without changing how
    many it sends, and the stale state then contaminates the new channels.
    """

    class Probe(Stateful[MockState]):
        def _reset_state(self, message):  # pragma: no cover - not exercised
            pass

        def _process(self, message):  # pragma: no cover - not exercised
            return message

        def stateful_op(self, state, message):  # pragma: no cover - not exercised
            raise NotImplementedError

    @staticmethod
    def _msg(n_time, labels, fs=100.0, key="dev", coord_time=False):
        import numpy as np

        time_axis = (
            AxisArray.CoordinateAxis(data=np.arange(n_time, dtype=float), dims=["time"])
            if coord_time
            else AxisArray.TimeAxis(fs=fs)
        )
        return AxisArray(
            np.zeros((n_time, len(labels))),
            dims=["time", "ch"],
            axes={
                "time": time_axis,
                "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
            },
            key=key,
            chunk_dim="time",
        )

    @staticmethod
    def _win_msg(n_win, n_time=10, labels=("c0", "c1")):
        import numpy as np

        return AxisArray(
            np.zeros((n_win, n_time, len(labels))),
            dims=["win", "time", "ch"],
            axes={
                "win": AxisArray.TimeAxis(fs=10.0),
                "time": AxisArray.TimeAxis(fs=100.0),
                "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
            },
            key="dev",
            chunk_dim="win",
        )

    def _resets(self, proc, messages):
        """Indices of the messages that would trigger a state reset."""
        out, prev = [], object()
        for idx, msg in enumerate(messages):
            msg_hash = proc._hash_message(msg)
            if msg_hash != prev:
                out.append(idx)
            prev = msg_hash
        return out

    def test_chunk_size_jitter_does_not_reset(self):
        """The streaming dim's length changes every message by definition."""
        proc = self.Probe()
        msgs = [self._msg(n, ["c0", "c1"]) for n in (30, 17, 30, 41)]
        assert self._resets(proc, msgs) == [0]

    def test_relabel_at_fixed_channel_count_resets(self):
        """The whole point: shape is identical, channels are not."""
        proc = self.Probe()
        msgs = [self._msg(30, ["c0", "c1"]), self._msg(30, ["x0", "x1"])]
        assert self._resets(proc, msgs) == [0, 1]

    def test_channel_count_sample_rate_and_key_reset(self):
        proc = self.Probe()
        msgs = [
            self._msg(30, ["c0", "c1"]),
            self._msg(30, ["c0", "c1", "c2"]),  # count
            self._msg(30, ["c0", "c1", "c2"], fs=200.0),  # sample rate
            self._msg(30, ["c0", "c1", "c2"], fs=200.0, key="other"),  # key
        ]
        assert self._resets(proc, msgs) == [0, 1, 2, 3]

    def test_time_offset_alone_does_not_reset(self):
        """`offset` advances every message and must never be folded in."""
        proc = self.Probe()
        msgs = []
        for i in range(3):
            m = self._msg(30, ["c0", "c1"])
            m.axes["time"] = AxisArray.TimeAxis(fs=100.0, offset=i * 0.3)
            msgs.append(m)
        assert self._resets(proc, msgs) == [0]

    def test_coordinate_streaming_axis_does_not_reset(self):
        """Irregular event streams carry per-message values on `time`; folding
        those in would reset on every message."""
        proc = self.Probe()
        msgs = [self._msg(5, ["c0", "c1"], coord_time=True) for _ in range(3)]
        assert self._resets(proc, msgs) == [0]

    def test_chunk_dim_names_the_dimension_that_grows(self):
        """Downstream of a windowing stage, `win` grows and `time` is fixed.

        A consumer cannot infer this -- `time` is the chunk dimension on a raw
        signal and a fixed within-window axis here -- so the producer declares
        it and the consumer needs to know nothing.
        """
        msgs = [
            self._win_msg(3),
            self._win_msg(1),  # window count jitters
            self._win_msg(4),
            self._win_msg(2, labels=("x0", "x1")),  # relabel
            self._win_msg(2, labels=("x0", "x1")),
            self._win_msg(2, n_time=20, labels=("x0", "x1")),  # window length
        ]
        assert self._resets(self.Probe(), msgs) == [0, 3, 5]

    def test_undeclared_falls_back_to_streaming_dims(self):
        """Without a declaration the class default is used, which is right for
        the common (time, ch) stream: chunk-size jitter must not reset."""
        import numpy as np

        def undeclared(n_time, labels=("c0", "c1")):
            return AxisArray(
                np.zeros((n_time, len(labels))),
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=100.0),
                    "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
                },
                key="dev",
            )

        proc = self.Probe()
        assert self._resets(proc, [undeclared(n) for n in (30, 17, 41)]) == [0]
        # ...and still catches a relabel, which is the point of the default.
        assert proc._hash_message(undeclared(30)) != proc._hash_message(undeclared(30, labels=("x0", "x1")))

    def test_fallback_naming_an_absent_dim_is_harmless(self):
        """A message with no `time` dim at all: nothing matches, so nothing is
        excluded, and every dimension is stable message to message."""
        import numpy as np

        def spectrum():
            return AxisArray(
                np.zeros((21, 2)),
                dims=["freq", "ch"],
                axes={
                    "freq": AxisArray.LinearAxis(gain=1.0, offset=5.0, unit="Hz"),
                    "ch": AxisArray.CoordinateAxis(data=np.array(["c0", "c1"]), dims=["ch"]),
                },
                key="dev",
            )

        assert self._resets(self.Probe(), [spectrum() for _ in range(3)]) == [0]

    def test_non_axisarray_message_resets_once(self):
        """Processors on other message types keep their previous behaviour."""
        proc = self.Probe()
        assert proc._hash_message(MockMessageA()) == 0
        assert self._resets(proc, [MockMessageA(), MockMessageA(), MockMessageA()]) == [0]

    def test_include_key_false(self):
        proc = self.Probe()
        assert proc._message_hash(self._msg(30, ["c0"]), include_key=False) == proc._message_hash(
            self._msg(30, ["c0"], key="other"), include_key=False
        )

    def test_exclude_dims_suppresses_channel_identity(self):
        proc = self.Probe()
        assert proc._message_hash(self._msg(30, ["c0", "c1"]), exclude_dims=("time", "ch")) == proc._message_hash(
            self._msg(30, ["x0", "x1"]), exclude_dims=("time", "ch")
        )

    def test_extra_is_folded_in(self):
        proc = self.Probe()
        msg = self._msg(30, ["c0", "c1"])
        assert proc._message_hash(msg, extra=("float32",)) != proc._message_hash(msg, extra=("float64",))

    def test_non_streaming_linear_axis_offset_is_folded_in(self):
        """A frequency band is located by its offset, not just its step size.

        A spectrum covering 5-25 Hz and one covering 70-90 Hz have the same
        gain and the same length; only the offset tells them apart. `offset` is
        dropped for the streaming dimension, where it merely counts elapsed
        samples, and nowhere else.
        """
        import numpy as np

        def spectrum(freq_offset):
            return AxisArray(
                np.zeros((4, 21, 2)),
                dims=["time", "freq", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=100.0),
                    "freq": AxisArray.LinearAxis(gain=1.0, offset=freq_offset, unit="Hz"),
                    "ch": AxisArray.CoordinateAxis(data=np.array(["c0", "c1"]), dims=["ch"]),
                },
                key="dev",
                chunk_dim="time",
            )

        proc = self.Probe()
        assert proc._hash_message(spectrum(5.0)) != proc._hash_message(spectrum(70.0))
        assert proc._hash_message(spectrum(5.0)) == proc._hash_message(spectrum(5.0))

    def test_exclude_dims_is_additive_to_the_chunk_dim(self):
        """Naming a dimension to ignore must not un-exclude the chunk dim."""
        import numpy as np

        def msg(n_time, labels):
            return AxisArray(
                np.zeros((n_time, len(labels))),
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=100.0),
                    "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
                },
                key="dev",
                chunk_dim="time",
            )

        proc = self.Probe()
        assert proc._message_hash(msg(30, ["c0", "c1"]), exclude_dims=("ch",)) == proc._message_hash(
            msg(17, ["x0", "x1"]), exclude_dims=("ch",)
        )
