"""
ezmsg-baseproc: Base processor classes for ezmsg.

This package provides the foundational processor architecture for building
signal processing pipelines in ezmsg.
"""

from .__version__ import __version__ as __version__

# Clock and Counter
from .clock import (
    Clock,
    ClockProducer,
    ClockSettings,
    ClockState,
)

# Composite processor classes
from .composite import (
    CompositeProcessor,
    CompositeProducer,
    CompositeStateful,
    _get_processor_message_type,
)
from .counter import (
    Counter,
    CounterSettings,
    CounterTransformer,
    CounterTransformerState,
)

# Base processor classes (non-stateful)
from .processor import (
    BaseConsumer,
    BaseProcessor,
    BaseProducer,
    BaseTransformer,
    _get_base_processor_message_in_type,
    _get_base_processor_message_out_type,
    _get_base_processor_settings_type,
)

# Protocols and type variables
from .protocols import (
    AdaptiveTransformer,
    Consumer,
    MessageInType,
    MessageOutType,
    Processor,
    Producer,
    SettingsType,
    StatefulConsumer,
    StatefulProcessor,
    StatefulProducer,
    StatefulTransformer,
    StateType,
    Transformer,
    processor_state,
)

# Stateful processor classes
from .stateful import (
    BaseAdaptiveTransformer,
    BaseAsyncTransformer,
    BaseStatefulConsumer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    BaseStatefulTransformer,
    Stateful,
    _get_base_processor_state_type,
)

# Unit classes for ezmsg integration
from .units import (
    AdaptiveTransformerType,
    BaseAdaptiveTransformerUnit,
    BaseConsumerUnit,
    BaseProcessorUnit,
    BaseProducerUnit,
    BaseTransformerUnit,
    ConsumerType,
    GenAxisArray,
    ProducerType,
    TransformerType,
    get_base_adaptive_transformer_type,
    get_base_consumer_type,
    get_base_producer_type,
    get_base_transformer_type,
)
from .util.asio import CoroutineExecutionError, SyncToAsyncGeneratorWrapper, run_coroutine_sync

# Utility classes and functions
from .util.message import SampleMessage, SampleTriggerMessage, is_sample_message
from .util.profile import profile_method, profile_subpub
from .util.typeresolution import check_message_type_compatibility, resolve_typevar

__all__ = [
    # Version
    "__version__",
    # Protocols
    "Processor",
    "Producer",
    "Consumer",
    "Transformer",
    "StatefulProcessor",
    "StatefulProducer",
    "StatefulConsumer",
    "StatefulTransformer",
    "AdaptiveTransformer",
    # Type variables
    "MessageInType",
    "MessageOutType",
    "SettingsType",
    "StateType",
    "ProducerType",
    "ConsumerType",
    "TransformerType",
    "AdaptiveTransformerType",
    # Decorators
    "processor_state",
    # Base processor classes
    "BaseProcessor",
    "BaseProducer",
    "BaseConsumer",
    "BaseTransformer",
    # Stateful classes
    "Stateful",
    "BaseStatefulProcessor",
    "BaseStatefulProducer",
    "BaseStatefulConsumer",
    "BaseStatefulTransformer",
    "BaseAdaptiveTransformer",
    "BaseAsyncTransformer",
    # Composite classes
    "CompositeStateful",
    "CompositeProcessor",
    "CompositeProducer",
    # Unit classes
    "BaseProducerUnit",
    "BaseProcessorUnit",
    "BaseConsumerUnit",
    "BaseTransformerUnit",
    "BaseAdaptiveTransformerUnit",
    "GenAxisArray",
    # Type resolution helpers
    "get_base_producer_type",
    "get_base_consumer_type",
    "get_base_transformer_type",
    "get_base_adaptive_transformer_type",
    "_get_base_processor_settings_type",
    "_get_base_processor_message_in_type",
    "_get_base_processor_message_out_type",
    "_get_base_processor_state_type",
    "_get_processor_message_type",
    # Message types
    "SampleMessage",
    "SampleTriggerMessage",
    "is_sample_message",
    # Profiling
    "profile_method",
    "profile_subpub",
    # Async utilities
    "CoroutineExecutionError",
    "SyncToAsyncGeneratorWrapper",
    "run_coroutine_sync",
    # Type utilities
    "check_message_type_compatibility",
    "resolve_typevar",
    # Clock and Counter
    "Clock",
    "ClockProducer",
    "ClockSettings",
    "ClockState",
    "Counter",
    "CounterSettings",
    "CounterTransformer",
    "CounterTransformerState",
]
