# ezmsg-baseproc

Base processor classes and protocols for building signal processing pipelines in [ezmsg](https://github.com/ezmsg-org/ezmsg).

## Installation

```bash
pip install ezmsg-baseproc
```

## Overview

This package provides the foundational processor architecture for ezmsg signal processing:

- **Protocols** - Type definitions for processors, transformers, consumers, and producers
- **Base Classes** - Abstract base classes for building stateless and stateful processors
- **Composite Processors** - Classes for chaining processors into pipelines
- **Unit Wrappers** - ezmsg Unit base classes that wrap processors for graph integration

## Module Structure

```
ezmsg.baseproc/
├── protocols.py      # Protocol definitions and type variables
├── processor.py      # Base non-stateful processors
├── stateful.py       # Stateful processor base classes
├── composite.py      # CompositeProcessor and CompositeProducer
├── units.py          # ezmsg Unit wrappers
└── util/
    ├── asio.py           # Async/sync utilities
    ├── message.py        # SampleMessage definitions
    ├── profile.py        # Profiling decorators
    └── typeresolution.py # Type resolution helpers
```

## Usage

### Creating a Simple Transformer

```python
from dataclasses import dataclass
from ezmsg.baseproc import BaseTransformer
from ezmsg.util.messages.axisarray import AxisArray

@dataclass
class MySettings:
    scale: float = 1.0

class MyTransformer(BaseTransformer[MySettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return message.replace(data=message.data * self.settings.scale)
```

### Creating a Stateful Transformer

```python
from ezmsg.baseproc import BaseStatefulTransformer, processor_state

@processor_state
class MyState:
    count: int = 0
    hash: int = -1

class MyStatefulTransformer(BaseStatefulTransformer[MySettings, AxisArray, AxisArray, MyState]):
    def _reset_state(self, message: AxisArray) -> None:
        self._state.count = 0

    def _process(self, message: AxisArray) -> AxisArray:
        self._state.count += 1
        return message
```

### Creating an ezmsg Unit

```python
from ezmsg.baseproc import BaseTransformerUnit

class MyUnit(BaseTransformerUnit[MySettings, AxisArray, AxisArray, MyTransformer]):
    SETTINGS = MySettings
    # That's all - the base class handles everything else!
```

## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install `uv` if not already installed
2. Clone and cd into the repository
3. Run `uv sync` to create a `.venv` and install dependencies
4. Run `uv run pytest tests` to run tests

## License

MIT License - see [LICENSE](LICENSE) for details.
