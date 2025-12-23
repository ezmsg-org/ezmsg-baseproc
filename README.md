# ezmsg-baseproc

Base processor classes and protocols for building message-processing components in [ezmsg](https://github.com/ezmsg-org/ezmsg).

## Installation

```bash
pip install ezmsg-baseproc
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-baseproc@dev
```

## Overview

``ezmsg-baseproc`` provides abstract base classes for creating message processors that can be used both standalone and within ezmsg pipelines. The package offers a consistent pattern for building:

* **Protocols** - Type definitions for processors, transformers, consumers, and producers
* **Processors** - Transform input messages to output messages
* **Producers** - Generate output messages without requiring input
* **Consumers** - Accept input messages without producing output
* **Transformers** - A specific type of processor with typed input/output
* **Stateful variants** - Processors that maintain state across invocations
* **Adaptive transformers** - Transformers that can be trained via ``partial_fit``
* **Composite processors** - Chain multiple processors together efficiently

All base classes support both synchronous and asynchronous operation, making them suitable for offline analysis and real-time streaming applications.

## Usage

### Creating a Simple Transformer

```python
from dataclasses import dataclass
from ezmsg.baseproc import BaseTransformer
from ezmsg.util.messages.axisarray import AxisArray, replace

@dataclass
class MySettings:
    scale: float = 1.0

class MyTransformer(BaseTransformer[MySettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data * self.settings.scale)
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
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`

## License

MIT License - see [LICENSE](LICENSE) for details.
