# ezmsg-example

Short description of your ezmsg package.

## Installation

```bash
pip install ezmsg-example
```

## Dependencies

- `ezmsg`
- `numpy`

## Usage

See the `examples` folder for usage examples.

```python
import ezmsg.core as ez
from ezmsg.example import MyUnit

# Your usage example here
```

## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork this repository and clone your fork locally.
3. Open a terminal and `cd` to the cloned folder.
4. Run `uv sync` to create a `.venv` and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After making changes, run the test suite: `uv run pytest tests`

## License

MIT License - see [LICENSE](LICENSE) for details.
