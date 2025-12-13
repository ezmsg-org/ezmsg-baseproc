# ezmsg Package Template

This is a template repository for creating new ezmsg namespace packages.

## Using This Template

### 1. Create Your Repository

Click "Use this template" on GitHub to create a new repository from this template.

### 2. Find and Replace Placeholders

After creating your repository, replace all occurrences of `example` with your package name:

| Find            | Replace With          | Example         |
|-----------------|-----------------------|-----------------|
| `ezmsg-example` | `ezmsg-{yourpackage}` | `ezmsg-sigproc` |
| `ezmsg.example` | `ezmsg.{yourpackage}` | `ezmsg.sigproc` |
| `ezmsg/example` | `ezmsg/{yourpackage}` | `ezmsg/sigproc` |

**Files to update:**
- [ ] `pyproject.toml` - Package name, description, authors, dependencies
- [ ] `README.md` - Package name, description, usage examples
- [ ] `.gitignore` - Update `__version__.py` path
- [ ] `docs/source/conf.py` - Project name, URLs
- [ ] `docs/source/index.rst` - Documentation content

**Directories to rename:**
- [ ] `src/ezmsg/example/` → `src/ezmsg/{yourpackage}/`

### 3. Update Package Metadata

Edit `pyproject.toml`:

```toml
[project]
name = "ezmsg-yourpackage"
description = "Your package description"
authors = [
  { name = "Your Name", email = "your.email@example.com" },
]
dependencies = [
  "ezmsg>=3.6.0",
  # Add your dependencies here
]
```

### 4. Set Up PyPI Publishing

To enable automatic publishing to PyPI on releases:

1. Create an account on [PyPI](https://pypi.org)
2. Create an API token for your package
3. Add the token as a repository secret named `PYPI_API_TOKEN` in GitHub Settings → Secrets and variables → Actions

### 5. Set Up GitHub Pages (for docs)

1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"
3. The docs workflow will automatically deploy on pushes to `main`

### 6. Clean Up

Delete this file (`TEMPLATE_README.md`) after you've completed the setup.

## Template Structure

```
ezmsg-template/
├── .github/workflows/     # CI/CD workflows
│   ├── python-tests.yml   # Test on multiple Python versions/OS
│   ├── python-publish.yml # Publish to PyPI on release
│   └── docs.yml           # Build and deploy documentation
├── src/ezmsg/example/     # Your package source code
│   └── __init__.py
├── tests/                 # Test files
│   └── test_example.py
├── docs/                  # Sphinx documentation
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   ├── Makefile
│   └── make.bat
├── examples/              # Usage examples
│   └── example.py
├── .pre-commit-config.yaml # Ruff linting/formatting
├── .gitignore
├── LICENSE
├── pyproject.toml         # Package configuration
└── README.md
```

## Development Workflow

### Install dependencies
```bash
uv sync
```

### Run tests
```bash
uv run pytest tests
```

### Run linting
```bash
uv run ruff check src
uv run ruff format src
```

### Build documentation locally
```bash
cd docs
uv run make html
open build/html/index.html
```

### Install pre-commit hooks
```bash
uv run pre-commit install
```

## Namespace Package Notes

This template uses PEP 420 implicit namespace packages:

- The `src/ezmsg/` directory does **not** contain an `__init__.py`
- Only `src/ezmsg/{yourpackage}/` contains `__init__.py`
- This allows multiple packages to coexist in the `ezmsg` namespace

## Questions?

- [ezmsg documentation](https://www.ezmsg.org)
- [ezmsg-org GitHub](https://github.com/ezmsg-org)
