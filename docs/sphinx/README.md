# PyQuifer API Documentation

This directory contains Sphinx configuration for generating API documentation from docstrings.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
cd PyQuifer
pip install -e ".[docs]"
```

Or install directly:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Building HTML

**Windows:**
```batch
cd docs\sphinx
make.bat html
```

**Linux/macOS:**
```bash
cd docs/sphinx
make html
```

The built documentation will be in `docs/sphinx/_build/html/`.

### Viewing

Open `_build/html/index.html` in your browser.

### Live Preview

For development with auto-rebuild:

```bash
pip install sphinx-autobuild
cd docs/sphinx
make livehtml
```

Then open http://127.0.0.1:8000 in your browser.

## Structure

```
sphinx/
├── conf.py              # Sphinx configuration
├── index.rst            # Main entry point
├── modules/
│   ├── index.rst        # API reference index
│   ├── foundation.rst   # Core, oscillators, LinOSS, multiplexing
│   ├── consciousness.rst # Consciousness, criticality, IIT
│   ├── learning.rst     # Learning, motivation, spiking
│   ├── dynamics.rst     # Hyperbolic, attractors, liquid networks
│   ├── embodiment.rst   # Somatic, metacognitive, morphological
│   └── social.rst       # Social, developmental, ecology
├── requirements.txt     # Doc dependencies
├── Makefile            # Unix build script
└── make.bat            # Windows build script
```

## Adding Documentation

The documentation is auto-generated from docstrings using `autodoc`. To improve documentation:

1. Add/improve docstrings in the Python source files
2. Rebuild with `make html`
3. Check the output in `_build/html/`

Docstrings should follow Google or NumPy style for best results.
