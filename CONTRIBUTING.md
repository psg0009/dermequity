# Contributing to DermEquity

Thank you for your interest in contributing to DermEquity! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)

---

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment. Please:

- Be respectful and inclusive
- Focus on constructive feedback
- Support fellow contributors
- Report unacceptable behavior

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/dermequity.git
cd dermequity
git remote add upstream https://github.com/psg0009/dermequity.git
```

---

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, PyTorch version)

### Suggesting Features

1. Open an issue with `[Feature Request]` prefix
2. Describe the feature and use case
3. Explain why it benefits the project

### Contributing Code

1. Find an issue to work on (or create one)
2. Comment that you're working on it
3. Fork, code, test, submit PR

---

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest pytest-cov flake8 black

# Run tests
pytest tests/ -v

# Run linting
flake8 dermequity/
black dermequity/ --check
```

---

## Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test locally**
   ```bash
   pytest tests/ -v
   flake8 dermequity/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "Add: brief description of change"
   ```
   
   Prefixes:
   - `Add:` New feature
   - `Fix:` Bug fix
   - `Update:` Improvement to existing feature
   - `Docs:` Documentation only
   - `Test:` Test only changes

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create PR on GitHub with:
   - Clear description
   - Link to related issue
   - Screenshots if applicable

6. **Address review feedback**

---

## Coding Standards

### Style

- Follow PEP 8
- Use Black for formatting
- Maximum line length: 100 characters
- Use type hints

### Documentation

- Docstrings for all public functions/classes (NumPy style)
- Update README for new features
- Update API.md for new public APIs

### Example Docstring

```python
def compute_equity_gap(metrics_by_group: Dict[int, float]) -> EquityGap:
    """
    Compute equity gap from metrics by group.
    
    Parameters
    ----------
    metrics_by_group : Dict[int, float]
        Dictionary mapping group ID to metric value
        
    Returns
    -------
    EquityGap
        Computed equity gap with best/worst group info
        
    Examples
    --------
    >>> metrics = {1: 0.9, 2: 0.8, 3: 0.6}
    >>> gap = compute_equity_gap(metrics)
    >>> gap.gap
    0.3
    """
```

### Testing

- Write tests for new functionality
- Maintain >80% coverage
- Use pytest fixtures
- Test edge cases

---

## Areas for Contribution

### High Priority

- [ ] Multi-class classification support
- [ ] Additional datasets (HAM10000, ISIC)
- [ ] More visualization options
- [ ] Performance optimizations

### Documentation

- [ ] Tutorial notebooks
- [ ] Video walkthrough
- [ ] API examples

### Research

- [ ] New mitigation strategies
- [ ] Intersection of skin tone + other demographics
- [ ] Calibration improvements

---

## Questions?

- Open an issue with `[Question]` prefix
- Email: pgosar@usc.edu

Thank you for contributing to fairer healthcare AI! 🩺
