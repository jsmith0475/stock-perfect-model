# Contributing to Stock Perfect Model

First off, thank you for considering contributing to Stock Perfect Model! It's people like you that make this project better.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (include code snippets if applicable)
- **Describe the behavior you observed and what you expected**
- **Include your environment** (Python version, OS, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed feature**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Make your changes** with clear, descriptive commit messages
4. **Test your changes** to ensure they work correctly
5. **Update documentation** if you're changing functionality
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/stock-perfect-model.git
cd stock-perfect-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python stock_perfect.py
```

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Comment complex mathematical operations

## Areas for Contribution

### High Priority
- [ ] Rolling window topology (time-series H1 persistence)
- [ ] Backtesting framework with transaction costs
- [ ] Unit tests for core functions
- [ ] Performance optimization for large universes

### Medium Priority
- [ ] Sector-level hierarchical graphs
- [ ] Additional distance metrics (mutual information, etc.)
- [ ] Visualization improvements
- [ ] Real-time streaming support

### Documentation
- [ ] More usage examples
- [ ] Video tutorials
- [ ] Jupyter notebook demos

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

---

Thank you for contributing! 🎉

