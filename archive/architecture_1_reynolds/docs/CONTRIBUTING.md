# Contributing to Blood Cell Analysis System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, TensorFlow version, OS)
- Error messages or logs

### Suggesting Enhancements

For feature requests or enhancements:
- Open an issue describing the feature
- Explain the use case and benefits
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/akshatac2e/Multi-Class-Disease-Classification-Model-using-Reynolds-Networks.git
   cd Multi-Class-Disease-Classification-Model-using-Reynolds-Networks
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update documentation if needed
   - Add tests if applicable

4. **Test your changes**
   ```bash
   python setup.py  # Verify setup
   python example_usage.py  # Test basic functionality
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of feature"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub

## Code Style

### Python Code
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings to all functions/classes
- Keep functions focused and modular

Example:
```python
def extract_cells(image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual cells from segmentation mask.
    
    Args:
        image: RGB image array (H, W, 3)
        mask: Segmentation mask (H, W, 3)
    
    Returns:
        List of extracted cell images
    """
    # Implementation
    pass
```

### Documentation
- Update README.md if adding major features
- Update README_SYSTEM.md for technical changes
- Add examples to example_usage.py if relevant
- Keep comments clear and concise

## Development Setup

1. **Install in development mode**
   ```bash
   pip install -r requirements.txt
   pip install pytest  # For testing
   ```

2. **Run setup verification**
   ```bash
   python setup.py
   ```

3. **Test your changes**
   - Test with small datasets first
   - Verify all three models still work
   - Check inference pipeline
   - Test batch processing

## Areas for Contribution

### High Priority
- [ ] Add more cell types (basophils, eosinophils, etc.)
- [ ] Implement additional augmentation strategies
- [ ] Add model export to ONNX/TFLite
- [ ] Create web interface for inference
- [ ] Add more comprehensive tests

### Medium Priority
- [ ] Improve segmentation model accuracy
- [ ] Add visualization tools for training metrics
- [ ] Create dataset preparation scripts
- [ ] Add support for more image formats
- [ ] Improve documentation with more examples

### Low Priority
- [ ] Add support for 3D microscopy stacks
- [ ] Implement ensemble models
- [ ] Add model compression techniques
- [ ] Create mobile app wrapper

## Testing Guidelines

When adding new features:
1. Test with minimal data first
2. Verify backward compatibility
3. Test edge cases (empty images, single cells, etc.)
4. Check memory usage with large batches
5. Verify output formats haven't changed

## Documentation Guidelines

- Keep README.md concise and user-friendly
- Put detailed technical info in README_SYSTEM.md
- Add working examples to example_usage.py
- Use clear section headers and formatting
- Include code examples where helpful

## Questions?

If you have questions about contributing:
- Open a discussion on GitHub
- Check existing issues for similar questions
- Review the documentation thoroughly

## Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Focus on the technical aspects
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to medical AI research! 🔬🩸
