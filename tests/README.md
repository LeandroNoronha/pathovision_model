# Tests

This directory contains unit and integration tests for the PathoVision v2 codebase.

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_models.py
```

## Test Coverage

Tests cover:

- **Data pipeline**: Dataset loading, transforms, augmentation correctness, sampler behavior
- **Models**: Forward pass shapes, feature extraction, checkpoint save/load, head configurations
- **Training**: Loss computation, gradient accumulation, mixed precision, callback triggers
- **Evaluation**: Metric computation, confusion matrix generation, report formatting
- **Hybrid pipeline**: Feature extraction dimensions, classical ML classifier fitting

## Notes

- Tests that require GPU are skipped automatically when no CUDA device is available
- Tests that require dataset files are skipped if `datasets/data/` is not present
- Use `pytest -m "not slow"` to skip long-running integration tests
