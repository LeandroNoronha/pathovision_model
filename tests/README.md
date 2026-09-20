# Tests

The test suite of the main pipeline has not been released yet; this directory is kept as a placeholder.

The tests that currently ship with the repository cover the independent re-execution package and live in `reproduction/tests/`:

```bash
python -m pytest reproduction/tests/ -v
```

They check the evidence generator (`reproduction/scripts/generate_evidence.py`) and the figure generator (`reproduction/scripts/generate_all_figures.py`).
