# Contributing

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/venkywonka/tiny-trtllm.git
cd tiny-trtllm
pip install -e ".[dev]"
```

## Running Tests

```bash
# All unit tests (CPU)
pytest tests/ -x -v

# With coverage
pytest tests/ --cov=tinytrtllm --cov-report=term-missing

# GPU tests only
pytest tests/ -m gpu -v

# Specific test file
pytest tests/test_scheduler.py -v
```

## Code Quality

```bash
# Lint
ruff check tinytrtllm/ tests/

# Format
ruff format tinytrtllm/ tests/

# Count LOC (target: under 5,000)
find tinytrtllm -name '*.py' | xargs wc -l

# Verify zero C++
find . -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' | head
```

## Building Docs

```bash
cd docs
pip install -r requirements.txt
make html
# Open build/html/index.html
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `gpu` | Requires CUDA GPU |
| `slow` | Slow integration tests |
| `tp` | Requires multi-GPU tensor parallelism |
| `benchmark` | Throughput benchmarks |
