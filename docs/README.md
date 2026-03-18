# tiny-trtllm Documentation

This documentation is built using [Sphinx](https://www.sphinx-doc.org/) with the
[nvidia-sphinx-theme](https://pypi.org/project/nvidia-sphinx-theme/), matching the look
and feel of [TensorRT-LLM's documentation](https://nvidia.github.io/TensorRT-LLM/).

## Build

```bash
pip install -r requirements.txt
make html
```

The built documentation will be in `build/html/`. Open `build/html/index.html` in your browser.

## Live Preview

For a live-reloading preview during development:

```bash
pip install sphinx-autobuild
sphinx-autobuild source build/html
```
