API Reference
=============

.. _api-reference:

Configuration
-------------

.. autopydantic_model:: tinytrtllm.config.TinyLlmArgs
   :members:

.. autopydantic_model:: tinytrtllm.config.SamplingParams
   :members:

LLM API
-------

.. autoclass:: tinytrtllm.llm.LLM
   :members:
   :undoc-members:

Engine
------

.. automodule:: tinytrtllm.engine.executor
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.scheduler
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.request
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.block_manager
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.sampler
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.resource_manager
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.engine.model_engine
   :members:
   :undoc-members:

Layers
------

.. automodule:: tinytrtllm.layers.attention
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.layers.linear
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.layers.embedding
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.layers.moe
   :members:
   :undoc-members:

Models
------

.. automodule:: tinytrtllm.models.base
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.models.llama
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.models.qwen3
   :members:
   :undoc-members:

.. automodule:: tinytrtllm.models.qwen3_moe
   :members:
   :undoc-members:

Server
------

.. automodule:: tinytrtllm.serve.server
   :members:
   :undoc-members:
