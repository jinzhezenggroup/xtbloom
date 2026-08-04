"""dpdata plugin modules for gpuxtb.

The ``dpdata.plugins`` entry point declared in ``pyproject.toml`` points at
``gpuxtb.plugins.dpdata``, so importing ``dpdata`` loads this module and
registers :class:`~gpuxtb.plugins.dpdata.GPUxtbDriver` under the key
``"gpuxtb"``.
"""