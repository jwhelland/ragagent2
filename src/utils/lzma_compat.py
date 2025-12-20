"""Ensure a working `lzma` module even when Python was built without it."""

from __future__ import annotations

import sys
from types import ModuleType


def ensure_lzma() -> ModuleType:
    """Import stdlib lzma, falling back to backports if missing.

    Returns:
        The imported lzma module.

    Raises:
        RuntimeError: If no lzma implementation is available.
    """
    try:
        import lzma as _lzma

        return _lzma
    except ModuleNotFoundError:
        try:
            import backports.lzma as _lzma  # type: ignore
        except ModuleNotFoundError:
            # Provide a lightweight shim so libraries that only import lzma can proceed.
            shim = ModuleType("lzma")

            def _unavailable(*_args: object, **_kwargs: object) -> None:  # pragma: no cover
                raise RuntimeError(
                    "lzma support is unavailable. Install liblzma/xz and reinstall Python, "
                    "or install backports.lzma with development headers."
                )

            shim.LZMAFile = _unavailable
            shim.open = _unavailable
            shim.compress = _unavailable
            shim.decompress = _unavailable
            shim.LZMACompressor = _unavailable
            shim.LZMADecompressor = _unavailable
            sys.modules.setdefault("lzma", shim)
            return shim

        # Register fallback so downstream imports resolve `import lzma`.
        sys.modules.setdefault("lzma", _lzma)
        return _lzma
