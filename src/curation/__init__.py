"""Curation package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from src.curation.review_interface import app as app
    from src.curation.review_interface import run as run

__all__ = ["app", "run"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from src.curation.review_interface import app, run

    return {"app": app, "run": run}[name]
