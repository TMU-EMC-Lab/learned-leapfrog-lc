"""Optional tqdm progress helpers (graceful fallback when tqdm is absent)."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    disable: bool = False,
    unit: str = "it",
    leave: bool = True,
) -> Iterable[T]:
    """Wrap *iterable* with tqdm when available and not disabled."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def set_postfix(bar: Iterable[Any], **kwargs: Any) -> None:
    if hasattr(bar, "set_postfix"):
        bar.set_postfix(**kwargs, refresh=True)


def progress_write(msg: str, *, disable: bool = False) -> None:
    if disable:
        print(msg)
        return
    try:
        from tqdm import tqdm

        tqdm.write(msg)
    except ImportError:
        print(msg)
