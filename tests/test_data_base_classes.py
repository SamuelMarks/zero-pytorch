"""Tests for base data classes, Samplers, DataLoader, and `default_collate`."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterator

import pytest

import zero_torch
from zero_torch.utils.data import BatchSampler, Dataset, IterableDataset, Sampler
from zero_torch.utils.data.dataloader import DataLoader, default_collate


def test_dataset_base() -> None:
    """Verifies `Dataset` abstract base class raises `NotImplementedError`."""
    d: Dataset[int] = Dataset()
    with pytest.raises(NotImplementedError):
        len(d)
    with pytest.raises(NotImplementedError):
        _ = d[0]


def test_iterable_dataset_base() -> None:
    """Verifies `IterableDataset` abstract base class raises `NotImplementedError`."""
    d: IterableDataset[int] = IterableDataset()
    with pytest.raises(NotImplementedError):
        iter(d)


def test_sampler_base() -> None:
    """Verifies `Sampler` abstract base class raises `NotImplementedError`."""
    s: Sampler[int] = Sampler()
    with pytest.raises(NotImplementedError):
        iter(s)


def test_batch_sampler_errors() -> None:
    """Verifies `BatchSampler` validation errors and batch yielding behavior."""
    with pytest.raises(ValueError):
        BatchSampler([], batch_size=0, drop_last=False)

    s = BatchSampler([1, 2, 3], batch_size=2, drop_last=False)
    assert len(s) == 2

    s2 = BatchSampler([1, 2, 3], batch_size=2, drop_last=True)
    assert len(s2) == 1

    assert list(iter(s)) == [[1, 2], [3]]
    assert list(iter(s2)) == [[1, 2]]


def test_default_collate() -> None:
    """Verifies `default_collate` across Tensors, scalars, strings, dicts, namedtuples, and tuples."""
    t1 = zero_torch.tensor([1.0])
    t2 = zero_torch.tensor([2.0])
    c_t = default_collate([t1, t2])
    assert isinstance(c_t, zero_torch.Tensor) and c_t.shape == (2, 1)

    c_f = default_collate([1.0, 2.0])
    assert isinstance(c_f, zero_torch.Tensor) and c_f.shape == (2,)

    c_i = default_collate([1, 2])
    assert isinstance(c_i, zero_torch.Tensor) and c_i.shape == (2,)

    assert default_collate(["a", "b"]) == ["a", "b"]

    d1: dict[str, int | float] = {"a": 1, "b": 2.0}
    d2: dict[str, int | float] = {"a": 3, "b": 4.0}
    c_d = default_collate([d1, d2])
    assert isinstance(c_d, dict) and "a" in c_d

    Point = namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    c_p = default_collate([p1, p2])
    assert isinstance(c_p, Point)

    c_tup = default_collate([(1, 2.0), (3, 4.0)])
    assert isinstance(c_tup, list) and len(c_tup) == 2


def test_dataloader_more() -> None:
    """Verifies `DataLoader` with map-style `Dataset` and unimplemented `Sampler`."""

    class DummyDataset(Dataset[int]):
        """Simple integer dataset of length 4."""

        def __len__(self) -> int:
            """Returns dataset length 4.

            Returns:
                Length 4.
            """
            return 4

        def __getitem__(self, idx: int) -> int:
            """Returns `idx`.

            Args:
                idx: Sample index.

            Returns:
                Index integer.
            """
            return idx

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    assert len(list(iter(dl))) == 2

    dl_bad = DataLoader(ds, batch_size=None, sampler=Sampler())
    with pytest.raises(NotImplementedError):
        list(dl_bad)


def test_dataloader_more2() -> None:
    """Verifies `DataLoader` with `shuffle=True`, custom `Sampler`, and `sampler=None`."""

    class DummyDataset(Dataset[int]):
        """Simple integer dataset of length 4."""

        def __len__(self) -> int:
            """Returns dataset length 4.

            Returns:
                Length 4.
            """
            return 4

        def __getitem__(self, idx: int) -> int:
            """Returns `idx`.

            Args:
                idx: Sample index.

            Returns:
                Index integer.
            """
            return idx

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    assert len(list(dl)) == 2

    class DummySampler(Sampler[int]):
        """Sampler yielding indices 0 and 1."""

        def __iter__(self) -> Iterator[int]:
            """Yields 0 and 1.

            Yields:
                Indices 0 and 1.
            """
            yield 0
            yield 1

    dl2 = DataLoader(ds, batch_size=None, sampler=DummySampler())
    res = list(dl2)
    assert res == [0, 1]


def test_dataloader_iterable_dataset() -> None:
    """Verifies `DataLoader` over an `IterableDataset`."""

    class DummyIterable(IterableDataset[int]):
        """IterableDataset yielding 1, 2, 3."""

        def __iter__(self) -> Iterator[int]:
            """Yields 1, 2, 3.

            Yields:
                Integers 1, 2, 3.
            """
            yield 1
            yield 2
            yield 3

    ds = DummyIterable()
    dl = DataLoader(ds, batch_size=None)
    assert list(dl) == [1, 2, 3]
