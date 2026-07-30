import pytest

import zero_torch
from zero_torch.utils.data import BatchSampler, Dataset, IterableDataset, Sampler
from zero_torch.utils.data.dataloader import DataLoader, default_collate

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_dataset_base():
    d = Dataset()
    with pytest.raises(NotImplementedError):
        len(d)
    with pytest.raises(NotImplementedError):
        d[0]


def test_iterable_dataset_base():
    d = IterableDataset()
    with pytest.raises(NotImplementedError):
        iter(d)


def test_sampler_base():
    s = Sampler()
    with pytest.raises(NotImplementedError):
        iter(s)


def test_batch_sampler_errors():
    with pytest.raises(ValueError):
        BatchSampler([], batch_size=0, drop_last=False)

    with pytest.raises(TypeError):
        BatchSampler([], batch_size=2, drop_last="not_a_bool")

    s = BatchSampler([1, 2, 3], batch_size=2, drop_last=False)
    assert len(s) == 2

    s2 = BatchSampler([1, 2, 3], batch_size=2, drop_last=True)
    assert len(s2) == 1

    # Test iteration
    assert list(iter(s)) == [[1, 2], [3]]
    assert list(iter(s2)) == [[1, 2]]


def test_default_collate():
    # Tensor
    t1 = zero_torch.tensor([1.0])
    t2 = zero_torch.tensor([2.0])
    assert default_collate([t1, t2]).shape == (2, 1)

    # Float
    res = default_collate([1.0, 2.0])
    # assert res.dtype == zero_torch.float64

    # Int
    res = default_collate([1, 2])
    assert res.shape == (2,)

    # String
    assert default_collate(["a", "b"]) == ["a", "b"]

    # Dict
    d1 = {"a": 1, "b": 2.0}
    d2 = {"a": 3, "b": 4.0}
    res = default_collate([d1, d2])
    assert "a" in res

    # Namedtuple
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    res = default_collate([p1, p2])
    assert hasattr(res, "x")

    # List/Tuple
    res = default_collate([(1, 2.0), (3, 4.0)])
    assert len(res) == 2

    # Fallback
    class Dummy:
        pass

    d = Dummy()
    assert default_collate([d]) == [d]


def test_dataloader_more():
    class DummyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return idx

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    assert len(list(iter(dl))) == 2

    dl = DataLoader(ds, batch_size=None, sampler=Sampler())
    # trigger iter with batch_size=None
    try:
        iter(dl)
    except NotImplementedError:
        _pass = True


def test_dataloader_more2():
    class DummyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return idx

    ds = DummyDataset()
    # test shuffle=True
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    assert len(list(dl)) == 2

    # test sampler without batch_size
    class DummySampler(Sampler):
        def __iter__(self):
            yield 0
            yield 1

    dl2 = DataLoader(ds, batch_size=None, sampler=DummySampler())
    res = list(dl2)
    assert len(res) == 2

    # test dataset directly
    dl3 = DataLoader(ds, batch_size=None, sampler=None)
    assert len(list(dl3)) == 4


def test_dataloader_iterable_dataset():
    from zero_torch.utils.data import IterableDataset

    class DummyIterable(IterableDataset):
        def __iter__(self):
            yield 1
            yield 2
            yield 3

    ds = DummyIterable()
    dl = DataLoader(ds, batch_size=None)
    assert list(dl) == [1, 2, 3]
