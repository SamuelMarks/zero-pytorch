"""Tests for base data classes, Samplers, DataLoader, and `default_collate`."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path

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
        d.__getitem__(0)


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
    assert len(dl) == 2
    assert len(list(iter(dl))) == 2

    dl_unbatched = DataLoader(ds, batch_size=None)
    assert len(dl_unbatched) == 4

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

    dl_no_sampler = DataLoader(ds, batch_size=None)
    dl_no_sampler.sampler = None
    assert len(dl_no_sampler) == 4


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
    with pytest.raises(TypeError):
        len(dl)


def test_samplers_generator_random_split_and_worker_init() -> None:
    """Verifies deterministic `generator` seeding on Samplers, fractional `random_split`, and `worker_init_fn`."""
    from ml_switcheroo_compiler.core.config import EagerMode

    from zero_torch.utils.data import (
        RandomSampler,
        SubsetRandomSampler,
        TensorDataset,
        WeightedRandomSampler,
        random_split,
    )

    with EagerMode():
        ds = TensorDataset(zero_torch.arange(10))
        rs1 = list(RandomSampler(ds, generator=42))
        rs2 = list(RandomSampler(ds, generator=42))
        assert rs1 == rs2
        rs_fn = list(RandomSampler(ds, generator=lambda: 0.42))
        assert len(rs_fn) == 10
        rs_fn_int = list(RandomSampler(ds, generator=lambda: 42.0))
        assert len(rs_fn_int) == 10

        srs1 = list(SubsetRandomSampler([0, 1, 2, 3], generator=99))
        srs2 = list(SubsetRandomSampler([0, 1, 2, 3], generator=99))
        assert srs1 == srs2

        wrs1 = list(WeightedRandomSampler([1.0, 2.0, 3.0], num_samples=4, generator=7))
        wrs2 = list(WeightedRandomSampler([1.0, 2.0, 3.0], num_samples=4, generator=7))
        assert wrs1 == wrs2

        parts_int = random_split(ds, [7, 3], generator=123)
        assert [len(p) for p in parts_int] == [7, 3]

        parts_frac = random_split(
            TensorDataset(zero_torch.arange(5)), [0.5, 0.5], generator=123
        )
        assert [len(p) for p in parts_frac] == [3, 2]

        with pytest.raises(ValueError):
            random_split(ds, [4, 4])

        worker_ids: list[int] = []
        dl = DataLoader(
            ds, batch_size=5, num_workers=2, worker_init_fn=worker_ids.append
        )
        assert len(list(dl)) == 2
        assert worker_ids == [0, 1]


def test_distributed_stores_and_multi_rank_reductions(tmp_path: Path) -> None:
    """Verifies `Store`, `HashStore`, `TCPStore`, `FileStore`, and multi-rank `ReduceOp` reductions.

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    from ml_switcheroo_compiler.core.config import EagerMode

    import zero_torch.distributed as dist

    with EagerMode():
        hs = dist.HashStore()
        hs.set("k1", "val1")
        hs.set("k2", b"val2")
        assert hs.get("k1") == b"val1"
        assert hs.get("k2") == b"val2"
        assert hs.add("cnt", 3) == 3
        assert hs.add("cnt", 2) == 5
        hs.set("bad_int", "abc")
        assert hs.add("bad_int", 4) == 4
        hs.wait(["k1", "cnt"])
        with pytest.raises(KeyError):
            hs.wait(["missing_key"])
        assert hs.num_keys() == 4
        assert hs.delete_key("k1") is True
        assert hs.delete_key("k1") is False

        ts = dist.TCPStore("127.0.0.1", 29501, world_size=2, is_master=True)
        ts.set("host_key", "ok")
        assert ts.get("host_key") == b"ok"

        store_file = Path(str(tmp_path)) / "fstore.txt"
        fs1 = dist.FileStore(str(store_file), world_size=1)
        fs1.set("persisted", "yes")
        store_file.write_text("persisted=yes\nmalformed_line\n", encoding="utf-8")
        fs2 = dist.FileStore(str(store_file), world_size=1)
        assert fs2.get("persisted") == b"yes"
        fs_missing_dir = dist.FileStore(str(Path(str(tmp_path)) / "no_dir" / "f.txt"))
        fs_missing_dir.set("k", "v")

        pg = dist.ProcessGroup(rank=0, world_size=2)
        pg.rank_buffers[1] = zero_torch.tensor([4.0, 2.0])

        t_sum = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, group=pg)
        assert t_sum.tolist() == [6.0, 8.0]

        t_prod = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce(t_prod, op=dist.ReduceOp.PRODUCT, group=pg)
        assert t_prod.tolist() == [8.0, 12.0]

        t_min = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce(t_min, op=dist.ReduceOp.MIN, group=pg)
        assert t_min.tolist() == [2.0, 2.0]

        t_max = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=pg)
        assert t_max.tolist() == [4.0, 6.0]

        t_avg = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce(t_avg, op=dist.ReduceOp.AVG, group=pg)
        assert t_avg.tolist() == [3.0, 4.0]

        t_coal = zero_torch.tensor([2.0, 6.0])
        dist.all_reduce_coalesced(t_coal, op=dist.ReduceOp.SUM, group=pg)
        assert t_coal.tolist() == [6.0, 8.0]

        out_rs = zero_torch.zeros((2,))
        dist.reduce_scatter(
            out_rs,
            [zero_torch.tensor([1.0, 3.0]), zero_torch.tensor([5.0, 7.0])],
            op=dist.ReduceOp.AVG,
            group=pg,
        )
        assert out_rs.tolist() == [3.0, 5.0]
        assert dist._apply_reduce_op(
            [zero_torch.tensor([2.0])], dist.ReduceOp.AVG
        ).tolist() == [2.0]
