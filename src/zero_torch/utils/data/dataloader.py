"""DataLoader System."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from typing import Generic, Protocol, TypeVar, runtime_checkable

import zero_torch
from zero_torch._types import CollateValue, OpArg

_T_co = TypeVar("_T_co", bound=CollateValue, covariant=True)
_CollateFn = Callable[[Sequence[CollateValue]], CollateValue]
_WorkerInitFn = Callable[[int], None]


@runtime_checkable
class _NamedTupleProtocol(Protocol):
    """Protocol for namedtuples with `_fields` and `_make`."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """Returns the field names of the namedtuple.

        Returns:
            Tuple of field names.
        """

    @classmethod
    def _make(cls, iterable: Iterable[CollateValue]) -> CollateValue:
        """Creates a new namedtuple instance from an iterable of field values.

        Args:
            iterable: Iterable of collated field values.

        Returns:
            Constructed namedtuple instance.
        """


class Dataset(Generic[_T_co]):
    """An abstract class representing a Dataset."""

    def __init__(self, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes the Dataset.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches a data sample for a given index key.

        Args:
            idx: Index of the item to fetch.

        Returns:
            The data sample at `idx`.
        """
        _ = idx
        raise NotImplementedError


class Sampler(Generic[_T_co]):
    """Base class for all Samplers."""

    def __init__(self, data_source: Sized | None = None) -> None:
        """Initializes the Sampler.

        Args:
            data_source: Dataset or sized object to sample from.
        """
        self.data_source: Sized | None = data_source

    def __iter__(self) -> Iterator[_T_co]:
        """Returns an iterator over sample indices.

        Returns:
            Iterator over elements of the sampler.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of samples in the sampler.

        Returns:
            Length of the sampler.
        """
        if self.data_source is not None:
            return len(self.data_source)
        raise NotImplementedError


class BatchSampler(Sampler[list[int]]):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        """Initializes the BatchSampler.

        Args:
            sampler: Base sampler or iterable of indices.
            batch_size: Size of mini-batch.
            drop_last: If True, drops the last batch if smaller than `batch_size`.
        """
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        self.sampler: Sampler[int] | Iterable[int] = sampler
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        """Iterates over mini-batches of indices.

        Yields:
            Lists of sample indices of length up to `batch_size`.
        """
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Returns the number of mini-batches.

        Returns:
            Total number of batches yielded by this sampler.
        """
        sampler_len = len(self.sampler) if isinstance(self.sampler, Sized) else 0
        if self.drop_last:
            return sampler_len // self.batch_size
        return (sampler_len + self.batch_size - 1) // self.batch_size


def default_collate(batch: Sequence[CollateValue]) -> CollateValue:
    """Puts each data field into a tensor or collection with outer dimension batch size.

    Args:
        batch: Sequence of data samples to collate.

    Returns:
        Collated data batch.
    """
    elem = batch[0]
    if isinstance(elem, zero_torch.Tensor):
        _ = get_worker_info()
        tensor_batch = [b for b in batch if isinstance(b, zero_torch.Tensor)]
        return zero_torch.stack(tensor_batch, dim=0)
    if isinstance(elem, (float, int)):
        num_batch = [b for b in batch if isinstance(b, (float, int))]
        return zero_torch.tensor(num_batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, dict):
        dict_batch = [b for b in batch if isinstance(b, dict)]
        return {key: default_collate([d[key] for d in dict_batch]) for key in elem}
    if isinstance(elem, tuple) and isinstance(elem, _NamedTupleProtocol):
        tuple_batch = [b for b in batch if isinstance(b, tuple)]
        collated_fields = [
            default_collate(list(samples)) for samples in zip(*tuple_batch)
        ]
        return elem._make(collated_fields)
    if isinstance(elem, (tuple, list)):
        seq_batch = [b for b in batch if isinstance(b, (tuple, list))]
        return [default_collate(list(samples)) for samples in zip(*seq_batch)]
    return list(batch)


def get_worker_info() -> dict[str, int] | None:
    """Returns information about the current DataLoader worker process.

    Returns:
        Worker info dictionary if in a worker process, else None.
    """
    return None


class DataLoader(Generic[_T_co]):
    """Data loader combines a dataset and a sampler, providing an iterable over the dataset."""

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler[int] | Iterable[int] | None = None,
        batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None = None,
        num_workers: int = 0,
        collate_fn: _CollateFn | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.0,
        worker_init_fn: _WorkerInitFn | None = None,
        multiprocessing_context: str | None = None,
        generator: Callable[[], float] | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        """Initializes the DataLoader.

        Args:
            dataset: Dataset from which to load the data.
            batch_size: How many samples per batch to load.
            shuffle: Set to True to have the data reshuffled at every epoch.
            sampler: Defines the strategy to draw samples from the dataset.
            batch_sampler: Like `sampler`, but returns a batch of indices at a time.
            num_workers: How many subprocesses to use for data loading.
            collate_fn: Merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory.
            drop_last: Set to True to drop the last incomplete batch.
            timeout: If positive, the timeout value for collecting a batch from workers.
            worker_init_fn: Function called on each worker subprocess after seeding.
            multiprocessing_context: Multiprocessing context name.
            generator: Random number generator used by RandomSampler.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: If True, the data loader will not shut down worker processes after a dataset has been consumed once.
            pin_memory_device: Device name for pinned memory.
            in_order: If False, allows out-of-order batch delivery from workers.
        """
        _ = (
            num_workers,
            pin_memory,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            prefetch_factor,
            persistent_workers,
            pin_memory_device,
            in_order,
        )
        self.dataset: Dataset[_T_co] = dataset
        self.batch_size: int | None = batch_size
        self.drop_last: bool = drop_last
        self.sampler: Sampler[int] | Iterable[int] | None = sampler
        self.batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None = (
            batch_sampler
        )
        self.collate_fn: _CollateFn = (
            collate_fn if collate_fn is not None else default_collate
        )

        if self.batch_sampler is None:
            from zero_torch.utils.data import (
                IterableDataset,
                RandomSampler,
                SequentialSampler,
            )

            if self.sampler is None and not isinstance(dataset, IterableDataset):
                if shuffle:
                    self.sampler = RandomSampler(dataset, generator=generator)
                else:
                    self.sampler = SequentialSampler(dataset)

            if self.batch_size is not None and self.sampler is not None:
                self.batch_sampler = BatchSampler(
                    self.sampler, self.batch_size, self.drop_last
                )

    def __iter__(self) -> Iterator[CollateValue | _T_co]:
        """Returns an iterator over the dataset batches or items.

        Yields:
            Collated batches or dataset items.
        """
        if self.batch_sampler is not None:
            for indices in self.batch_sampler:
                batch_items: list[_T_co] = [self.dataset[i] for i in indices]
                yield self.collate_fn(batch_items)
        elif self.sampler is not None:
            for index in self.sampler:
                yield self.dataset[index]
        elif isinstance(self.dataset, Iterable):
            yield from self.dataset
