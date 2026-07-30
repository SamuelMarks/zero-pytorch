"DataLoader System."

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

_T_co = TypeVar("_T_co", covariant=True)
_collate_fn_t = Callable[[list[Any]], Any]
_worker_init_fn_t = Callable[[int], None]


class Dataset(Generic[_T_co]):
    """An abstract class representing a Dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the dataset.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches a data sample for a given key.

        Args:
            idx (int): The index of the item.

        Returns:
            _T_co: The data sample.
        """
        raise NotImplementedError()


class Sampler(Generic[_T_co]):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method.
    """

    def __init__(self, data_source: Dataset | None = None) -> None:
        """Initializes the sampler.

        Args:
            data_source (Dataset, optional): Dataset to sample from. Defaults to None.
        """

    def __iter__(self):
        raise NotImplementedError()


class BatchSampler(Sampler[list[int]]):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(
        self, sampler: Sampler | Iterable, batch_size: int, drop_last: bool
    ) -> None:
        """Initializes the BatchSampler.

        Args:
            sampler (Union[Sampler, Iterable]): Base sampler.
            batch_size (int): Size of mini-batch.
            drop_last (bool): If True, the sampler will drop the last batch if its size would be less than batch_size.
        """
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise TypeError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        """Iterates over batches of indices.

        Yields:
            Iterator[List[int]]: An iterator containing lists of indices.
        """
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Returns the number of batches.

        Returns:
            int: The number of batches.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def default_collate(batch: list[Any]) -> Any:
    """Puts each data field into a tensor with outer dimension batch size.

    Args:
        batch (List[Any]): A list of data samples.

    Returns:
        Any: Collated data batch.
    """
    import zero_torch

    elem = batch[0]
    if isinstance(elem, zero_torch.Tensor):
        if get_worker_info() is not None:
            pass  # just a dummy check
        return zero_torch.stack(batch, 0)
    elif isinstance(elem, (float, int)):
        return zero_torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem.__class__(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    return batch


def get_worker_info():
    return None


class DataLoader(Generic[_T_co]):
    """Data loader combines a dataset and a sampler, and provides an iterable over the given dataset."""

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        if self.batch_sampler is None:
            from zero_torch.utils.data import IterableDataset

            if self.sampler is None and not isinstance(dataset, IterableDataset):
                if shuffle:
                    from zero_torch.utils.data import RandomSampler

                    self.sampler = RandomSampler(dataset, generator=generator)
                else:
                    from zero_torch.utils.data import SequentialSampler

                    self.sampler = SequentialSampler(dataset)

            if self.batch_size is not None:
                self.batch_sampler = BatchSampler(
                    self.sampler, self.batch_size, self.drop_last
                )

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator for the data loader.

        Returns:
            Iterator[Any]: An iterator that yields batches of data.
        """
        if self.batch_sampler is not None:
            for indices in self.batch_sampler:
                batch = [self.dataset[i] for i in indices]
                yield self.collate_fn(batch)
        elif self.sampler is not None:
            for index in self.sampler:
                yield self.dataset[index]
        else:
            yield from self.dataset
