"""Data utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from typing import TypeVar

from zero_torch._types import CollateValue, OpArg
from zero_torch.tensor import Tensor

from .dataloader import BatchSampler, DataLoader, Dataset, Sampler

__all__ = [
    "BatchSampler",
    "ChainDataset",
    "ConcatDataset",
    "DFIterDataPipe",
    "DataChunk",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "IterDataPipe",
    "IterableDataset",
    "MapDataPipe",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "StackDataset",
    "Subset",
    "SubsetRandomSampler",
    "TensorDataset",
    "WeightedRandomSampler",
    "functional_datapipe",
    "guaranteed_datapipes_determinism",
    "non_deterministic",
    "random_split",
    "runtime_validation_disabled",
]
_T_co = TypeVar("_T_co", bound=CollateValue, covariant=True)


def _make_rng(generator: Callable[[], float] | int | None) -> random.Random:
    """Creates a `random.Random` instance seeded from `generator` when provided.

    Args:
        generator: Optional seed integer, callable returning float/int, or None.

    Returns:
        Configured `random.Random` instance.
    """
    if isinstance(generator, int):
        return random.Random(generator)
    if callable(generator):
        val = generator()
        seed = (
            int(val * 1_000_000)
            if isinstance(val, float) and abs(val) < 1.0
            else int(val)
        )
        return random.Random(seed)
    return random.Random()


class IterableDataset(Dataset[_T_co]):
    """An iterable Dataset."""

    def __init__(self, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes the IterableDataset.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def __iter__(self) -> Iterator[_T_co]:
        """Returns an iterator over the dataset elements.

        Returns:
            Iterator over dataset items.
        """
        raise NotImplementedError


class ChainDataset(IterableDataset[_T_co]):
    """Dataset for chaining multiple IterableDatasets."""

    def __init__(self, datasets: Iterable[Dataset[_T_co]]) -> None:
        """Initializes the ChainDataset.

        Args:
            datasets: Iterable of datasets to be chained together.
        """
        super().__init__()
        self.datasets: list[Dataset[_T_co]] = list(datasets)

    def __iter__(self) -> Iterator[_T_co]:
        """Iterates sequentially over all chained datasets.

        Yields:
            Items from each dataset in `self.datasets`.
        """
        for d in self.datasets:
            if isinstance(d, IterableDataset):
                yield from d
            else:
                for i in range(len(d)):
                    yield d[i]

    def __len__(self) -> int:
        """Returns the sum of lengths of all chained datasets.

        Returns:
            Total number of items across all datasets.
        """
        return sum(len(d) for d in self.datasets)


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets."""

    def __init__(
        self, datasets: list[Dataset[_T_co]], cumulative_sizes: list[int] | None = None
    ) -> None:
        """Initializes the ConcatDataset.

        Args:
            datasets: List of datasets to be concatenated.
            cumulative_sizes: Optional list of cumulative sizes of the datasets.
        """
        super().__init__()
        self.datasets: list[Dataset[_T_co]] = list(datasets)
        if cumulative_sizes is not None:
            self.cumulative_sizes: list[int] = list(cumulative_sizes)
        else:
            acc: list[int] = []
            running = 0
            for d in self.datasets:
                try:
                    running += len(d)
                except NotImplementedError:
                    running += 0
                acc.append(running)
            self.cumulative_sizes = acc

    def __len__(self) -> int:
        """Returns the total length of the concatenated dataset.

        Returns:
            Total sample count.
        """
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches the item at `idx` from the appropriate constituent dataset.

        Args:
            idx: Index in `[-len(self), len(self))`.

        Returns:
            Sample at `idx`.
        """
        total = len(self)
        if idx < 0:
            if -idx > total:
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = total + idx
        if idx >= total or idx < 0:
            raise IndexError("ConcatDataset index out of range")
        dataset_idx = next(
            i for i, cum_sz in enumerate(self.cumulative_sizes) if idx < cum_sz
        )
        sample_idx = (
            idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        )
        return self.datasets[dataset_idx][sample_idx]


class DFIterDataPipe:
    """DataFrame-oriented iterable-style DataPipe."""

    def __init__(self, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes the DFIterDataPipe.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.args: tuple[OpArg, ...] = args
        self.kwargs: dict[str, OpArg] = dict(kwargs)


class DataChunk:
    """Container wrapping a chunk of items yielded by a DataPipe."""

    def __init__(self, items: Iterable[CollateValue]) -> None:
        """Initializes the DataChunk.

        Args:
            items: Iterable of items to populate the chunk.
        """
        self.items: list[CollateValue] = list(items)

    def __iter__(self) -> Iterator[CollateValue]:
        """Iterates over the items in the chunk.

        Returns:
            Iterator over `self.items`.
        """
        return iter(self.items)

    def __len__(self) -> int:
        """Returns the number of items in the chunk.

        Returns:
            Length of `self.items`.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> CollateValue:
        """Returns the item at `idx` in the chunk.

        Args:
            idx: Item index.

        Returns:
            Item at `idx`.
        """
        return self.items[idx]

    def raw_iterator(self) -> Iterator[CollateValue]:
        """Returns a raw iterator over the underlying items.

        Returns:
            Iterator over `self.items`.
        """
        return iter(self.items)


class DistributedSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset across distributed workers."""

    def __init__(
        self,
        dataset: Dataset[_T_co],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool | None = True,
        seed: int | None = 0,
        drop_last: bool | None = False,
    ) -> None:
        """Initializes DistributedSampler.

        Args:
            dataset: Dataset used for sampling.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process within `num_replicas`.
            shuffle: If True, sampler will shuffle the indices.
            seed: Random seed used to shuffle the sampler.
            drop_last: If True, drops the tail of the data to make it evenly divisible.
        """
        super().__init__(dataset)
        self.dataset: Dataset[_T_co] = dataset
        self.num_replicas: int = num_replicas if num_replicas is not None else 1
        self.rank: int = rank if rank is not None else 0
        self.epoch: int = 0
        self.drop_last: bool = bool(drop_last)
        self.shuffle: bool = True if shuffle is None else shuffle
        self.seed: int = seed if seed is not None else 0

    def __iter__(self) -> Iterator[int]:
        """Yields distributed shard indices for `self.rank`.

        Returns:
            Iterator over shard indices.
        """
        ds_len = len(self.dataset)
        indices = list(range(ds_len))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.drop_last:
            num_samples = ds_len // self.num_replicas
            total_size = num_samples * self.num_replicas
            indices = indices[:total_size]
        else:
            num_samples = math.ceil(ds_len / self.num_replicas)
            total_size = num_samples * self.num_replicas
            padding_size = total_size - len(indices)
            indices += (indices * math.ceil(padding_size / max(1, len(indices))))[
                :padding_size
            ]
        subsampled = indices[self.rank : total_size : self.num_replicas]
        return iter(subsampled)

    def __len__(self) -> int:
        """Returns the number of samples assigned to each replica.

        Returns:
            Sample count per replica.
        """
        ds_len = len(self.dataset)
        if self.drop_last:
            return ds_len // self.num_replicas
        return math.ceil(ds_len / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler to ensure deterministic shuffling across epochs.

        Args:
            epoch: Epoch number.
        """
        self.epoch = epoch


class IterDataPipe:
    """Iterable-style DataPipe."""

    def __init__(
        self,
        functions: dict[str, Callable[[CollateValue], CollateValue]] | None = None,
        reduce_ex_hook: Callable[[int], CollateValue] | None = None,
        getstate_hook: Callable[[], CollateValue] | None = None,
        str_hook: Callable[[], str] | None = None,
        repr_hook: Callable[[], str] | None = None,
        _valid_iterator_id: int | None = None,
        _number_of_samples_yielded: int = 0,
        _snapshot_state: str = "'_SnapshotState'",
        _fast_forward_iterator: Iterator[CollateValue] | None = None,
    ) -> None:
        """Initializes the IterDataPipe.

        Args:
            functions: Dictionary of registered datapipe functions.
            reduce_ex_hook: Custom hook for `__reduce_ex__`.
            getstate_hook: Custom hook for `__getstate__`.
            str_hook: Custom hook for `__str__`.
            repr_hook: Custom hook for `__repr__`.
            _valid_iterator_id: Internal valid iterator identifier.
            _number_of_samples_yielded: Count of samples yielded by this pipe.
            _snapshot_state: Internal snapshot state string.
            _fast_forward_iterator: Optional fast-forward iterator.
        """
        _ = (
            reduce_ex_hook,
            getstate_hook,
            str_hook,
            repr_hook,
            _valid_iterator_id,
            _number_of_samples_yielded,
            _snapshot_state,
            _fast_forward_iterator,
        )
        self.functions: dict[str, Callable[[CollateValue], CollateValue]] = (
            functions if functions is not None else {}
        )


class MapDataPipe:
    """Map-style DataPipe."""

    def __init__(
        self,
        functions: dict[str, Callable[[CollateValue], CollateValue]] | None = None,
        reduce_ex_hook: Callable[[int], CollateValue] | None = None,
        getstate_hook: Callable[[], CollateValue] | None = None,
        str_hook: Callable[[], str] | None = None,
        repr_hook: Callable[[], str] | None = None,
    ) -> None:
        """Initializes the MapDataPipe.

        Args:
            functions: Dictionary of registered datapipe functions.
            reduce_ex_hook: Custom hook for `__reduce_ex__`.
            getstate_hook: Custom hook for `__getstate__`.
            str_hook: Custom hook for `__str__`.
            repr_hook: Custom hook for `__repr__`.
        """
        _ = (reduce_ex_hook, getstate_hook, str_hook, repr_hook)
        self.functions: dict[str, Callable[[CollateValue], CollateValue]] = (
            functions if functions is not None else {}
        )


class RandomSampler(Sampler[int]):
    """Samples elements randomly."""

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: Callable[[], float] | int | None = None,
    ) -> None:
        """Initializes the RandomSampler.

        Args:
            data_source: Dataset to sample from.
            replacement: Samples are drawn on-demand with replacement if True.
            num_samples: Number of samples to draw; defaults to `len(data_source)`.
            generator: Generator used in sampling.
        """
        super().__init__(data_source)
        self.data_source: Sized = data_source
        self.replacement: bool = replacement
        self.num_samples: int = (
            num_samples if num_samples is not None else len(data_source)
        )
        self.generator: Callable[[], float] | int | None = generator

    def __iter__(self) -> Iterator[int]:
        """Yields random sample indices.

        Yields:
            Random integer indices in `[0, len(data_source))`.
        """
        n = len(self.data_source)
        rng = _make_rng(self.generator)
        if self.replacement:
            for _ in range(self.num_samples):
                yield rng.randint(0, n - 1)
        else:
            indices = list(range(n))
            rng.shuffle(indices)
            yield from indices[: self.num_samples]

    def __len__(self) -> int:
        """Returns the number of samples drawn.

        Returns:
            Total sample count.
        """
        return self.num_samples


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Sized) -> None:
        """Initializes the SequentialSampler.

        Args:
            data_source: Dataset to sample from.
        """
        super().__init__(data_source)
        self.data_source: Sized = data_source

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over sequential indices `0..len(data_source)-1`.

        Returns:
            Sequential index iterator.
        """
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        """Returns the length of `data_source`.

        Returns:
            Number of elements in `data_source`.
        """
        return len(self.data_source)


class StackDataset(Dataset[CollateValue]):
    """Dataset as a stacking of multiple datasets."""

    def __init__(
        self,
        *args: Dataset[CollateValue],
        datasets: tuple[Dataset[CollateValue], ...]
        | dict[str, Dataset[CollateValue]]
        | None = None,
        **kwargs: Dataset[CollateValue],
    ) -> None:
        """Initializes the StackDataset.

        Args:
            *args: Positional datasets to stack.
            datasets: Explicit tuple or dict of datasets to stack.
            **kwargs: Keyword datasets to stack.
        """
        super().__init__()
        if datasets is None:
            if not args and not kwargs:
                raise ValueError("No datasets provided")
            self.datasets: (
                tuple[Dataset[CollateValue], ...] | dict[str, Dataset[CollateValue]]
            ) = args if args else kwargs
        else:
            self.datasets = datasets

    def __getitem__(
        self, idx: int
    ) -> tuple[CollateValue, ...] | dict[str, CollateValue]:
        """Fetches the stacked sample at `idx` across all constituent datasets.

        Args:
            idx: Sample index.

        Returns:
            Tuple or dict of samples at `idx`.
        """
        if isinstance(self.datasets, dict):
            return {k: ds[idx] for k, ds in self.datasets.items()}
        return tuple(ds[idx] for ds in self.datasets)

    def __len__(self) -> int:
        """Returns the length of the stacked datasets.

        Returns:
            Length of the first constituent dataset.
        """
        if isinstance(self.datasets, dict):
            first = next(iter(self.datasets.values()))
            return len(first)
        return len(self.datasets[0])


class Subset(Dataset[_T_co]):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        """Initializes the Subset dataset.

        Args:
            dataset: The whole Dataset.
            indices: Indices in the whole set selected for subset.
        """
        super().__init__()
        self.dataset: Dataset[_T_co] = dataset
        self.indices: Sequence[int] = indices

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches the item at subset index `idx`.

        Args:
            idx: Subset index.

        Returns:
            The item from the underlying dataset at `self.indices[idx]`.
        """
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        """Returns the number of elements in the subset.

        Returns:
            Length of `self.indices`.
        """
        return len(self.indices)


class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(
        self,
        indices: Sequence[int],
        generator: Callable[[], float] | int | None = None,
    ) -> None:
        """Initializes the SubsetRandomSampler.

        Args:
            indices: Sequence of indices to sample from.
            generator: Optional random generator used in sampling.
        """
        super().__init__()
        self.indices: Sequence[int] = indices
        self.generator: Callable[[], float] | int | None = generator

    def __iter__(self) -> Iterator[int]:
        """Iterates over the subset of indices in random order.

        Returns:
            Iterator over shuffled indices.
        """
        shuffled = list(self.indices)
        rng = _make_rng(self.generator)
        rng.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        """Returns the number of indices in the subset.

        Returns:
            Length of `self.indices`.
        """
        return len(self.indices)


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """Dataset wrapping tensors."""

    def __init__(self, *tensors: Tensor) -> None:
        """Initializes the TensorDataset.

        Args:
            *tensors: Tensors that have the same size of the first dimension.
        """
        super().__init__()
        if len(tensors) > 1:
            first_len = tensors[0].shape[0] if len(tensors[0].shape) > 0 else 0
            if any(
                (t.shape[0] if len(t.shape) > 0 else 0) != first_len
                for t in tensors[1:]
            ):
                raise ValueError("Size mismatch between tensors")
        self.tensors: tuple[Tensor, ...] = tensors

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Fetches a tuple of tensor slices at `index`.

        Args:
            index: Sample index along dimension 0.

        Returns:
            Tuple of tensor elements at `index`.
        """
        return tuple(t[index] if len(t.shape) > 0 else t for t in self.tensors)

    def __len__(self) -> int:
        """Returns the size of the first dimension of the tensors.

        Returns:
            Dataset length.
        """
        return len(self.tensors[0]) if len(self.tensors) > 0 else 0


class WeightedRandomSampler(Sampler[int]):
    """Samples elements from `[0, .., len(weights)-1]` with given probabilities."""

    def __init__(
        self,
        weights: Sequence[float] | Tensor,
        num_samples: int,
        replacement: bool = True,
        generator: Callable[[], float] | int | None = None,
    ) -> None:
        """Initializes the WeightedRandomSampler.

        Args:
            weights: Sequence or Tensor of weights (not necessarily summing to one).
            num_samples: Number of samples to draw.
            replacement: If True, samples are drawn with replacement.
            generator: Generator used in sampling.
        """
        super().__init__()
        self.weights: Sequence[float] | Tensor = weights
        self.num_samples: int = num_samples
        self.replacement: bool = replacement
        self.generator: Callable[[], float] | int | None = generator

    def __iter__(self) -> Iterator[int]:
        """Yields weighted random indices.

        Returns:
            Iterator over sampled indices.
        """
        if isinstance(self.weights, Tensor):
            raw_list = self.weights.tolist()
            w_list: list[float] = (
                [float(x) for x in raw_list if isinstance(x, (int, float))]
                if isinstance(raw_list, list)
                else [float(raw_list)]
                if isinstance(raw_list, (int, float))
                else []
            )
        else:
            w_list = [float(w) for w in self.weights]
        population = list(range(len(w_list)))
        rng = _make_rng(self.generator)
        if self.replacement:
            chosen = rng.choices(population, weights=w_list, k=self.num_samples)
            return iter(chosen)
        selected: list[int] = []
        avail_pop = list(population)
        avail_w = list(w_list)
        for _ in range(min(self.num_samples, len(avail_pop))):
            pick = rng.choices(avail_pop, weights=avail_w, k=1)[0]
            pos = avail_pop.index(pick)
            selected.append(pick)
            avail_pop.pop(pos)
            avail_w.pop(pos)
        return iter(selected)

    def __len__(self) -> int:
        """Returns the number of samples drawn.

        Returns:
            Total sample count `self.num_samples`.
        """
        return self.num_samples


def random_split(
    dataset: Dataset[_T_co],
    lengths: Sequence[int | float],
    generator: Callable[[], float] | int | None = None,
) -> list[Subset[_T_co]]:
    """Randomly splits a dataset into non-overlapping new datasets of given lengths.

    Supports both integer counts (summing to `len(dataset)`) and fractional proportions
    (summing to `1.0`).

    Args:
        dataset: Dataset to be split.
        lengths: Lengths or fractions of splits to be produced.
        generator: Generator used for the random permutation.

    Returns:
        List of `Subset` datasets.
    """
    total = len(dataset)
    if any(isinstance(x, float) for x in lengths) and math.isclose(
        sum(float(x) for x in lengths), 1.0, rel_tol=1e-5
    ):
        int_lengths = [math.floor(float(x) * total) for x in lengths]
        remainder = total - sum(int_lengths)
        for i in range(remainder):
            int_lengths[i % len(int_lengths)] += 1
    else:
        int_lengths = [int(x) for x in lengths]
        if sum(int_lengths) != total:
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!"
            )

    indices = list(range(total))
    rng = _make_rng(generator)
    rng.shuffle(indices)
    subsets: list[Subset[_T_co]] = []
    offset = 0
    for length in int_lengths:
        subsets.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return subsets


def functional_datapipe(name: str, enable_df_api_tracing: bool = False) -> None:
    """Registers a function as a functional DataPipe.

    Args:
        name: The name of the datapipe.
        enable_df_api_tracing: Whether to enable DataFrame API tracing.
    """
    _ = (name, enable_df_api_tracing)


def guaranteed_datapipes_determinism(prev: bool) -> None:
    """Context helper to guarantee determinism in DataPipes.

    Args:
        prev: Previous determinism flag state.
    """
    _ = prev


def non_deterministic(
    cls: type[IterDataPipe] | None = None,
    *,
    deterministic_fn: Callable[[], bool],
    arg: type[IterDataPipe] | Callable[[], bool],
) -> None:
    """Marks an IterDataPipe class as non-deterministic.

    Args:
        cls: Optional IterDataPipe class to mark.
        deterministic_fn: Callback returning whether determinism is required.
        arg: Target class or callback argument.
    """
    _ = (cls, deterministic_fn, arg)


def runtime_validation_disabled(prev: bool) -> None:
    """Context helper to disable DataPipe runtime validation.

    Args:
        prev: Previous validation state flag.
    """
    _ = prev
