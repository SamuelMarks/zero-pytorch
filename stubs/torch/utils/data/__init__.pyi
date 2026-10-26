"""Type stubs for torch.utils.data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from typing import Generic, TypeVar

import torch
from torch import Tensor

_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")

class Dataset(Generic[_T_co]):
    def __init__(
        self, *args: Tensor | int | str, **kwargs: Tensor | int | str
    ) -> None: ...

class IterableDataset(Dataset[_T_co]):
    def __init__(
        self, *args: Tensor | int | str, **kwargs: Tensor | int | str
    ) -> None: ...

class Sampler(Generic[_T_co]):
    def __init__(self, data_source: Sized | None = None) -> None: ...

class BatchSampler:
    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int] = ...,
        batch_size: int = ...,
        drop_last: bool = ...,
    ) -> None: ...

class ChainDataset:
    def __init__(self, datasets: Iterable[IterableDataset[_T_co]] = ...) -> None: ...

class ConcatDataset(Dataset[_T_co]):
    def __init__(
        self,
        datasets: Sequence[Dataset[_T_co]] = ...,
        cumulative_sizes: list[int] | None = None,
    ) -> None: ...

class DFIterDataPipe:
    def __init__(
        self, *args: Tensor | int | str, **kwargs: Tensor | int | str
    ) -> None: ...

class DataChunk(Generic[_T]):
    def __init__(self, items: Iterable[_T] = ...) -> None: ...

class DataLoader(Generic[_T_co]):
    def __init__(
        self,
        dataset: Dataset[_T_co] = ...,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler[int] | Iterable[int] | None = None,
        batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[_T_co]], _T_co] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context: str | None = None,
        generator: torch.Generator | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None: ...

class DistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset[_T_co] = ...,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None: ...

class IterDataPipe(IterableDataset[_T_co]):
    def __init__(
        self,
        functions: dict[str, Callable[..., _T_co]] | None = None,
        reduce_ex_hook: Callable[..., tuple[type, tuple[int, ...]]] | None = None,
        getstate_hook: Callable[..., dict[str, int]] | None = None,
        str_hook: Callable[..., str] | None = None,
        repr_hook: Callable[..., str] | None = None,
        _valid_iterator_id: int | None = None,
        _number_of_samples_yielded: int = 0,
        _snapshot_state: str = "NotStarted",
        _fast_forward_iterator: Iterator[_T_co] | None = None,
    ) -> None: ...

class MapDataPipe(Dataset[_T_co]):
    def __init__(
        self,
        functions: dict[str, Callable[..., _T_co]] | None = None,
        reduce_ex_hook: Callable[..., tuple[type, tuple[int, ...]]] | None = None,
        getstate_hook: Callable[..., dict[str, int]] | None = None,
        str_hook: Callable[..., str] | None = None,
        repr_hook: Callable[..., str] | None = None,
    ) -> None: ...

class RandomSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized = ...,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None: ...

class SequentialSampler(Sampler[int]):
    def __init__(self, data_source: Sized = ...) -> None: ...

class StackDataset(Dataset[_T_co]):
    def __init__(
        self,
        *args: Dataset[_T_co],
        **kwargs: Dataset[_T_co],
    ) -> None: ...

class Subset(Dataset[_T_co]):
    def __init__(
        self, dataset: Dataset[_T_co] = ..., indices: Sequence[int] = ...
    ) -> None: ...

class SubsetRandomSampler(Sampler[int]):
    def __init__(
        self,
        indices: Sequence[int] = ...,
        generator: torch.Generator | None = None,
    ) -> None: ...

class TensorDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, *tensors: Tensor) -> None: ...

class WeightedRandomSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float] | Tensor = ...,
        num_samples: int = ...,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None: ...

class functional_datapipe:
    def __init__(
        self, name: str = ..., enable_df_api_tracing: bool = False
    ) -> None: ...

class guaranteed_datapipes_determinism:
    def __init__(self, prev: bool = ...) -> None: ...

class non_deterministic:
    def __init__(
        self,
        cls: type[IterDataPipe[Tensor]] | None = None,
        deterministic_fn: Callable[[], bool] = ...,
        arg: type[IterDataPipe[Tensor]] | Callable[[], bool] | None = None,
    ) -> None: ...

class runtime_validation_disabled:
    def __init__(self, prev: bool = ...) -> None: ...
