from collections.abc import Generator, Iterable, Iterator, Sequence, Sized
from typing import (
    Any,
    Callable,
    TypeVar,
)

import torch

_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")

class BatchSampler:
    def __init__(
        self,
        sampler: Sampler | Iterable = ...,
        batch_size: int = ...,
        drop_last: bool = ...,
    ) -> None: ...

class ChainDataset:
    def __init__(self, datasets: Iterable[IterableDataset] = ...) -> None: ...

class ConcatDataset:
    def __init__(
        self, datasets: list[Dataset[_T_co]] = ..., cumulative_sizes: list[int] = ...
    ) -> None: ...

class DFIterDataPipe:
    def __init__(self, *args, **kwargs) -> Any: ...

class DataChunk:
    def __init__(self, items: Iterable[_T] = ...) -> None: ...

class DataLoader:
    def __init__(
        self,
        dataset: Dataset[_T_co] = ...,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        num_workers: int = 0,
        collate_fn: Any | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Any | None = None,
        multiprocessing_context: Any = None,
        generator: Any = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> Any: ...

class Dataset:
    def __init__(self, *args, **kwargs) -> Any: ...

class DistributedSampler:
    def __init__(
        self,
        dataset: Dataset = ...,
        num_replicas: int | None = "```(None)```",
        rank: int | None = "```(None)```",
        shuffle: bool | None = True,
        seed: int | None = 0,
        drop_last: bool | None = False,
    ) -> None: ...

class IterDataPipe:
    def __init__(
        self,
        functions: dict[str, Callable] = {},
        reduce_ex_hook: Callable | None = "```(None)```",
        getstate_hook: Callable | None = "```(None)```",
        str_hook: Callable | None = "```(None)```",
        repr_hook: Callable | None = "```(None)```",
        _valid_iterator_id: int | None = "```(None)```",
        _number_of_samples_yielded: int = 0,
        _snapshot_state: Any = "Any",
        _fast_forward_iterator: Iterator | None = "```(None)```",
    ) -> Any: ...

class IterableDataset:
    def __init__(self, *args, **kwargs) -> Any: ...

class MapDataPipe:
    def __init__(
        self,
        functions: dict[str, Callable] = {},
        reduce_ex_hook: Callable | None = "```(None)```",
        getstate_hook: Callable | None = "```(None)```",
        str_hook: Callable | None = "```(None)```",
        repr_hook: Callable | None = "```(None)```",
    ) -> Any: ...

class RandomSampler:
    def __init__(
        self,
        data_source: Sized = ...,
        replacement: bool = False,
        num_samples: int | None = "```(None)```",
        generator: Generator = "```(None)```",
    ) -> None: ...

class Sampler:
    def __init__(self, data_source: Dataset = "```(None)```") -> None: ...

class SequentialSampler:
    def __init__(self, data_source: Sized = ...) -> None: ...

class StackDataset:
    def __init__(self, datasets: tuple | dict = ..., *args, **kwargs) -> None: ...

class Subset:
    def __init__(
        self, dataset: Dataset[_T_co] = ..., indices: Sequence[int] = ...
    ) -> None: ...

class SubsetRandomSampler:
    def __init__(
        self, indices: Sequence[int] = ..., generator: Generator = "```(None)```"
    ) -> None: ...

class TensorDataset:
    def __init__(self, *tensors) -> None: ...

class WeightedRandomSampler:
    def __init__(
        self,
        weights: torch.Tensor = ...,
        num_samples: int = ...,
        replacement: bool = True,
        generator: Generator = "```(None)```",
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
        cls: type[IterDataPipe] | None = "```(None)```",
        deterministic_fn: Callable[[], bool] = ...,
        arg: type[IterDataPipe] | Callable[[], bool] = ...,
    ) -> None: ...

class runtime_validation_disabled:
    def __init__(self, prev: bool = ...) -> None: ...
