from typing import (
    Any,
    Optional,
    Callable,
    Iterable,
    Sequence,
    Sized,
    Iterator,
    Generator,
    TypeVar,
)
import torch

_R = Any
_FanMode = Any
_NonlinearityType = Any
_Optional = Optional
_empty = Any
ParamsT = Any
_size_1_t = Any
_size_2_t = Any
_size_3_t = Any
_size_any_t = Any
_size_2_opt_t = Any
_size_3_opt_t = Any
_ratio_2_t = Any
_ratio_3_t = Any
optional = Any
UninitializedParameter = Any
Parameter = Any
real = Any
NamedShape = Any
_ratio_any_t = Any
_size_any_opt_t = Any
_size = Any
iterable = Iterable
NoneType = type(None)
_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")
_collate_fn_t = Any
_worker_init_fn_t = Any
_SnapshotState = Any

class BatchSampler:
    def __init__(
        self,
        sampler: Sampler | Iterable = ...,
        batch_size: int = ...,
        drop_last: bool = ...,
    ) -> NoneType: ...

class ChainDataset:
    def __init__(self, datasets: Iterable[IterableDataset] = ...) -> NoneType: ...

class ConcatDataset:
    def __init__(
        self, datasets: list[Dataset[_T_co]] = ..., cumulative_sizes: list[int] = ...
    ) -> NoneType: ...

class DFIterDataPipe:
    def __init__(self, *args, **kwargs) -> Any: ...

class DataChunk:
    def __init__(self, items: Iterable[_T] = ...) -> NoneType: ...

class DataLoader:
    def __init__(
        self,
        dataset: Dataset[_T_co] = ...,
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
    ) -> NoneType: ...

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
        _snapshot_state: _SnapshotState = "_SnapshotState",
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
    ) -> NoneType: ...

class Sampler:
    def __init__(self, data_source: Dataset = "```(None)```") -> NoneType: ...

class SequentialSampler:
    def __init__(self, data_source: Sized = ...) -> NoneType: ...

class StackDataset:
    def __init__(self, datasets: tuple | dict = ..., *args, **kwargs) -> NoneType: ...

class Subset:
    def __init__(
        self, dataset: Dataset[_T_co] = ..., indices: Sequence[int] = ...
    ) -> NoneType: ...

class SubsetRandomSampler:
    def __init__(
        self, indices: Sequence[int] = ..., generator: Generator = "```(None)```"
    ) -> NoneType: ...

class TensorDataset:
    def __init__(self, *tensors) -> NoneType: ...

class WeightedRandomSampler:
    def __init__(
        self,
        weights: torch.Tensor = ...,
        num_samples: int = ...,
        replacement: bool = True,
        generator: Generator = "```(None)```",
    ) -> NoneType: ...

class functional_datapipe:
    def __init__(
        self, name: str = ..., enable_df_api_tracing: bool = False
    ) -> NoneType: ...

class guaranteed_datapipes_determinism:
    def __init__(self, prev: bool = ...) -> NoneType: ...

class non_deterministic:
    def __init__(
        self,
        cls: type[IterDataPipe] | None = "```(None)```",
        deterministic_fn: Callable[[], bool] = ...,
        arg: type[IterDataPipe] | Callable[[], bool] = ...,
    ) -> NoneType: ...

class runtime_validation_disabled:
    def __init__(self, prev: bool = ...) -> NoneType: ...
