import ml_switcheroo

"Data utilities."
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
    TypeVar,
)
from zero_torch.tensor import Tensor
from .dataloader import DataLoader, Dataset, BatchSampler, Sampler

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
    "runtime_validation_disabled",
]
_T_co = TypeVar("_T_co", covariant=True)


class IterableDataset(Dataset[_T_co]):
    """An iterable Dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ChainDataset(IterableDataset):
    """Dataset for chaining multiple IterableDatasets."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        pass


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets."""

    def __init__(
        self, datasets: List[Dataset[_T_co]], cumulative_sizes: List[int]
    ) -> None:
        pass


class DFIterDataPipe:
    """Iterable-style DataPipe."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class DataChunk:
    """Built-in mutable sequence."""

    def __init__(self, items: Iterable[Any]) -> None:
        pass


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
    ) -> None:
        pass


class IterDataPipe:
    """Iterable-style DataPipe."""

    def __init__(
        self,
        functions: Dict[str, Callable] = {},
        reduce_ex_hook: Optional[Callable] = None,
        getstate_hook: Optional[Callable] = None,
        str_hook: Optional[Callable] = None,
        repr_hook: Optional[Callable] = None,
        _valid_iterator_id: Optional[int] = None,
        _number_of_samples_yielded: int = 0,
        _snapshot_state: Any = "'_SnapshotState'",
        _fast_forward_iterator: Optional[Iterator] = None,
    ) -> None:
        pass


class MapDataPipe:
    """Map-style DataPipe."""

    def __init__(
        self,
        functions: Dict[str, Callable] = {},
        reduce_ex_hook: Optional[Callable] = None,
        getstate_hook: Optional[Callable] = None,
        str_hook: Optional[Callable] = None,
        repr_hook: Optional[Callable] = None,
    ) -> None:
        pass


class RandomSampler(Sampler):
    """Samples elements randomly."""

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: "Generator" = None,
    ) -> None:
        pass


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Sized) -> None:
        pass


class StackDataset(Dataset):
    """Dataset as a stacking of multiple datasets."""

    def __init__(
        self,
        *args: Dataset,
        datasets: Optional[Union[Tuple, Dict]] = None,
        **kwargs: Dataset,
    ) -> None:
        pass


class Subset(Dataset[_T_co]):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        pass

    def __getitem__(self, idx: int) -> _T_co:
        pass

    def __len__(self) -> int:
        pass


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(self, indices: Sequence[int], generator: "Generator" = None) -> None:
        pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, *tensors: "tuple[Tensor, ...]") -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        pass

    def __len__(self) -> int:
        pass


class WeightedRandomSampler(Sampler):
    """Samples elements from [0,..,len(weights)-1] with given probabilities."""

    def __init__(
        self,
        weights: Tensor,
        num_samples: int,
        replacement: bool = True,
        generator: "Generator" = None,
    ) -> None:
        pass


def functional_datapipe(name: str, enable_df_api_tracing: bool = False) -> None:
    pass


def guaranteed_datapipes_determinism(prev: bool) -> None:
    pass


def non_deterministic(
    cls: "type[IterDataPipe] | None" = None,
    *,
    deterministic_fn: "Callable[[], bool]",
    arg: "type[IterDataPipe] | Callable[[], bool]",
) -> None:
    pass


def runtime_validation_disabled(prev: bool) -> None:
    pass
