"""Data utilities."""

import ml_switcheroo_compiler as ml_switcheroo
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
        """Initializes the IterableDataset.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass


class ChainDataset(IterableDataset):
    """Dataset for chaining multiple IterableDatasets."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        """Initializes the ChainDataset.

        Args:
            datasets (Iterable[Dataset]): datasets to be chained together.
        """
        pass


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets."""

    def __init__(
        self, datasets: List[Dataset[_T_co]], cumulative_sizes: List[int]
    ) -> None:
        """Initializes the ConcatDataset.

        Args:
            datasets (List[Dataset[_T_co]]): list of datasets to be concatenated.
            cumulative_sizes (List[int]): list of cumulative sizes of the datasets.
        """
        pass


class DFIterDataPipe:
    """Iterable-style DataPipe."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the DataPipe.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass


class DataChunk:
    """Built-in mutable sequence."""

    def __init__(self, items: Iterable[Any]) -> None:
        """Initializes the DataChunk.

        Args:
            items (Iterable[Any]): Iterable of items to populate the chunk.
        """
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
        """Initializes DistributedSampler.

        Args:
            dataset (Dataset): Dataset used for sampling.
            num_replicas (Optional[int], optional): Number of processes participating in distributed training. Defaults to None.
            rank (Optional[int], optional): Rank of the current process within num_replicas. Defaults to None.
            shuffle (Optional[bool], optional): If True, sampler will shuffle the indices. Defaults to True.
            seed (Optional[int], optional): Random seed. Defaults to 0.
            drop_last (Optional[bool], optional): If True, then the sampler will drop the tail of the data. Defaults to False.
        """
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
        """Initializes the IterDataPipe.

        Args:
            functions (Dict[str, Callable], optional): Functions dictionary. Defaults to {}.
            reduce_ex_hook (Optional[Callable], optional): Hook for reduce_ex. Defaults to None.
            getstate_hook (Optional[Callable], optional): Hook for getstate. Defaults to None.
            str_hook (Optional[Callable], optional): Hook for str. Defaults to None.
            repr_hook (Optional[Callable], optional): Hook for repr. Defaults to None.
            _valid_iterator_id (Optional[int], optional): Internal iterator ID. Defaults to None.
            _number_of_samples_yielded (int, optional): Internal sample count. Defaults to 0.
            _snapshot_state (Any, optional): Internal snapshot state. Defaults to "'_SnapshotState'".
            _fast_forward_iterator (Optional[Iterator], optional): Internal fast forward iterator. Defaults to None.
        """
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
        """Initializes the MapDataPipe.

        Args:
            functions (Dict[str, Callable], optional): Functions dictionary. Defaults to {}.
            reduce_ex_hook (Optional[Callable], optional): Hook for reduce_ex. Defaults to None.
            getstate_hook (Optional[Callable], optional): Hook for getstate. Defaults to None.
            str_hook (Optional[Callable], optional): Hook for str. Defaults to None.
            repr_hook (Optional[Callable], optional): Hook for repr. Defaults to None.
        """
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
        """Initializes the RandomSampler.

        Args:
            data_source (Sized): Dataset to sample from.
            replacement (bool, optional): Samples are drawn on-demand with replacement if True. Defaults to False.
            num_samples (Optional[int], optional): Number of samples to draw. Defaults to None.
            generator (Generator, optional): Generator used in sampling. Defaults to None.
        """
        pass


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Sized) -> None:
        """Initializes the SequentialSampler.

        Args:
            data_source (Sized): Dataset to sample from.
        """
        pass


class StackDataset(Dataset):
    """Dataset as a stacking of multiple datasets."""

    def __init__(
        self,
        *args: Dataset,
        datasets: Optional[Union[Tuple, Dict]] = None,
        **kwargs: Dataset,
    ) -> None:
        """Initializes the StackDataset.

        Args:
            *args: Datasets to stack.
            datasets (Optional[Union[Tuple, Dict]], optional): Tuple or dict of datasets. Defaults to None.
            **kwargs: Additional datasets to stack.
        """
        pass


class Subset(Dataset[_T_co]):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        """Initializes the Subset dataset.

        Args:
            dataset (Dataset[_T_co]): The whole Dataset.
            indices (Sequence[int]): Indices in the whole set selected for subset.
        """
        pass

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches the item from the subset.

        Args:
            idx (int): The index.

        Returns:
            _T_co: The subset item.
        """
        pass

    def __len__(self) -> int:
        """Returns the length of the subset.

        Returns:
            int: The length of the subset.
        """
        pass


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(self, indices: Sequence[int], generator: "Generator" = None) -> None:
        """Initializes the SubsetRandomSampler.

        Args:
            indices (Sequence[int]): A sequence of indices.
            generator (Generator, optional): Generator used in sampling. Defaults to None.
        """
        pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, *tensors: "tuple[Tensor, ...]") -> None:
        """Initializes the TensorDataset.

        Args:
            *tensors: Tensors that have the same size of the first dimension.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        """Fetches the item from the tensor dataset.

        Args:
            index (int): The index.

        Returns:
            Tuple[Tensor, ...]: A tuple of tensors.
        """
        pass

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
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
        """Initializes the WeightedRandomSampler.

        Args:
            weights (Tensor): A sequence of weights.
            num_samples (int): Number of samples to draw.
            replacement (bool, optional): If True, samples are drawn with replacement. Defaults to True.
            generator (Generator, optional): Generator used in sampling. Defaults to None.
        """
        pass


def functional_datapipe(name: str, enable_df_api_tracing: bool = False) -> None:
    """Decorator to register a function as a datapipe.

    Args:
        name (str): The name of the datapipe.
        enable_df_api_tracing (bool, optional): Enables tracing. Defaults to False.
    """
    pass


def guaranteed_datapipes_determinism(prev: bool) -> None:
    """Context manager to guarantee determinism in datapipes.

    Args:
        prev (bool): Previous state.
    """
    pass


def non_deterministic(
    cls: "type[IterDataPipe] | None" = None,
    *,
    deterministic_fn: "Callable[[], bool]",
    arg: "type[IterDataPipe] | Callable[[], bool]",
) -> None:
    """Marks a datapipe as non-deterministic.

    Args:
        cls (type[IterDataPipe] | None, optional): Datapipe class. Defaults to None.
        deterministic_fn (Callable[[], bool]): Function returning boolean.
        arg (type[IterDataPipe] | Callable[[], bool]): Target argument.
    """
    pass


def runtime_validation_disabled(prev: bool) -> None:
    """Context manager to disable runtime validation.

    Args:
        prev (bool): Previous state.
    """
    pass
