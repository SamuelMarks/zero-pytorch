"DataLoader System."

from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
)

_T_co = TypeVar("_T_co", covariant=True)
_collate_fn_t = Callable[[List[Any]], Any]
_worker_init_fn_t = Callable[[int], None]


class Dataset(Generic[_T_co]):
    """An abstract class representing a Dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the dataset.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        pass

    def __getitem__(self, idx: int) -> _T_co:
        """Fetches a data sample for a given key.

        Args:
            idx (int): The index of the item.

        Returns:
            _T_co: The data sample.
        """
        pass


class Sampler(Generic[_T_co]):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method.
    """

    def __init__(self, data_source: "Dataset" = None) -> None:
        """Initializes the sampler.

        Args:
            data_source (Dataset, optional): Dataset to sample from. Defaults to None.
        """
        pass


class BatchSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(
        self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool
    ) -> None:
        """Initializes the BatchSampler.

        Args:
            sampler (Union[Sampler, Iterable]): Base sampler.
            batch_size (int): Size of mini-batch.
            drop_last (bool): If True, the sampler will drop the last batch if its size would be less than batch_size.
        """
        pass

    def __iter__(self) -> Iterator[List[int]]:
        """Iterates over batches of indices.

        Yields:
            Iterator[List[int]]: An iterator containing lists of indices.
        """
        pass

    def __len__(self) -> int:
        """Returns the number of batches.

        Returns:
            int: The number of batches.
        """
        pass


def default_collate(batch: List[Any]) -> Any:
    """Puts each data field into a tensor with outer dimension batch size.

    Args:
        batch (List[Any]): A list of data samples.

    Returns:
        Any: Collated data batch.
    """
    pass


class DataLoader(Generic[_T_co]):
    """Data loader combines a dataset and a sampler, and provides an iterable over the given dataset."""

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        """Initializes the DataLoader.

        Args:
            dataset (Dataset[_T_co]): Dataset from which to load the data.
            batch_size (Optional[int], optional): How many samples per batch to load. Defaults to 1.
            shuffle (Optional[bool], optional): Set to True to have the data reshuffled at every epoch. Defaults to None.
            sampler (Union[Sampler, Iterable, None], optional): Defines the strategy to draw samples from the dataset. Defaults to None.
            batch_sampler (Union[Sampler[List], Iterable[List], None], optional): Like sampler, but returns a batch of indices at a time. Defaults to None.
            num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 0.
            collate_fn (Optional[_collate_fn_t], optional): Merges a list of samples to form a mini-batch of Tensor(s). Defaults to None.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into device pinned memory. Defaults to False.
            drop_last (bool, optional): Set to True to drop the last incomplete batch. Defaults to False.
            timeout (float, optional): If positive, the timeout value for collecting a batch from workers. Defaults to 0.
            worker_init_fn (Optional[_worker_init_fn_t], optional): If not None, this will be called on each worker subprocess with the worker id. Defaults to None.
            multiprocessing_context (Any, optional): Context for multiprocessing. Defaults to None.
            generator (Any, optional): Random number generator. Defaults to None.
            prefetch_factor (Optional[int], optional): Number of batches loaded in advance by each worker. Defaults to None.
            persistent_workers (bool, optional): If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to False.
            pin_memory_device (str, optional): The device to pin memory to if pin_memory is True. Defaults to "".
            in_order (bool, optional): Yield samples in order. Defaults to True.
        """
        pass

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator for the data loader.

        Returns:
            Iterator[Any]: An iterator that yields batches of data.
        """
        pass
