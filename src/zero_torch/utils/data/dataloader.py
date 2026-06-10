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
        """Function."""
        pass

    def __len__(self) -> int:
        """Function."""
        pass

    def __getitem__(self, idx: int) -> _T_co:
        """Function."""
        pass


class Sampler(Generic[_T_co]):
    """Base class for all Samplers."""

    def __init__(self, data_source: "Dataset" = None) -> None:
        """Function."""
        pass


class BatchSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(
        self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool
    ) -> None:
        """Function."""
        pass

    def __iter__(self) -> Iterator[List[int]]:
        """Function."""
        pass

    def __len__(self) -> int:
        """Function."""
        pass


def default_collate(batch: List[Any]) -> Any:
    """Function."""
    pass


class DataLoader(Generic[_T_co]):
    """Data loader combines a dataset and a sampler."""

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
        """Function."""
        pass

    def __iter__(self) -> Iterator[Any]:
        """Function."""
        pass
