"""Distributed communication package (`zero_torch.distributed`)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import timedelta

from zero_torch._types import DeviceLike, OpArg
from zero_torch.tensor import Tensor


class ReduceOp:
    """Enumeration of distributed collective reduction operations."""

    SUM: str = "SUM"
    PRODUCT: str = "PRODUCT"
    MIN: str = "MIN"
    MAX: str = "MAX"
    BAND: str = "BAND"
    BOR: str = "BOR"
    BXOR: str = "BXOR"
    AVG: str = "AVG"


class DebugLevel:
    """Enumeration of distributed debug levels."""

    OFF: str = "OFF"
    INFO: str = "INFO"
    DETAIL: str = "DETAIL"


class ProcessGroup:
    """Process group handle for single-rank or simulated multi-rank collectives."""

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        backend: str = "gloo",
        ranks: Sequence[int] | None = None,
    ) -> None:
        """Initializes a `ProcessGroup`.

        Args:
            rank: Rank of the current process in the group.
            world_size: Total number of processes in the group.
            backend: Communication backend name.
            ranks: Global ranks belonging to this group.
        """
        self.rank: int = rank
        self.world_size: int = world_size
        self.backend: str = backend
        self.ranks: list[int] = (
            list(ranks) if ranks is not None else list(range(world_size))
        )


class Work:
    """Asynchronous collective work handle."""

    def __init__(self, completed: bool = True) -> None:
        """Initializes a `Work` handle.

        Args:
            completed: Whether the work operation has already completed.
        """
        self._completed: bool = completed

    def is_completed(self) -> bool:
        """Returns True if the asynchronous operation has completed.

        Returns:
            Completion flag.
        """
        return self._completed

    def wait(self, timeout: timedelta | None = None) -> bool:
        """Waits for the asynchronous operation to complete.

        Args:
            timeout: Optional timeout duration.

        Returns:
            True once completed.
        """
        _ = timeout
        self._completed = True
        return True


class DeviceMesh:
    """Logical multidimensional device mesh for distributed tensor parallelism."""

    def __init__(
        self,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...] | None = None,
    ) -> None:
        """Initializes a `DeviceMesh`.

        Args:
            device_type: Device type string (e.g. `'cpu'`).
            mesh_shape: Shape tuple of the device mesh.
            mesh_dim_names: Optional names for each mesh dimension.
        """
        self.device_type: str = device_type
        self.mesh_shape: tuple[int, ...] = mesh_shape
        self.mesh_dim_names: tuple[str, ...] | None = mesh_dim_names


_DEFAULT_PG: ProcessGroup | None = None
_DEBUG_LEVEL: str = DebugLevel.OFF
_RENDEZVOUS_HANDLERS: dict[str, Callable[[], None]] = {}


def GroupName(x: str, /) -> str:
    """Typed constructor for a distributed process group name.

    Args:
        x: Group name string.

    Returns:
        Group name string `x`.
    """
    return x


def is_available() -> bool:
    """Returns True if the distributed package is available.

    Returns:
        True.
    """
    return True


def is_initialized() -> bool:
    """Checks if the default process group has been initialized.

    Returns:
        True if `init_process_group` has been called and not destroyed.
    """
    return _DEFAULT_PG is not None


def is_backend_available(backend: str) -> bool:
    """Checks whether the specified communication backend is available.

    Args:
        backend: Backend name (`'gloo'`, `'nccl'`, `'mpi'`, `'ucc'`, `'xccl'`).

    Returns:
        True if backend is supported (`'gloo'`).
    """
    return backend.lower() == "gloo"


def is_gloo_available() -> bool:
    """Checks if the Gloo backend is available.

    Returns:
        True.
    """
    return True


def is_mpi_available() -> bool:
    """Checks if the MPI backend is available.

    Returns:
        False in pure Python mode.
    """
    return False


def is_nccl_available() -> bool:
    """Checks if the NCCL backend is available.

    Returns:
        False in pure Python mode.
    """
    return False


def is_ucc_available() -> bool:
    """Checks if the UCC backend is available.

    Returns:
        False in pure Python mode.
    """
    return False


def is_xccl_available() -> bool:
    """Checks if the XCCL backend is available.

    Returns:
        False in pure Python mode.
    """
    return False


def is_torchelastic_launched() -> bool:
    """Checks whether this process was launched with `torch.distributed.elastic`.

    Returns:
        False.
    """
    return False


def init_process_group(
    backend: str | None = None,
    init_method: str | None = None,
    timeout: timedelta | str | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: OpArg = None,
    group_name: str = "",
    pg_options: OpArg = None,
    device_id: DeviceLike = None,
    _ranks: Sequence[int] | None = None,
    enable_reconfigure: bool = False,
) -> None:
    """Initializes the default distributed process group.

    Args:
        backend: The backend to use (`'gloo'`, `'nccl'`, etc.).
        init_method: URL specifying how to initialize the process group.
        timeout: Timeout for operations executed against the process group.
        world_size: Number of processes participating in the job.
        rank: Rank of the current process.
        store: Key/value store accessible to all workers.
        group_name: Group name.
        pg_options: Process group options.
        device_id: Optional bound device ID.
        _ranks: Optional rank list.
        enable_reconfigure: Whether reconfiguration is enabled.
    """
    global _DEFAULT_PG
    _ = (
        init_method,
        timeout,
        store,
        group_name,
        pg_options,
        device_id,
        enable_reconfigure,
    )
    eff_backend = backend if backend is not None else "gloo"
    eff_ws = max(1, world_size)
    eff_rank = max(0, rank)
    _DEFAULT_PG = ProcessGroup(
        rank=eff_rank, world_size=eff_ws, backend=eff_backend, ranks=_ranks
    )


def destroy_process_group(group: ProcessGroup | None = None) -> None:
    """Destroys a given process group, and deinitializes the distributed package if default.

    Args:
        group: Process group to destroy, or `None` for the default process group.
    """
    global _DEFAULT_PG
    if group is None or group is _DEFAULT_PG:
        _DEFAULT_PG = None


def get_rank(group: ProcessGroup | None = None) -> int:
    """Returns the rank of the current process in the provided `group`.

    Args:
        group: ProcessGroup to query, or `None` for default group.

    Returns:
        Process rank (`0` if single-process).
    """
    pg = group if group is not None else _DEFAULT_PG
    return pg.rank if pg is not None else 0


def get_world_size(group: ProcessGroup | None = None) -> int:
    """Returns the number of processes in the current process group.

    Args:
        group: ProcessGroup to query, or `None` for default group.

    Returns:
        World size (`1` if single-process).
    """
    pg = group if group is not None else _DEFAULT_PG
    return pg.world_size if pg is not None else 1


def get_backend(group: ProcessGroup | None = None) -> str:
    """Returns the backend of the given process group.

    Args:
        group: ProcessGroup to query, or `None` for default group.

    Returns:
        Backend name string.
    """
    pg = group if group is not None else _DEFAULT_PG
    return pg.backend if pg is not None else "gloo"


def get_backend_config(group: ProcessGroup | None = None) -> str:
    """Returns the backend configuration string of the given process group.

    Args:
        group: ProcessGroup to query, or `None` for default group.

    Returns:
        Backend configuration string.
    """
    return get_backend(group)


def get_backend_impl(
    group: ProcessGroup | str | None = None,
    device: DeviceLike = None,
) -> str:
    """Returns the backend implementation identifier for `group` and `device`.

    Args:
        group: ProcessGroup or group name.
        device: Optional target device.

    Returns:
        Backend implementation identifier.
    """
    _ = device
    if isinstance(group, str):
        return group
    return get_backend(group)


def get_default_backend_for_device(device: DeviceLike) -> str:
    """Returns the default backend for the given device.

    Args:
        device: Device string or device object.

    Returns:
        Default backend name (`'gloo'` for CPU, `'nccl'` for CUDA).
    """
    if isinstance(device, str) and "cuda" in device.lower():
        return "nccl"
    return "gloo"


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """Translates a group rank into a global rank.

    Args:
        group: ProcessGroup to query.
        group_rank: Rank within `group`.

    Returns:
        Global rank integer.
    """
    if 0 <= group_rank < len(group.ranks):
        return group.ranks[group_rank]
    return group_rank


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """Translates a global rank into a group rank.

    Args:
        group: ProcessGroup to query.
        global_rank: Global rank integer.

    Returns:
        Rank within `group`.
    """
    if global_rank in group.ranks:
        return group.ranks.index(global_rank)
    return global_rank


def get_node_local_rank(fallback_rank: int | None = None) -> int:
    """Returns the local rank of the current process relative to the node.

    Args:
        fallback_rank: Rank to return if local rank is not set in the environment.

    Returns:
        Node-local rank integer.
    """
    return fallback_rank if fallback_rank is not None else 0


def get_pg_count() -> int:
    """Returns the number of initialized process groups.

    Returns:
        Count of process groups.
    """
    return 1 if _DEFAULT_PG is not None else 0


def get_process_group_ranks(group: ProcessGroup | None = None) -> list[int]:
    """Gets all ranks associated with `group`.

    Args:
        group: ProcessGroup to query, or `None` for default group.

    Returns:
        List of global rank integers.
    """
    pg = group if group is not None else _DEFAULT_PG
    return list(pg.ranks) if pg is not None else [0]


def get_debug_level() -> str:
    """Returns the current distributed debug level.

    Returns:
        Current debug level string.
    """
    return _DEBUG_LEVEL


def set_debug_level(level: str) -> None:
    """Sets the distributed debug level.

    Args:
        level: Debug level string (`'OFF'`, `'INFO'`, `'DETAIL'`).
    """
    global _DEBUG_LEVEL
    _DEBUG_LEVEL = level


def set_debug_level_from_env() -> None:
    """Sets the distributed debug level from environment configuration."""
    set_debug_level(DebugLevel.OFF)


def set_timeout(timeout: timedelta, group: ProcessGroup | None = None) -> None:
    """Sets the default timeout for all future operations on a process group.

    Args:
        timeout: Timeout duration.
        group: Target ProcessGroup or `None` for default group.
    """
    _ = (timeout, group)


def init_device_mesh(
    device_type: str,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim_names: tuple[str, ...] | None = None,
    backend_override: dict[int | str, str] | None = None,
) -> DeviceMesh:
    """Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names`.

    Args:
        device_type: Device type (`'cpu'`, `'cuda'`, etc.).
        mesh_shape: Shape tuple of the mesh.
        mesh_dim_names: Names of each mesh dimension.
        backend_override: Optional backend override mapping.

    Returns:
        Constructed `DeviceMesh`.
    """
    _ = backend_override
    return DeviceMesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)


def new_group(
    ranks: Sequence[int] | None = None,
    timeout: timedelta | None = None,
    backend: str | None = None,
    pg_options: OpArg = None,
    use_local_synchronization: bool = False,
    group_desc: str | None = None,
    device_id: DeviceLike = None,
    sort_ranks: bool = True,
) -> ProcessGroup:
    """Creates a new distributed group.

    Args:
        ranks: List of ranks of group members.
        timeout: Timeout for operations executed against the process group.
        backend: The backend to use.
        pg_options: Process group options.
        use_local_synchronization: Perform local group creation only.
        group_desc: Description of the group.
        device_id: Optional bound device ID.
        sort_ranks: Whether to sort `ranks`.

    Returns:
        Created `ProcessGroup`.
    """
    _ = (timeout, pg_options, use_local_synchronization, group_desc, device_id)
    r_list = sorted(ranks) if (ranks is not None and sort_ranks) else list(ranks or [0])
    eff_backend = backend if backend is not None else "gloo"
    return ProcessGroup(
        rank=0, world_size=max(1, len(r_list)), backend=eff_backend, ranks=r_list
    )


def new_subgroups(
    group_size: int | None = None,
    group: ProcessGroup | None = None,
    timeout: timedelta | None = None,
    backend: str | None = None,
    pg_options: OpArg = None,
    group_desc: str | None = None,
) -> tuple[ProcessGroup, list[ProcessGroup]]:
    """Creates subgroups of equal size.

    Args:
        group_size: Size of each subgroup.
        group: Parent process group.
        timeout: Timeout for operations.
        backend: Backend name.
        pg_options: Process group options.
        group_desc: Group description.

    Returns:
        Tuple `(current_rank_subgroup, all_subgroups)`.
    """
    _ = (group, timeout, pg_options, group_desc)
    sz = group_size if group_size is not None else 1
    pg = new_group(ranks=list(range(sz)), backend=backend)
    return (pg, [pg])


def new_subgroups_by_enumeration(
    ranks_per_subgroup_list: Sequence[Sequence[int]],
    timeout: timedelta | None = None,
    backend: str | None = None,
    pg_options: OpArg = None,
    group_desc: str | None = None,
) -> tuple[ProcessGroup | None, list[ProcessGroup]]:
    """Creates subgroups by dividing the global world according to `ranks_per_subgroup_list`.

    Args:
        ranks_per_subgroup_list: List of rank sequences per subgroup.
        timeout: Timeout for operations.
        backend: Backend name.
        pg_options: Process group options.
        group_desc: Group description.

    Returns:
        Tuple `(my_group, all_subgroups)`.
    """
    _ = (timeout, pg_options, group_desc)
    subgroups = [new_group(ranks=r, backend=backend) for r in ranks_per_subgroup_list]
    my_group = subgroups[0] if subgroups else None
    return (my_group, subgroups)


def split_group(
    parent_pg: ProcessGroup | None = None,
    split_ranks: Sequence[Sequence[int]] | None = None,
    timeout: timedelta | None = None,
    pg_options: OpArg = None,
    group_desc: str | None = None,
    backend: str | None = None,
) -> ProcessGroup:
    """Creates a new process group split from `parent_pg`.

    Args:
        parent_pg: Parent process group.
        split_ranks: List of rank lists to split into.
        timeout: Timeout for operations.
        pg_options: Process group options.
        group_desc: Group description.
        backend: Backend name.

    Returns:
        Created child `ProcessGroup`.
    """
    _ = (parent_pg, timeout, pg_options, group_desc)
    first_ranks = list(split_ranks[0]) if split_ranks else [0]
    return new_group(ranks=first_ranks, backend=backend)


def shrink_group(
    ranks_to_exclude: Sequence[int],
    group: ProcessGroup | None = None,
    shrink_flags: int | str = "SHRINK_DEFAULT",
    pg_options: OpArg = None,
) -> ProcessGroup:
    """Shrinks a process group by excluding specified ranks.

    Args:
        ranks_to_exclude: Ranks to remove from the process group.
        group: Process group to shrink.
        shrink_flags: Shrink mode flags.
        pg_options: Process group options.

    Returns:
        Shrunk `ProcessGroup`.
    """
    _ = (shrink_flags, pg_options)
    base_ranks = get_process_group_ranks(group)
    excluded = set(ranks_to_exclude)
    remaining = [r for r in base_ranks if r not in excluded]
    return new_group(ranks=remaining or [0])


def barrier(
    group: ProcessGroup | None = None,
    async_op: bool = False,
    device_ids: Sequence[int] | None = None,
    timeout: timedelta | None = None,
) -> Work | None:
    """Synchronizes all processes in `group`.

    Args:
        group: Process group to synchronize.
        async_op: Whether to return an asynchronous `Work` handle.
        device_ids: Optional list of GPU/device IDs.
        timeout: Optional timeout duration.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (group, device_ids, timeout)
    return Work() if async_op else None


def monitored_barrier(
    group: ProcessGroup | None = None,
    timeout: timedelta | None = None,
    wait_all_ranks: bool = False,
) -> None:
    """Synchronizes processes similar to `barrier` with configurable timeout monitoring.

    Args:
        group: Process group to synchronize.
        timeout: Timeout duration.
        wait_all_ranks: Whether to wait for all ranks to fail/complete.
    """
    _ = (group, timeout, wait_all_ranks)


def broadcast(
    tensor: Tensor,
    src: int | None = 0,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_src: int | None = None,
) -> Work | None:
    """Broadcasts the tensor to the whole group.

    Args:
        tensor: Data to be sent if `src` is the rank of current process, and tensor to save received data otherwise.
        src: Source rank in global process group.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_src: Source rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (tensor, src, group, group_src)
    return Work() if async_op else None


def broadcast_object_list(
    object_list: list[OpArg],
    src: int | None = 0,
    group: ProcessGroup | None = None,
    device: DeviceLike = None,
    group_src: int | None = None,
    weights_only: bool = False,
) -> None:
    """Broadcasts picklable objects in `object_list` to the whole group.

    Args:
        object_list: List of input objects to broadcast.
        src: Source rank.
        group: Process group to work on.
        device: Optional device on which to serialize.
        group_src: Source rank within `group`.
        weights_only: Whether to enforce weights_only unpickling.
    """
    _ = (object_list, src, group, device, group_src, weights_only)


def all_reduce(
    tensor: Tensor,
    op: str = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Reduces the tensor data across all machines in a way that all get the final result.

    Args:
        tensor: Input and output of the collective. The function operates in-place.
        op: One of the values from `torch.distributed.ReduceOp` enum.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (tensor, op, group)
    return Work() if async_op else None


def all_reduce_coalesced(
    tensors: Sequence[Tensor] | Tensor,
    op: str = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Coalesced `all_reduce` across a list of tensors.

    Args:
        tensors: Tensor or sequence of tensors to reduce in-place.
        op: Reduction operation.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (tensors, op, group)
    return Work() if async_op else None


def reduce(
    tensor: Tensor,
    dst: int | None = 0,
    op: str = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> Work | None:
    """Reduces the tensor data across all machines to `dst`.

    Args:
        tensor: Input and output of the collective.
        dst: Destination rank.
        op: Reduction operation.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_dst: Destination rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (tensor, dst, op, group, group_dst)
    return Work() if async_op else None


def all_gather(
    tensor_list: list[Tensor],
    tensor: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Gathers tensors from the whole group in a list.

    Args:
        tensor_list: Output list of appropriately-sized tensors.
        tensor: Tensor to be broadcast from current process.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = group
    if tensor_list:
        tensor_list[0].copy_(tensor)
    return Work() if async_op else None


def all_gather_into_tensor(
    output_tensor: Tensor,
    input_tensor: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Gathers tensors from all ranks and puts them in a single `output_tensor`.

    Args:
        output_tensor: Output tensor to accommodate tensor elements from all ranks.
        input_tensor: Tensor to be gathered from current process.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = group
    output_tensor.copy_(input_tensor)
    return Work() if async_op else None


def all_gather_single(
    output_tensor: Tensor,
    input_tensor: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Alias for `all_gather_into_tensor`.

    Args:
        output_tensor: Output tensor.
        input_tensor: Input tensor.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    return all_gather_into_tensor(
        output_tensor, input_tensor, group=group, async_op=async_op
    )


def all_gather_coalesced(
    output_tensor_lists: list[list[Tensor]],
    input_tensor_list: list[Tensor],
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Gathers input tensors from the whole group in a list in a coalesced manner.

    Args:
        output_tensor_lists: Output list of tensor lists per rank.
        input_tensor_list: Input tensor list from current rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = group
    if output_tensor_lists and output_tensor_lists[0]:
        for dst_t, src_t in zip(output_tensor_lists[0], input_tensor_list):
            dst_t.copy_(src_t)
    return Work() if async_op else None


def all_gather_object(
    object_list: list[OpArg],
    obj: OpArg,
    group: ProcessGroup | None = None,
    weights_only: bool = False,
) -> None:
    """Gathers picklable objects from the whole group into `object_list`.

    Args:
        object_list: Output list of size `world_size`.
        obj: Picklable Python object to gather from current process.
        group: Process group to work on.
        weights_only: Whether to enforce weights_only unpickling.
    """
    _ = (group, weights_only)
    if object_list:
        object_list[0] = obj


def gather(
    tensor: Tensor,
    gather_list: list[Tensor] | None = None,
    dst: int | None = 0,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> Work | None:
    """Gathers a list of tensors in a single process.

    Args:
        tensor: Input tensor.
        gather_list: List of appropriately-sized tensors to use for gathered data on `dst`.
        dst: Destination rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_dst: Destination rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (dst, group, group_dst)
    if gather_list:
        gather_list[0].copy_(tensor)
    return Work() if async_op else None


def gather_into_tensor(
    tensor: Tensor,
    gather_tensor: Tensor | None = None,
    dst: int | None = 0,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> Work | None:
    """Gathers the input tensor from all ranks into a single output `gather_tensor` on `dst`.

    Args:
        tensor: Input tensor to gather.
        gather_tensor: Output buffer tensor on `dst`.
        dst: Destination rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_dst: Destination rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (dst, group, group_dst)
    if gather_tensor is not None:
        gather_tensor.copy_(tensor)
    return Work() if async_op else None


def gather_single(
    tensor: Tensor,
    gather_tensor: Tensor | None = None,
    dst: int | None = 0,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> Work | None:
    """Alias for `gather_into_tensor`.

    Args:
        tensor: Input tensor to gather.
        gather_tensor: Output buffer tensor on `dst`.
        dst: Destination rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_dst: Destination rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    return gather_into_tensor(
        tensor,
        gather_tensor=gather_tensor,
        dst=dst,
        group=group,
        async_op=async_op,
        group_dst=group_dst,
    )


def gather_object(
    obj: OpArg,
    object_gather_list: list[OpArg] | None = None,
    dst: int | None = 0,
    group: ProcessGroup | None = None,
    group_dst: int | None = None,
    weights_only: bool = False,
) -> None:
    """Gathers picklable objects from the whole group in a single process.

    Args:
        obj: Input object to send.
        object_gather_list: Output list on `dst` rank.
        dst: Destination rank.
        group: Process group to work on.
        group_dst: Destination rank on `group`.
        weights_only: Whether to enforce weights_only unpickling.
    """
    _ = (dst, group, group_dst, weights_only)
    if object_gather_list:
        object_gather_list[0] = obj


def scatter(
    tensor: Tensor,
    scatter_list: list[Tensor] | None = None,
    src: int | None = 0,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_src: int | None = None,
) -> Work | None:
    """Scatters a list of tensors to all processes in a group.

    Args:
        tensor: Output tensor.
        scatter_list: List of tensors to scatter on `src`.
        src: Source rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        group_src: Source rank on `group`.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (src, group, group_src)
    if scatter_list:
        tensor.copy_(scatter_list[0])
    return Work() if async_op else None


def scatter_object_list(
    scatter_object_output_list: list[OpArg],
    scatter_object_input_list: list[OpArg] | None = None,
    src: int | None = 0,
    group: ProcessGroup | None = None,
    group_src: int | None = None,
    weights_only: bool = False,
) -> None:
    """Scatters picklable objects in `scatter_object_input_list` to the whole group.

    Args:
        scatter_object_output_list: Non-empty 1-element list to store received object.
        scatter_object_input_list: List of objects to scatter on `src`.
        src: Source rank.
        group: Process group to work on.
        group_src: Source rank on `group`.
        weights_only: Whether to enforce weights_only unpickling.
    """
    _ = (src, group, group_src, weights_only)
    if scatter_object_output_list and scatter_object_input_list:
        scatter_object_output_list[0] = scatter_object_input_list[0]


def reduce_scatter(
    output: Tensor,
    input_list: list[Tensor],
    op: str = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output: Output tensor.
        input_list: List of tensors to reduce and scatter.
        op: Reduction operation.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (op, group)
    if input_list:
        output.copy_(input_list[0])
    return Work() if async_op else None


def reduce_scatter_tensor(
    output: Tensor,
    input: Tensor,
    op: str = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Reduces, then scatters a tensor across all ranks in a group.

    Args:
        output: Output tensor.
        input: Input tensor.
        op: Reduction operation.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (op, group)
    output.copy_(input)
    return Work() if async_op else None


def reduce_scatter_single(
    output: Tensor,
    input: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    op: str = ReduceOp.SUM,
) -> Work | None:
    """Alias for `reduce_scatter_tensor`.

    Args:
        output: Output tensor.
        input: Input tensor.
        group: Process group to work on.
        async_op: Whether this op should be an async op.
        op: Reduction operation.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    return reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)


def all_to_all(
    output_tensor_list: list[Tensor],
    input_tensor_list: list[Tensor],
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Scatters list of input tensors to all processes in a group and returns gathered list.

    Args:
        output_tensor_list: List of tensors to be gathered one per rank.
        input_tensor_list: List of tensors to scatter one per rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = group
    for out_t, in_t in zip(output_tensor_list, input_tensor_list):
        out_t.copy_(in_t)
    return Work() if async_op else None


def all_to_all_single(
    output: Tensor,
    input: Tensor,
    output_split_sizes: Sequence[int] | None = None,
    input_split_sizes: Sequence[int] | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Work | None:
    """Splits input tensor and scatters the split list to all processes in a group.

    Args:
        output: Gathered concatenated output tensor.
        input: Input tensor to scatter.
        output_split_sizes: Output split sizes per rank.
        input_split_sizes: Input split sizes per rank.
        group: Process group to work on.
        async_op: Whether this op should be an async op.

    Returns:
        `Work` handle if `async_op=True`, else `None`.
    """
    _ = (output_split_sizes, input_split_sizes, group)
    output.copy_(input)
    return Work() if async_op else None


def send(
    tensor: Tensor,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_dst: int | None = None,
) -> None:
    """Sends a tensor synchronously.

    Args:
        tensor: Tensor to send.
        dst: Destination rank.
        group: Process group to work on.
        tag: Tag to match send with remote recv.
        group_dst: Destination rank on `group`.
    """
    _ = (tensor, dst, group, tag, group_dst)


def recv(
    tensor: Tensor,
    src: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_src: int | None = None,
) -> int:
    """Receives a tensor synchronously.

    Args:
        tensor: Tensor to fill with received data.
        src: Source rank.
        group: Process group to work on.
        tag: Tag to match recv with remote send.
        group_src: Source rank on `group`.

    Returns:
        Sender rank (`0` if unspecified).
    """
    _ = (tensor, group, tag, group_src)
    return src if src is not None else 0


def isend(
    tensor: Tensor,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_dst: int | None = None,
) -> Work:
    """Sends a tensor asynchronously.

    Args:
        tensor: Tensor to send.
        dst: Destination rank.
        group: Process group to work on.
        tag: Tag to match send with remote irecv.
        group_dst: Destination rank on `group`.

    Returns:
        Asynchronous `Work` handle.
    """
    _ = (tensor, dst, group, tag, group_dst)
    return Work()


def irecv(
    tensor: Tensor,
    src: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_src: int | None = None,
) -> Work:
    """Receives a tensor asynchronously.

    Args:
        tensor: Tensor to fill with received data.
        src: Source rank.
        group: Process group to work on.
        tag: Tag to match irecv with remote isend.
        group_src: Source rank on `group`.

    Returns:
        Asynchronous `Work` handle.
    """
    _ = (tensor, src, group, tag, group_src)
    return Work()


def batch_isend_irecv(p2p_op_list: Sequence[OpArg]) -> list[Work]:
    """Sends or receives a batch of tensors asynchronously and returns a list of requests.

    Args:
        p2p_op_list: List of point-to-point operations.

    Returns:
        List of `Work` handles.
    """
    return [Work() for _ in p2p_op_list]


def send_object_list(
    object_list: Sequence[OpArg],
    dst: int | None = None,
    group: ProcessGroup | None = None,
    device: DeviceLike = None,
    group_dst: int | None = None,
    use_batch: bool = False,
    weights_only: bool = False,
) -> int:
    """Sends picklable objects in `object_list` synchronously.

    Args:
        object_list: Sequence of objects to send.
        dst: Destination rank.
        group: Process group to work on.
        device: Optional device.
        group_dst: Destination rank on `group`.
        use_batch: Whether to batch objects.
        weights_only: Whether to enforce weights_only unpickling.

    Returns:
        Status code (`0`).
    """
    _ = (object_list, dst, group, device, group_dst, use_batch, weights_only)
    return 0


def recv_object_list(
    object_list: list[OpArg],
    src: int | None = None,
    group: ProcessGroup | None = None,
    device: DeviceLike = None,
    group_src: int | None = None,
    use_batch: bool = False,
    weights_only: bool = False,
) -> int:
    """Receives picklable objects in `object_list` synchronously.

    Args:
        object_list: List to fill with received objects.
        src: Source rank.
        group: Process group to work on.
        device: Optional device.
        group_src: Source rank on `group`.
        use_batch: Whether to batch objects.
        weights_only: Whether to enforce weights_only unpickling.

    Returns:
        Sender rank (`0` if unspecified).
    """
    _ = (object_list, group, device, group_src, use_batch, weights_only)
    return src if src is not None else 0


def breakpoint(rank: int = 0, skip: int = 0, timeout_s: int = 3600) -> None:
    """Sets a distributed breakpoint on a single rank while other ranks wait.

    Args:
        rank: Rank on which to trigger the breakpoint.
        skip: Number of calls to skip before breaking.
        timeout_s: Barrier timeout in seconds.
    """
    _ = (rank, skip, timeout_s)


@contextmanager
def record_comm(name: str) -> Iterator[None]:
    """Context manager to set a custom profiling name for communication collectives.

    Args:
        name: Custom profiling label.

    Yields:
        None.
    """
    _ = name
    yield


def register_rendezvous_handler(scheme: str, handler: Callable[[], None]) -> None:
    """Registers a new rendezvous handler for the given URL `scheme`.

    Args:
        scheme: URL scheme string (e.g. `'tcp'`, `'env'`, `'file'`).
        handler: Rendezvous generator/callable.
    """
    _RENDEZVOUS_HANDLERS[scheme] = handler


def rendezvous(
    url: str,
    rank: int = -1,
    world_size: int = -1,
    **kwargs: OpArg,
) -> Iterator[tuple[dict[str, str], int, int]]:
    """Initializes a rendezvous iterator for `url`, `rank`, and `world_size`.

    Args:
        url: Rendezvous URL string.
        rank: Rank of the current process.
        world_size: Total number of processes.
        **kwargs: Additional keyword arguments.

    Yields:
        Tuple `(store_dict, rank, world_size)`.
    """
    _ = (url, kwargs)
    yield ({}, max(0, rank), max(1, world_size))


def supports_complex(reduceOp: str) -> bool:
    """Returns True if complex tensors are supported for the given `reduceOp`.

    Args:
        reduceOp: Reduction operation identifier.

    Returns:
        True for `SUM`, `AVG`, and `PRODUCT`.
    """
    return reduceOp in (ReduceOp.SUM, ReduceOp.AVG, ReduceOp.PRODUCT)
