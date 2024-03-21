import funcnodes as fn
from typing import (
    Union,
    List,
    Optional,
    Iterable,
    Tuple,
    Sequence,
    Literal,
    Any,
    Callable,
)

from exposedfunctionality import controlled_wrapper as wraps
import numpy
from .._dtypes import dtype_from_name, DTYPE_ENUM
from .._types import (
    array_like,
    ndarray,
    ndarray_or_number,
    shape_like,
    axis_like,
    scalar,
    indices_or_sections,
    ndarray_or_scalar,
    int_array,
    bitarray,
    int_bool_array,
    int_or_int_array,
    NoValue,
    real_array,
    buffer_like,
)
from ._fromnumeric import *
from ._multiarray import *
from ._defchararray import *
from ._datetime import *


@fn.NodeDecorator(
    node_id="np.empty",
    name="empty",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.empty)
def empty(
    shape: shape_like,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    # order: OrderCF = "C",
    # like: Optional[array_like] = None,
):
    res = numpy.empty(
        shape=shape,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.empty_like",
    name="empty_like",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.empty_like)
def empty_like(
    prototype: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = "K",
    # subok: Optional[bool] = True,
    # shape: Optional[shape_like] = None,
):  # params ['prototype'] ['dtype', 'order', 'subok'] []
    res = numpy.empty_like(
        prototype=prototype,
        dtype=dtype_from_name(dtype),
        # order=order,
        # subok=subok,
        # shape=shape,
    )
    return res


@fn.NodeDecorator(
    node_id="np.eye",
    name="eye",
    outputs=[{"name": "I", "type": "ndarray"}],
)
@wraps(numpy.eye)
def eye(
    N: int,
    M: Optional[int] = None,
    k: Optional[int] = 0,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    # order: OrderCF = "C",
    # like: Optional[array_like] = None,
):  # params ['N'] ['M', 'k', 'dtype', 'order', 'like'] []
    res = numpy.eye(
        N=N,
        M=M,
        k=k,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.identity",
    name="identity",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.identity)
def identity(
    n: int,
    dtype: Optional[DTYPE_ENUM] = None,
    # like: Optional[array_like] = None,
):  # params ['n'] ['dtype', 'like'] []
    res = numpy.identity(
        n=n,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ones",
    name="ones",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ones)
def ones(
    shape: shape_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderCF = "C",
    # like: Optional[array_like] = None,
):  # params ['shape'] ['dtype', 'order', 'like'] []
    res = numpy.ones(
        shape=shape,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ones_like",
    name="ones_like",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ones_like)
def ones_like(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = "K",
    # subok: Optional[bool] = True,
    # shape: Optional[shape_like] = None,
):  # params ['a'] ['dtype', 'order', 'subok', 'shape'] []
    res = numpy.ones_like(
        a=a,
        dtype=dtype_from_name(dtype),
        # order=order,
        # subok=subok,
        # shape=shape,
    )
    return res


@fn.NodeDecorator(
    node_id="np.zeros",
    name="zeros",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.zeros)
def zeros(
    shape: shape_like,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    # order: OrderCF = "C",
    # like: Optional[array_like] = None,
):  # params ['shape'] ['dtype', 'order', 'like'] []
    res = numpy.zeros(
        shape=shape,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.zeros_like",
    name="zeros_like",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.zeros_like)
def zeros_like(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = "K",
    # subok: Optional[bool] = True,
    # shape: Optional[shape_like] = None,
):  # params ['a'] ['dtype', 'order', 'subok', 'shape'] []
    res = numpy.zeros_like(
        a=a,
        dtype=dtype_from_name(dtype),
        # order=order,
        # subok=subok,
        # shape=shape,
    )
    return res


@fn.NodeDecorator(
    node_id="np.full",
    name="full",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.full)
def full(
    shape: shape_like,
    fill_value: ndarray_or_scalar,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderCF = "C",
    # like: Optional[array_like] = None,
):  # params ['shape', 'fill_value'] ['dtype', 'order', 'like'] []
    res = numpy.full(
        shape=shape,
        fill_value=fill_value,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.full_like",
    name="full_like",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.full_like)
def full_like(
    a: array_like,
    fill_value: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = "K",
    # subok: Optional[bool] = True,
    # shape: Optional[shape_like] = None,
):  # params ['a', 'fill_value'] ['dtype', 'order'] []
    res = numpy.full_like(
        a=a,
        fill_value=fill_value,
        dtype=dtype_from_name(dtype),
        # order=order,
        # subok=subok,
        # shape=shape,
    )
    return res


@fn.NodeDecorator(
    node_id="np.meshgrid",
    name="meshgrid",
    outputs=[{"name": "XY", "type": "List[ndarray]"}],
)
@wraps(numpy.meshgrid)
def meshgrid(
    xi: List[array_like],
    indexing: Literal["xy", "ij"] = "xy",
    sparse: Optional[bool] = False,
):
    res = numpy.meshgrid(
        *xi,
        indexing=indexing,
        sparse=sparse,
    )
    return res


@fn.NodeDecorator(
    node_id="np.array",
    name="array",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.array)
def array(
    object: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    copy: Optional[bool] = True,
    # order: OrderKACF = "K",
    subok: Optional[bool] = False,
    ndmin: Optional[int] = 0,
    # like: Optional[array_like] = None,
):  # params ['object'] ['dtype', 'copy', 'order', 'subok'] []
    res = numpy.array(
        object=object,
        dtype=dtype_from_name(dtype),
        copy=copy,
        # order=order,
        # subok=subok,
        ndmin=ndmin,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.asarray",
    name="asarray",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.asarray)
def asarray(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = None,
    # like: Optional[array_like] = None,
):  # params ['a'] ['dtype', 'order', 'like'] []
    res = numpy.asarray(
        a=a,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.asanyarray",
    name="asanyarray",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.asanyarray)
def asanyarray(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = None,
    # like: Optional[array_like] = None,
):  # params ['a'] ['dtype', 'order', 'like'] []
    res = numpy.asanyarray(
        a=a,
        dtype=dtype_from_name(dtype),
        # order=order,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ascontiguousarray",
    name="ascontiguousarray",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ascontiguousarray)
def ascontiguousarray(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # like: Optional[array_like] = None,
):  # params ['a'] ['dtype', 'like'] []
    res = numpy.ascontiguousarray(
        a=a,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.atleast_1d",
    name="atleast_1d",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.atleast_1d)
def atleast_1d(
    arr: array_like,
):
    res = numpy.atleast_1d(
        arr,
    )
    return res


@fn.NodeDecorator(
    node_id="np.atleast_2d",
    name="atleast_2d",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.atleast_2d)
def atleast_2d(
    arr: array_like,
):
    res = numpy.atleast_2d(
        arr,
    )
    return res


@fn.NodeDecorator(
    node_id="np.atleast_3d",
    name="atleast_3d",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.atleast_3d)
def atleast_3d(
    arr: array_like,
):
    res = numpy.atleast_3d(
        arr,
    )
    return res


@fn.NodeDecorator(
    node_id="np.copy",
    name="copy",
    outputs=[{"name": "arr", "type": "ndarray"}],
)
@wraps(numpy.copy)
def copy(
    a: array_like,
    # order: OrderKACF = "K",
    subok: Optional[bool] = False,
):  # params ['a'] ['order', 'subok'] []
    res = numpy.copy(
        a=a,
        # order=order,
        # subok=subok,
    )
    return res


@fn.NodeDecorator(
    node_id="np.frombuffer",
    name="frombuffer",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.frombuffer)
def frombuffer(
    buffer: buffer_like,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
    # like: Optional[array_like] = None,
):  # params ['buffer'] ['dtype', 'count', 'offset', 'like'] []
    res = numpy.frombuffer(
        buffer=buffer_like,
        dtype=dtype_from_name(dtype),
        count=count,
        offset=offset,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.from_dlpack",
    name="from_dlpack",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.from_dlpack)
def from_dlpack(
    x: object,
):  # params ['x'] [] []
    res = numpy.from_dlpack(
        x=x,
    )
    return res


# @fn.NodeDecorator(
#     node_id="np.fromfile",
#     name="fromfile",
#     outputs=[],
# )
# @wraps(numpy.fromfile)
# def fromfile(
#     file: file or str or Path,
#     dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
#     count: int = -1,
#     sep: str = "",
#     offset: int = 0,
#    # like: Optional[array_like] = None,
# ):  # params ['file'] ['dtype', 'count', 'sep', 'offset', 'like'] []
#     res = numpy.fromfile(
#         file=file,
#         dtype=dtype_from_name(dtype),
#         count=count,
#         sep=sep,
#         offset=offset,
#         # like=like,
#     )
#     return res


@fn.NodeDecorator(
    node_id="np.fromfunction",
    name="fromfunction",
    outputs=[{"name": "fromfunction", "type": "ndarray_or_number"}],
)
@wraps(numpy.fromfunction)
def fromfunction(
    function: Callable,
    shape: shape_like,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    # like: Optional[array_like] = None,
):  # params ['function', 'shape'] ['dtype', 'like'] []
    res = numpy.fromfunction(
        function=function,
        shape=shape,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fromiter",
    name="fromiter",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.fromiter)
def fromiter(
    iter: Iterable,
    dtype: DTYPE_ENUM,
    count: Optional[int] = -1,
    # like: Optional[array_like] = None,
):  # params ['iter', 'dtype'] ['count', 'like'] []
    res = numpy.fromiter(
        iter=iter,
        dtype=dtype_from_name(dtype),
        count=count,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fromstring",
    name="fromstring",
    outputs=[{"name": "arr", "type": "ndarray"}],
)
@wraps(numpy.fromstring)
def fromstring(
    string: str,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    count: Optional[int] = -1,
    sep: Optional[str] = ",",
    # like: Optional[array_like] = None,
):  # params ['string'] ['dtype', 'count', 'like'] []
    res = numpy.fromstring(
        string=string,
        dtype=dtype_from_name(dtype),
        count=count,
        sep=sep,
        # like=like,
    )
    return res


# @fn.NodeDecorator(
#     node_id="np.loadtxt",
#     name="loadtxt",
#     outputs=[{"name": "out", "type": "ndarray"}],
# )
# @wraps(numpy.loadtxt)
# def loadtxt(
#     fname: file, str,
#     pathlib.Path, list of str, generator,
#     dtype:  DTYPE_ENUM = DTYPE_ENUM.float32,
#     comments: str or sequence of str or None,
#     optional = '#', delimiter: Optional[str] = None,
#     converters: dict or callable, optional = None,
#     skiprows: Optional[int]= 0, usecols: int or sequence, optional = None,
#     unpack: Optional[bool]= False,
#     ndmin: Optional[int]= 0,
#     encoding: Optional[str] = 'bytes',
#     max_rows: Optional[int]= None,
#     quotechar: unicode character or None,
#     optional = None,
#    # like: Optional[array_like] = None,
# ):  # params ['fname'] ['dtype', 'comments', 'delimiter'] []
#     res = numpy.loadtxt(
#         fname=fname,
#         dtype=dtype_from_name(dtype),
#         comments=comments,
#         delimiter=delimiter,
#         converters=converters,
#         skiprows=skiprows,
#         usecols=usecols,
#         unpack=unpack,
#         ndmin=ndmin,
#         encoding=encoding,
#         max_rows=max_rows,
#         quotechar=quotechar,
#         # like=like,
#     )
#     return res


@fn.NodeDecorator(
    node_id="np.arange",
    name="arange",
    outputs=[{"name": "arange", "type": "ndarray"}],
)
@wraps(numpy.arange)
def arange(
    stop: scalar,
    start: Optional[scalar] = 0,
    step: Optional[scalar] = 1,
    dtype: Optional[DTYPE_ENUM] = None,
    # like: Optional[array_like] = None,
):  # params ['stop'] ['start', 'step', 'dtype', 'like'] []
    res = numpy.arange(
        start=start,
        stop=stop,
        step=step,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.linspace",
    name="linspace",
    outputs=[
        {"name": "samples", "type": "ndarray"},
        {"name": "step", "type": "float"},
    ],
)
@wraps(numpy.linspace)
def linspace(
    start: array_like,
    stop: array_like,
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    # retstep: Optional[bool] = False,
    dtype: Optional[DTYPE_ENUM] = None,
    axis: Optional[int] = 0,
):  # params ['start', 'stop'] ['num', 'endpoint'] []
    res = numpy.linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        retstep=True,
        dtype=dtype_from_name(dtype),
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logspace",
    name="logspace",
    outputs=[{"name": "samples", "type": "ndarray"}],
)
@wraps(numpy.logspace)
def logspace(
    start: array_like,
    stop: array_like,
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    base: Optional[ndarray_or_scalar] = 10.0,
    dtype: Optional[DTYPE_ENUM] = None,
    axis: Optional[int] = 0,
):  # params ['start', 'stop'] ['num', 'endpoint', 'base'] []
    res = numpy.logspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype_from_name(dtype),
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.geomspace",
    name="geomspace",
    outputs=[{"name": "samples", "type": "ndarray"}],
)
@wraps(numpy.geomspace)
def geomspace(
    start: array_like,
    stop: array_like,
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    dtype: Optional[DTYPE_ENUM] = None,
    axis: Optional[int] = 0,
):  # params ['start', 'stop'] ['num', 'endpoint'] []
    res = numpy.geomspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        dtype=dtype_from_name(dtype),
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.diag",
    name="diag",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.diag)
def diag(
    v: array_like,
    k: Optional[int] = 0,
):  # params ['v'] ['k'] []
    res = numpy.diag(
        v=v,
        k=k,
    )
    return res


@fn.NodeDecorator(
    node_id="np.diagflat",
    name="diagflat",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.diagflat)
def diagflat(
    v: array_like,
    k: Optional[int] = 0,
):  # params ['v'] ['k'] []
    res = numpy.diagflat(
        v=v,
        k=k,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tri",
    name="tri",
    outputs=[{"name": "tri", "type": "ndarray"}],
)
@wraps(numpy.tri)
def tri(
    N: int,
    M: Optional[int] = None,
    k: Optional[int] = 0,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float32,
    # like: Optional[array_like] = None,
):  # params ['N'] ['M', 'k', 'dtype', 'like'] []
    res = numpy.tri(
        N=N,
        M=M,
        k=k,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tril",
    name="tril",
    outputs=[{"name": "tril", "type": "ndarray"}],
)
@wraps(numpy.tril)
def tril(
    m: ndarray,
    k: Optional[int] = 0,
):  # params ['m'] ['k'] []
    res = numpy.tril(
        m=m,
        k=k,
    )
    return res


@fn.NodeDecorator(
    node_id="np.triu",
    name="triu",
    outputs=[{"name": "triu", "type": "ndarray"}],
)
@wraps(numpy.triu)
def triu(
    m: ndarray,
    k: Optional[int] = 0,
):
    res = numpy.triu(
        m=m,
        k=k,
    )
    return res


@fn.NodeDecorator(
    node_id="np.vander",
    name="vander",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.vander)
def vander(
    x: array_like,
    N: Optional[int] = None,
    increasing: Optional[bool] = False,
):  # params ['x'] ['N', 'increasing'] []
    res = numpy.vander(
        x=x,
        N=N,
        increasing=increasing,
    )
    return res


@fn.NodeDecorator(
    node_id="np.mat",
    name="mat",
    outputs=[{"name": "mat", "type": "matrix"}],
)
@wraps(numpy.mat)
def mat(
    data: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
):  # params ['data'] ['dtype'] []
    res = numpy.mat(
        data=data,
        dtype=dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.bmat",
    name="bmat",
    outputs=[{"name": "out", "type": "matrix"}],
)
@wraps(numpy.bmat)
def bmat(
    obj: Union[str, ndarray],
    ldict: Optional[dict] = None,
    gdict: Optional[dict] = None,
):  # params ['obj'] ['ldict', 'gdict'] []
    res = numpy.bmat(
        obj=obj,
        ldict=ldict,
        gdict=gdict,
    )
    return res


@fn.NodeDecorator(
    node_id="np.copyto",
    name="copyto",
    outputs=[],
)
@wraps(numpy.copyto)
def copyto(
    dst: ndarray,
    src: array_like,
    # casting: casting_literal = "same_kind",
    # where: Union[bool_array, bool] = True,
):  # params ['dst', 'src'] ['casting', 'where'] []
    res = numpy.copyto(
        dst=dst,
        src=src,
        # casting=casting,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.shape",
    name="shape",
    outputs=[{"name": "shape", "type": "shape_like"}],
)
@wraps(numpy.shape)
def shape(
    a: array_like,
):  # params ['a'] [] []
    res = numpy.shape(
        a=a,
    )
    return res


@fn.NodeDecorator(
    node_id="np.size",
    name="size",
    outputs=[{"name": "size", "type": "Union[int, int_array]"}],
)
@wraps(numpy.size)
def size(
    a: array_like,
    axis: Optional[int] = None,
):
    res = numpy.size(
        a=a,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ndim",
    name="ndim",
    outputs=[{"name": "ndim", "type": "int"}],
)
@wraps(numpy.ndim)
def ndim(
    a: array_like,
):
    res = numpy.ndim(
        a=a,
    )
    return res


@fn.NodeDecorator(
    node_id="np.reshape",
    name="reshape",
    outputs=[{"name": "reshaped_array", "type": "ndarray"}],
)
@wraps(numpy.reshape)
def reshape(
    a: array_like,
    newshape: shape_like,
    # order: Optional[Literal["C", "F", "A"]] = "C",
):  # params ['a', 'newshape'] ['order'] []
    res = numpy.reshape(
        a=a,
        newshape=newshape,
        # order=order,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ravel",
    name="ravel",
    outputs=[{"name": "y", "type": "array_like"}],
)
@wraps(numpy.ravel)
def ravel(
    a: array_like,
    # order: Optional[Literal["C", "F", "A", "K"]] = "C",
):  # params ['a'] ['order'] []
    res = numpy.ravel(
        a=a,
        # order=order,
    )
    return res


@fn.NodeDecorator(
    node_id="np.moveaxis",
    name="moveaxis",
    outputs=[{"name": "result", "type": "ndarray"}],
)
@wraps(numpy.moveaxis)
def moveaxis(
    a: ndarray,
    source: axis_like,
    destination: axis_like,
):  # params ['a', 'source', 'destination'] [] []
    res = numpy.moveaxis(
        a=a,
        source=source,
        destination=destination,
    )
    return res


@fn.NodeDecorator(
    node_id="np.rollaxis",
    name="rollaxis",
    outputs=[{"name": "res", "type": "ndarray"}],
)
@wraps(numpy.rollaxis)
def rollaxis(
    a: ndarray,
    axis: int,
    start: Optional[int] = 0,
):  # params ['a', 'axis'] ['start'] []
    res = numpy.rollaxis(
        a=a,
        axis=axis,
        start=start,
    )
    return res


@fn.NodeDecorator(
    node_id="np.swapaxes",
    name="swapaxes",
    outputs=[{"name": "a_swapped", "type": "ndarray"}],
)
@wraps(numpy.swapaxes)
def swapaxes(
    a: array_like,
    axis1: int,
    axis2: int,
):  # params ['a', 'axis1', 'axis2'] [] []
    res = numpy.swapaxes(
        a=a,
        axis1=axis1,
        axis2=axis2,
    )
    return res


@fn.NodeDecorator(
    node_id="np.transpose",
    name="transpose",
    outputs=[{"name": "p", "type": "ndarray"}],
)
@wraps(numpy.transpose)
def transpose(
    a: array_like,
    axes: Optional[axis_like] = None,
):  # params ['a'] ['axes'] []
    res = numpy.transpose(
        a=a,
        axes=axes,
    )
    return res


@fn.NodeDecorator(
    node_id="np.broadcast_to",
    name="broadcast_to",
    outputs=[{"name": "broadcast", "type": "ndarray"}],
)
@wraps(numpy.broadcast_to)
def broadcast_to(
    array: array_like,
    shape: shape_like,
    subok: Optional[bool] = False,
):  # params ['array', 'shape'] ['subok'] []
    res = numpy.broadcast_to(
        array=array,
        shape=shape,
        # subok=subok,
    )
    return res


@fn.NodeDecorator(
    node_id="np.broadcast_arrays",
    name="broadcast_arrays",
    outputs=[{"name": "broadcasted", "type": "List[ndarray]"}],
)
@wraps(numpy.broadcast_arrays)
def broadcast_arrays(
    args: Sequence[array_like],
    subok: Optional[bool] = False,
):  # params [] ['args', 'subok'] []
    res = numpy.broadcast_arrays(
        *args,
        # subok=subok,
    )
    return res


@fn.NodeDecorator(
    node_id="np.expand_dims",
    name="expand_dims",
    outputs=[{"name": "result", "type": "ndarray"}],
)
@wraps(numpy.expand_dims)
def expand_dims(
    a: array_like,
    axis: axis_like,
):  # params ['a', 'axis'] [] []
    res = numpy.expand_dims(
        a=a,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.squeeze",
    name="squeeze",
    outputs=[{"name": "squeezed", "type": "ndarray"}],
)
@wraps(numpy.squeeze)
def squeeze(
    a: array_like,
    axis: Optional[axis_like] = None,
):  # params ['a'] ['axis'] []
    res = numpy.squeeze(
        a=a,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.asmatrix",
    name="asmatrix",
    outputs=[{"name": "mat", "type": "matrix"}],
)
@wraps(numpy.asmatrix)
def asmatrix(
    data: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
):  # params ['data'] ['dtype'] []
    res = numpy.asmatrix(
        data=data,
        dtype=dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.asfarray",
    name="asfarray",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.asfarray)
def asfarray(
    a: array_like,
    dtype: DTYPE_ENUM = DTYPE_ENUM.float64,
):  # params ['a'] ['dtype'] []
    res = numpy.asfarray(
        a=a,
        dtype=dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.asfortranarray",
    name="asfortranarray",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.asfortranarray)
def asfortranarray(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # like: Optional[array_like] = None,
):  # params ['a'] ['dtype', 'like'] []
    res = numpy.asfortranarray(
        a=a,
        dtype=dtype_from_name(dtype),
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.asarray_chkfinite",
    name="asarray_chkfinite",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.asarray_chkfinite)
def asarray_chkfinite(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = None,
):  # params ['a'] ['dtype', 'order'] []
    res = numpy.asarray_chkfinite(
        a=a,
        dtype=dtype_from_name(dtype),
        # order=order,
    )
    return res


@fn.NodeDecorator(
    node_id="np.require",
    name="require",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.require)
def require(
    a: array_like,
    dtype: Optional[DTYPE_ENUM] = None,
    requirements: Literal["F", "C", "A", "W", "O", "E"] = None,
    # like: Optional[array_like] = None,
):  # params ['a'] ['dtype', 'requirements', 'like'] []
    res = numpy.require(
        a=a,
        dtype=dtype_from_name(dtype),
        requirements=requirements,
        # like=like,
    )
    return res


@fn.NodeDecorator(
    node_id="np.concatenate",
    name="concatenate",
    outputs=[{"name": "res", "type": "ndarray"}],
)
@wraps(numpy.concatenate)
def concatenate(
    arrs: List[ndarray],
    axis: Optional[int] = 0,
    # out: Optional[ndarray] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # casting: casting_literal = "same_kind",
):  # params [] ['axis', 'out', 'dtype', 'casting'] []
    res = numpy.concatenate(
        *arrs,
        axis=axis,
        # out=out,
        dtype=dtype_from_name(dtype),
        # casting=casting,
    )
    return res


@fn.NodeDecorator(
    node_id="np.stack",
    name="stack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.stack)
def stack(
    arrays: Sequence[array_like],
    axis: Optional[int] = 0,
    # out: Optional[ndarray] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # casting: casting_literal = "same_kind",
):  # params ['arrays'] ['axis', 'out', 'dtype', 'casting'] []
    res = numpy.stack(
        arrays=arrays,
        axis=axis,
        # out=out,
        dtype=dtype_from_name(dtype),
        # casting=casting,
    )
    return res


# @fn.NodeDecorator(
#     node_id="np.block",
#     name="block",
#     outputs=[
#     {
#         "name": "block_array",
#         "type": "ndarray"
#     }
# ],
#     )
# @wraps(numpy.block)
# def block(arrays: nested list of array_like or scalars (but not tuples), ): # params ['arrays'] [] []
#     res = numpy.block(arrays=arrays, )
#     return res


@fn.NodeDecorator(
    node_id="np.vstack",
    name="vstack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.vstack)
def vstack(
    tup: Sequence[ndarray],
    dtype: Optional[DTYPE_ENUM] = None,
    # casting: casting_literal = "same_kind",
):  # params ['tup'] ['dtype', 'casting'] []
    res = numpy.vstack(
        tup=tup,
        dtype=dtype_from_name(dtype),
        # casting=casting,
    )
    return res


@fn.NodeDecorator(
    node_id="np.hstack",
    name="hstack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.hstack)
def hstack(
    tup: Sequence[ndarray],
    dtype: Optional[DTYPE_ENUM] = None,
    # casting: casting_literal = "same_kind",
):  # params ['tup'] ['dtype', 'casting'] []
    res = numpy.hstack(
        tup=tup,
        dtype=dtype_from_name(dtype),
        # casting=casting,
    )
    return res


@fn.NodeDecorator(
    node_id="np.dstack",
    name="dstack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.dstack)
def dstack(
    tup: Sequence[ndarray],
):  # params ['tup'] [] []
    res = numpy.dstack(
        tup=tup,
    )
    return res


@fn.NodeDecorator(
    node_id="np.column_stack",
    name="column_stack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.column_stack)
def column_stack(
    tup: Sequence[ndarray],
):  # params ['tup'] [] []
    res = numpy.column_stack(
        tup=tup,
    )
    return res


@fn.NodeDecorator(
    node_id="np.row_stack",
    name="row_stack",
    outputs=[{"name": "stacked", "type": "ndarray"}],
)
@wraps(numpy.row_stack)
def row_stack(
    tup: Sequence[ndarray],
    dtype: Optional[DTYPE_ENUM] = None,
    # casting: casting_literal = "same_kind",
):  # params ['tup'] ['dtype', 'casting'] []
    res = numpy.row_stack(
        tup=tup,
        dtype=dtype_from_name(dtype),
        # casting=casting,
    )
    return res


@fn.NodeDecorator(
    node_id="np.split",
    name="split",
    outputs=[{"name": "sub-arrays", "type": "List[ndarray]"}],
)
@wraps(numpy.split)
def split(
    ary: ndarray,
    indices_or_sections: indices_or_sections,
    axis: Optional[int] = 0,
):  # params ['ary', 'indices_or_sections'] ['axis'] []
    res = numpy.split(
        ary=ary,
        indices_or_sections=indices_or_sections,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.array_split",
    name="array_split",
    outputs=[{"name": "sub-arrays", "type": "List[ndarray]"}],
)
@wraps(numpy.array_split)
def array_split(
    ary: ndarray,
    indices_or_sections: shape_like,
    axis: Optional[int] = 0,
):
    res = numpy.array_split(
        ary=ary,
        indices_or_sections=indices_or_sections,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.dsplit",
    name="dsplit",
    outputs=[{"name": "sub-arrays", "type": "List[ndarray]"}],
)
@wraps(numpy.dsplit)
def dsplit(
    ary: ndarray,
    indices_or_sections: indices_or_sections,
):
    res = numpy.dsplit(
        ary=ary,
        indices_or_sections=indices_or_sections,
    )
    return res


@fn.NodeDecorator(
    node_id="np.hsplit",
    name="hsplit",
    outputs=[{"name": "sub-arrays", "type": "List[ndarray]"}],
)
@wraps(numpy.hsplit)
def hsplit(
    ary: ndarray,
    indices_or_sections: indices_or_sections,
):
    res = numpy.hsplit(
        ary=ary,
        indices_or_sections=indices_or_sections,
    )
    return res


@fn.NodeDecorator(
    node_id="np.vsplit",
    name="vsplit",
    outputs=[{"name": "sub-arrays", "type": "List[ndarray]"}],
)
@wraps(numpy.vsplit)
def vsplit(
    ary: ndarray,
    indices_or_sections: indices_or_sections,
):
    res = numpy.vsplit(
        ary=ary,
        indices_or_sections=indices_or_sections,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tile",
    name="tile",
    outputs=[{"name": "c", "type": "ndarray"}],
)
@wraps(numpy.tile)
def tile(
    A: array_like,
    reps: array_like,
):  # params ['A', 'reps'] [] []
    res = numpy.tile(
        A=A,
        reps=reps,
    )
    return res


@fn.NodeDecorator(
    node_id="np.repeat",
    name="repeat",
    outputs=[{"name": "repeated_array", "type": "ndarray"}],
)
@wraps(numpy.repeat)
def repeat(
    a: array_like,
    repeats: int_or_int_array,
    axis: Optional[int] = None,
):  # params ['a', 'repeats'] ['axis'] []
    res = numpy.repeat(
        a=a,
        repeats=repeats,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.delete",
    name="delete",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.delete)
def delete(
    arr: array_like,
    obj: int_or_int_array,
    axis: Optional[int] = None,
):  # params ['arr', 'obj'] ['axis'] []
    res = numpy.delete(
        arr=arr,
        obj=obj,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.insert",
    name="insert",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.insert)
def insert(
    arr: array_like,
    obj: int_or_int_array,
    values: array_like,
    axis: Optional[int] = None,
):  # params ['arr', 'obj', 'values'] ['axis'] []
    res = numpy.insert(
        arr=arr,
        obj=obj,
        values=values,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.append",
    name="append",
    outputs=[{"name": "append", "type": "ndarray"}],
)
@wraps(numpy.append)
def append(
    arr: array_like,
    values: array_like,
    axis: Optional[int] = None,
):  # params ['arr', 'values'] ['axis'] []
    res = numpy.append(
        arr=arr,
        values=values,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.pad",
    name="pad",
    outputs=[{"name": "padded", "type": "ndarray"}],
)
@wraps(numpy.pad)
def pad(
    array: array_like,
    pad_width: Union[int, Tuple[int, int], List[Tuple[int, int]]],
    mode: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
        "empty",
    ] = "constant",
    constant_values: Union[scalar, Tuple[scalar, scalar]] = 0,
):
    res = numpy.pad(
        array=array,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )
    return res


@fn.NodeDecorator(
    node_id="np.resize",
    name="resize",
    outputs=[{"name": "reshaped_array", "type": "ndarray"}],
)
@wraps(numpy.resize)
def resize(
    a: array_like,
    new_shape: shape_like,
):  # params ['a', 'new_shape'] [] []
    res = numpy.resize(
        a=a,
        new_shape=new_shape,
    )
    return res


@fn.NodeDecorator(
    node_id="np.trim_zeros",
    name="trim_zeros",
    outputs=[{"name": "trimmed", "type": "ndarray"}],
)
@wraps(numpy.trim_zeros)
def trim_zeros(
    filt: List[scalar],
    trim: Optional[str] = "fb",
):  # params ['filt'] ['trim'] []
    res = numpy.trim_zeros(
        filt=filt,
        trim=trim,
    )
    return res


@fn.NodeDecorator(
    node_id="np.unique",
    name="unique",
    outputs=[
        {"name": "unique", "type": "ndarray"},
        {"name": "unique_indices", "type": "Union[ndarray,None]"},
        {"name": "unique_inverse", "type": "Union[ndarray,None]"},
        {"name": "unique_counts", "type": "Union[ndarray,None]"},
    ],
)
@wraps(numpy.unique)
def unique(
    ar: array_like,
    return_index: Optional[bool] = False,
    return_inverse: Optional[bool] = False,
    return_counts: Optional[bool] = False,
    axis: Optional[int] = None,
    equal_nan: Optional[bool] = True,
):  # params ['ar'] ['return_index', 'return_inverse'] []
    res = numpy.unique(
        ar=ar,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
    )
    _uq = res[0]
    ix = 1
    if return_index:
        _uq_ix = res[ix]
        ix += 1
    else:
        _uq_ix = None

    if return_inverse:
        _uq_inv = res[ix]
        ix += 1
    else:
        _uq_inv = None

    if return_counts:
        _uq_cnt = res[ix]
        ix += 1
    else:
        _uq_cnt = None

    return _uq, _uq_ix, _uq_inv, _uq_cnt


@fn.NodeDecorator(
    node_id="np.flip",
    name="flip",
    outputs=[{"name": "out", "type": "array_like"}],
)
@wraps(numpy.flip)
def flip(
    m: array_like,
    axis: Optional[axis_like] = None,
):  # params ['m'] ['axis'] []
    res = numpy.flip(
        m=m,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fliplr",
    name="fliplr",
    outputs=[{"name": "f", "type": "ndarray"}],
)
@wraps(numpy.fliplr)
def fliplr(
    m: array_like,
):  # params ['m'] [] []
    res = numpy.fliplr(
        m=m,
    )
    return res


@fn.NodeDecorator(
    node_id="np.flipud",
    name="flipud",
    outputs=[{"name": "out", "type": "array_like"}],
)
@wraps(numpy.flipud)
def flipud(
    m: array_like,
):  # params ['m'] [] []
    res = numpy.flipud(
        m=m,
    )
    return res


@fn.NodeDecorator(
    node_id="np.roll",
    name="roll",
    outputs=[{"name": "res", "type": "ndarray"}],
)
@wraps(numpy.roll)
def roll(
    a: array_like,
    shift: axis_like,
    axis: Optional[axis_like] = None,
):  # params ['a', 'shift'] ['axis'] []
    res = numpy.roll(
        a=a,
        shift=shift,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.rot90",
    name="rot90",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.rot90)
def rot90(
    m: array_like,
    k: int = 1,
    axes: axis_like = (0, 1),
):  # params ['m'] ['k', 'axes'] []
    res = numpy.rot90(
        m=m,
        k=k,
        axes=axes,
    )
    return res


@fn.NodeDecorator(
    node_id="np.bitwise_and",
    name="bitwise_and",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.bitwise_and)
def bitwise_and(
    x1: int_bool_array,
    x2: int_bool_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.bitwise_and(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.bitwise_or",
    name="bitwise_or",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.bitwise_or)
def bitwise_or(
    x1: int_bool_array,
    x2: int_bool_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.bitwise_or(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.bitwise_xor",
    name="bitwise_xor",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.bitwise_xor)
def bitwise_xor(
    x1: int_bool_array,
    x2: int_bool_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.bitwise_xor(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.invert",
    name="invert",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.invert)
def invert(
    x: int_bool_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.invert(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.left_shift",
    name="left_shift",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.left_shift)
def left_shift(
    x1: int_or_int_array,
    x2: int_or_int_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.left_shift(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.right_shift",
    name="right_shift",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.right_shift)
def right_shift(
    x1: int_or_int_array,
    x2: int_or_int_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.right_shift(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.packbits",
    name="packbits",
    outputs=[{"name": "packed", "type": "ndarray"}],
)
@wraps(numpy.packbits)
def packbits(
    a: int_bool_array,
    axis: Optional[int] = None,
    bitorder: Optional[Literal["big", "little"]] = "big",
):  # params ['a'] ['axis', 'bitorder'] []
    res = numpy.packbits(
        a=a,
        axis=axis,
        bitorder=bitorder,
    )
    return res


@fn.NodeDecorator(
    node_id="np.unpackbits",
    name="unpackbits",
    outputs=[{"name": "unpacked", "type": "bitarray"}],
)
@wraps(numpy.unpackbits)
def unpackbits(
    a: bitarray,
    axis: Optional[int] = None,
    count: Optional[int] = None,
    bitorder: Optional[Literal["big", "little"]] = "big",
):  # params ['a'] ['axis', 'count', 'bitorder'] []
    res = numpy.unpackbits(
        a=a,
        axis=axis,
        count=count,
        bitorder=bitorder,
    )
    return res


@fn.NodeDecorator(
    node_id="np.binary_repr",
    name="binary_repr",
    outputs=[{"name": "bin", "type": "str"}],
)
@wraps(numpy.binary_repr)
def binary_repr(
    num: int,
    width: Optional[int] = None,
):  # params ['num'] ['width'] []
    res = numpy.binary_repr(
        num=num,
        width=width,
    )
    return res


@fn.NodeDecorator(
    node_id="np.dot",
    name="dot",
    outputs=[{"name": "output", "type": "ndarray"}],
)
@wraps(numpy.dot)
def dot(
    a: array_like,
    b: array_like,
    # out: Optional[ndarray] = None,
):  # params ['a', 'b'] ['out'] []
    res = numpy.dot(
        a=a,
        b=b,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.vdot",
    name="vdot",
    outputs=[{"name": "output", "type": "ndarray"}],
)
@wraps(numpy.vdot)
def vdot(
    a: array_like,
    b: array_like,
):  # params ['a', 'b'] [] []
    res = numpy.vdot(
        a=a,
        b=b,
    )
    return res


@fn.NodeDecorator(
    node_id="np.inner",
    name="inner",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.inner)
def inner(
    a: array_like,
    b: array_like,
):  # params ['a', 'b'] [] []
    res = numpy.inner(
        a=a,
        b=b,
    )
    return res


@fn.NodeDecorator(
    node_id="np.outer",
    name="outer",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.outer)
def outer(
    a: ndarray,
    b: ndarray,
    # out: Optional[ndarray] = None,
):  # params ['a', 'b'] ['out'] []
    res = numpy.outer(
        a=a,
        b=b,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.matmul",
    name="matmul",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.matmul)
def matmul(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'casting', 'order'] []
    res = numpy.matmul(
        x1=x1,
        x2=x2,
        # out=out,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tensordot",
    name="tensordot",
    outputs=[{"name": "output", "type": "ndarray"}],
)
@wraps(numpy.tensordot)
def tensordot(
    a: array_like,
    b: array_like,
    axes: axis_like = 2,
):  # params ['a', 'b'] ['axes'] []
    res = numpy.tensordot(
        a=a,
        b=b,
        axes=axes,
    )
    return res


@fn.NodeDecorator(
    node_id="np.einsum",
    name="einsum",
    outputs=[{"name": "output", "type": "ndarray"}],
)
@wraps(numpy.einsum)
def einsum(
    subscripts: str,
    operands: List[array_like],
    # out: Optional[ndarray] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # order: OrderKACF = "K",
    # casting: casting_literal = "safe",
    optimize: Optional[Literal[False, True, "greedy", "optimal"]] = False,
):  # params ['subscripts'] ['operands', 'out', 'dtype'] []
    res = numpy.einsum(
        subscripts,
        *operands,
        # out=out,
        dtype=dtype_from_name(dtype),
        # order=order,
        # casting=casting,
        optimize=optimize,
    )
    return res


@fn.NodeDecorator(
    node_id="np.einsum_path",
    name="einsum_path",
    outputs=[
        {"name": "path", "type": "List[Tuple]"},
        {"name": "string_repr", "type": "str"},
    ],
)
@wraps(numpy.einsum_path)
def einsum_path(
    subscripts: str,
    operands: List[array_like],
    optimize: Optional[Literal[False, True, "greedy", "optimal"]] = "greedy",
):  # params ['subscripts'] ['operands', 'optimize'] []
    res = numpy.einsum_path(
        subscripts,
        *operands,
        optimize=optimize,
    )
    return res


@fn.NodeDecorator(
    node_id="np.kron",
    name="kron",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.kron)
def kron(
    a: array_like,
    b: array_like,
):  # params ['a', 'b'] [] []
    res = numpy.kron(
        a=a,
        b=b,
    )
    return res


@fn.NodeDecorator(
    node_id="np.trace",
    name="trace",
    outputs=[{"name": "sum_along_diagonals", "type": "ndarray"}],
)
@wraps(numpy.trace)
def trace(
    a: array_like,
    offset: Optional[int] = 0,
    axis1: Optional[int] = 0,
    axis2: Optional[int] = 1,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['offset', 'axis1', 'axis2', 'dtype', 'out'] []
    res = numpy.trace(
        a=a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype_from_name(dtype),
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.all",
    name="all",
    outputs=[{"name": "all", "type": "bool_or_bool_array"}],
)
@wraps(numpy.all)
def all(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'where'] []
    res = numpy.all(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.any",
    name="any",
    outputs=[{"name": "any", "type": "bool_or_bool_array"}],
)
@wraps(numpy.any)
def any(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'where'] []
    res = numpy.any(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isfinite",
    name="isfinite",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.isfinite)
def isfinite(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.isfinite(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isinf",
    name="isinf",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.isinf)
def isinf(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.isinf(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isnan",
    name="isnan",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.isnan)
def isnan(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.isnan(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isnat",
    name="isnat",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.isnat)
def isnat(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.isnat(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isneginf",
    name="isneginf",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.isneginf)
def isneginf(
    x: array_like,
    out: Optional[array_like] = None,
):  # params ['x'] ['out'] []
    res = numpy.isneginf(
        x=x,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isposinf",
    name="isposinf",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.isposinf)
def isposinf(
    x: array_like,
    out: Optional[array_like] = None,
):  # params ['x'] ['out'] []
    res = numpy.isposinf(
        x=x,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.iscomplex",
    name="iscomplex",
    outputs=[{"name": "out", "type": "bool_or_bool_array"}],
)
@wraps(numpy.iscomplex)
def iscomplex(
    x: array_like,
):  # params ['x'] [] []
    res = numpy.iscomplex(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.iscomplexobj",
    name="iscomplexobj",
    outputs=[{"name": "iscomplexobj", "type": "bool"}],
)
@wraps(numpy.iscomplexobj)
def iscomplexobj(
    x: ndarray_or_number,
):  # params ['x'] [] []
    res = numpy.iscomplexobj(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isfortran",
    name="isfortran",
    outputs=[{"name": "isfortran", "type": "bool"}],
)
@wraps(numpy.isfortran)
def isfortran(
    a: ndarray,
):  # params ['a'] [] []
    res = numpy.isfortran(
        a=a,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isreal",
    name="isreal",
    outputs=[{"name": "out", "type": "bool_or_bool_array"}],
)
@wraps(numpy.isreal)
def isreal(
    x: array_like,
):  # params ['x'] [] []
    res = numpy.isreal(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isrealobj",
    name="isrealobj",
    outputs=[{"name": "y", "type": "bool"}],
)
@wraps(numpy.isrealobj)
def isrealobj(
    x: ndarray_or_number,
):  # params ['x'] [] []
    res = numpy.isrealobj(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isscalar",
    name="isscalar",
    outputs=[{"name": "val", "type": "bool"}],
)
@wraps(numpy.isscalar)
def isscalar(
    element: ndarray_or_number,
):  # params ['element'] [] []
    res = numpy.isscalar(
        element=element,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logical_and",
    name="logical_and",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.logical_and)
def logical_and(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.logical_and(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logical_or",
    name="logical_or",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.logical_or)
def logical_or(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.logical_or(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logical_not",
    name="logical_not",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.logical_not)
def logical_not(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting'] []
    res = numpy.logical_not(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logical_xor",
    name="logical_xor",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.logical_xor)
def logical_xor(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.logical_xor(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.allclose",
    name="allclose",
    outputs=[{"name": "allclose", "type": "bool"}],
)
@wraps(numpy.allclose)
def allclose(
    a: array_like,
    b: array_like,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):  # params ['a', 'b'] ['rtol', 'atol', 'equal_nan'] []
    res = numpy.allclose(
        a=a,
        b=b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return res


@fn.NodeDecorator(
    node_id="np.isclose",
    name="isclose",
    outputs=[{"name": "y", "type": "array_like"}],
)
@wraps(numpy.isclose)
def isclose(
    a: array_like,
    b: array_like,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):  # params ['a', 'b'] ['rtol', 'atol', 'equal_nan'] []
    res = numpy.isclose(
        a=a,
        b=b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return res


@fn.NodeDecorator(
    node_id="np.array_equal",
    name="array_equal",
    outputs=[{"name": "b", "type": "bool"}],
)
@wraps(numpy.array_equal)
def array_equal(
    a1: array_like,
    a2: array_like,
    equal_nan: bool = False,
):  # params ['a1', 'a2'] ['equal_nan'] []
    res = numpy.array_equal(
        a1=a1,
        a2=a2,
        equal_nan=equal_nan,
    )
    return res


@fn.NodeDecorator(
    node_id="np.array_equiv",
    name="array_equiv",
    outputs=[{"name": "out", "type": "bool"}],
)
@wraps(numpy.array_equiv)
def array_equiv(
    a1: array_like,
    a2: array_like,
):  # params ['a1', 'a2'] [] []
    res = numpy.array_equiv(
        a1=a1,
        a2=a2,
    )
    return res


@fn.NodeDecorator(
    node_id="np.greater",
    name="greater",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.greater)
def greater(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.greater(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.greater_equal",
    name="greater_equal",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.greater_equal)
def greater_equal(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.greater_equal(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.less",
    name="less",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.less)
def less(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.less(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.less_equal",
    name="less_equal",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.less_equal)
def less_equal(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.less_equal(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.equal",
    name="equal",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.equal)
def equal(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.equal(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.not_equal",
    name="not_equal",
    outputs=[{"name": "y", "type": "bool_or_bool_array"}],
)
@wraps(numpy.not_equal)
def not_equal(
    x1: ndarray_or_scalar,
    x2: ndarray_or_scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    return numpy.not_equal(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
    )


@fn.NodeDecorator(
    node_id="np.sin",
    name="sin",
    outputs=[{"name": "y", "type": "array_like"}],
)
@wraps(numpy.sin)
def sin(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.sin(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cos",
    name="cos",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.cos)
def cos(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.cos(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tan",
    name="tan",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.tan)
def tan(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.tan(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arcsin",
    name="arcsin",
    outputs=[{"name": "angle", "type": "ndarray"}],
)
@wraps(numpy.arcsin)
def arcsin(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arcsin(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arccos",
    name="arccos",
    outputs=[{"name": "angle", "type": "ndarray"}],
)
@wraps(numpy.arccos)
def arccos(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arccos(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arctan",
    name="arctan",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.arctan)
def arctan(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arctan(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.hypot",
    name="hypot",
    outputs=[{"name": "z", "type": "ndarray"}],
)
@wraps(numpy.hypot)
def hypot(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.hypot(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arctan2",
    name="arctan2",
    outputs=[{"name": "angle", "type": "ndarray"}],
)
@wraps(numpy.arctan2)
def arctan2(
    x1: real_array,
    x2: real_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.arctan2(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.degrees",
    name="degrees",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.degrees)
def degrees(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.degrees(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.radians",
    name="radians",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.radians)
def radians(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.radians(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.unwrap",
    name="unwrap",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.unwrap)
def unwrap(
    p: array_like,
    discont: Optional[float] = None,
    axis: Optional[int] = -1,
    period: Optional[float] = 6.283185307179586,
):  # params ['p'] ['discont', 'axis', 'period'] []
    res = numpy.unwrap(
        p=p,
        discont=discont,
        axis=axis,
        period=period,
    )
    return res


@fn.NodeDecorator(
    node_id="np.deg2rad",
    name="deg2rad",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.deg2rad)
def deg2rad(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.deg2rad(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.rad2deg",
    name="rad2deg",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.rad2deg)
def rad2deg(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.rad2deg(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.sinh",
    name="sinh",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.sinh)
def sinh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.sinh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cosh",
    name="cosh",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.cosh)
def cosh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.cosh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.tanh",
    name="tanh",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.tanh)
def tanh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.tanh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arcsinh",
    name="arcsinh",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.arcsinh)
def arcsinh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arcsinh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arccosh",
    name="arccosh",
    outputs=[{"name": "arccosh", "type": "ndarray"}],
)
@wraps(numpy.arccosh)
def arccosh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arccosh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.arctanh",
    name="arctanh",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.arctanh)
def arctanh(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.arctanh(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.round",
    name="round",
    outputs=[{"name": "rounded_array", "type": "ndarray"}],
)
@wraps(numpy.round)
def round(
    a: array_like,
    decimals: Optional[int] = 0,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['decimals', 'out'] []
    res = numpy.round(
        a=a,
        decimals=decimals,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.rint",
    name="rint",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.rint)
def rint(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.rint(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fix",
    name="fix",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.fix)
def fix(
    x: array_like,
    # out: Optional[ndarray] = None,
):  # params ['x'] ['out'] []
    res = numpy.fix(
        x=x,
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.floor",
    name="floor",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.floor)
def floor(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.floor(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ceil",
    name="ceil",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ceil)
def ceil(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.ceil(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.trunc",
    name="trunc",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.trunc)
def trunc(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.trunc(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.prod",
    name="prod",
    outputs=[{"name": "product_along_axis", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.prod)
def prod(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims'] []
    res = numpy.prod(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.sum",
    name="sum",
    outputs=[{"name": "sum_along_axis", "type": "ndarray"}],
)
@wraps(numpy.sum)
def sum(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims'] []
    res = numpy.sum(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanprod",
    name="nanprod",
    outputs=[{"name": "nanprod", "type": "ndarray"}],
)
@wraps(numpy.nanprod)
def nanprod(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims'] []
    res = numpy.nanprod(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nansum",
    name="nansum",
    outputs=[{"name": "nansum", "type": "ndarray."}],
)
@wraps(numpy.nansum)
def nansum(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims'] []
    res = numpy.nansum(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cumprod",
    name="cumprod",
    outputs=[{"name": "cumprod", "type": "ndarray"}],
)
@wraps(numpy.cumprod)
def cumprod(
    a: array_like,
    axis: Optional[int] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['axis', 'dtype', 'out'] []
    res = numpy.cumprod(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cumsum",
    name="cumsum",
    outputs=[{"name": "cumsum_along_axis", "type": "ndarray."}],
)
@wraps(numpy.cumsum)
def cumsum(
    a: array_like,
    axis: Optional[int] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['axis', 'dtype', 'out'] []
    res = numpy.cumsum(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nancumprod",
    name="nancumprod",
    outputs=[{"name": "nancumprod", "type": "ndarray"}],
)
@wraps(numpy.nancumprod)
def nancumprod(
    a: array_like,
    axis: Optional[int] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['axis', 'dtype', 'out'] []
    res = numpy.nancumprod(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nancumsum",
    name="nancumsum",
    outputs=[{"name": "nancumsum", "type": "ndarray."}],
)
@wraps(numpy.nancumsum)
def nancumsum(
    a: array_like,
    axis: Optional[int] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
):  # params ['a'] ['axis', 'dtype', 'out'] []
    res = numpy.nancumsum(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
    )
    return res


@fn.NodeDecorator(
    node_id="np.diff",
    name="diff",
    outputs=[{"name": "diff", "type": "ndarray"}],
)
@wraps(numpy.diff)
def diff(
    a: array_like,
    n: Optional[int] = 1,
    axis: Optional[int] = -1,
    prepend: Optional[array_like] = NoValue,
    append: Optional[array_like] = NoValue,
):  # params ['a'] ['n', 'axis', 'prepend', 'append'] []
    res = numpy.diff(
        a=a,
        n=n,
        axis=axis,
        prepend=prepend,
        append=append,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ediff1d",
    name="ediff1d",
    outputs=[{"name": "ediff1d", "type": "ndarray"}],
)
@wraps(numpy.ediff1d)
def ediff1d(
    ary: array_like,
    to_end: Optional[array_like] = None,
    to_begin: Optional[array_like] = None,
):  # params ['ary'] ['to_end', 'to_begin'] []
    res = numpy.ediff1d(
        ary=ary,
        to_end=to_end,
        to_begin=to_begin,
    )
    return res


@fn.NodeDecorator(
    node_id="np.gradient",
    name="gradient",
    outputs=[{"name": "gradient", "type": "Union[ndarray, List[ndarray]]"}],
)
@wraps(numpy.gradient)
def gradient(
    f: array_like,
    varargs: List[ndarray_or_scalar],
    edge_order: Optional[Literal[1, 2]] = 1,
    axis: Optional[axis_like] = None,
):  # params ['f'] ['varargs', 'axis', 'edge_order'] []
    res = numpy.gradient(
        f,
        *varargs,
        edge_order=edge_order,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cross",
    name="cross",
    outputs=[{"name": "c", "type": "ndarray"}],
)
@wraps(numpy.cross)
def cross(
    a: array_like,
    b: array_like,
    axisa: Optional[int] = -1,
    axisb: Optional[int] = -1,
    axisc: Optional[int] = -1,
    axis: Optional[int] = None,
):  # params ['a', 'b'] ['axisa', 'axisb', 'axisc', 'axis'] []
    res = numpy.cross(
        a=a,
        b=b,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.trapz",
    name="trapz",
    outputs=[{"name": "trapz", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.trapz)
def trapz(
    y: array_like,
    x: Optional[array_like] = None,
    dx: Optional[scalar] = 1.0,
    axis: Optional[int] = -1,
):  # params ['y'] ['x', 'dx', 'axis'] []
    res = numpy.trapz(
        y=y,
        x=x,
        dx=dx,
        axis=axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.exp",
    name="exp",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.exp)
def exp(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.exp(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.expm1",
    name="expm1",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.expm1)
def expm1(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.expm1(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.exp2",
    name="exp2",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.exp2)
def exp2(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.exp2(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.log",
    name="log",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.log)
def log(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.log(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.log10",
    name="log10",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.log10)
def log10(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.log10(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.log2",
    name="log2",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.log2)
def log2(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.log2(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.log1p",
    name="log1p",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.log1p)
def log1p(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.log1p(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logaddexp",
    name="logaddexp",
    outputs=[{"name": "result", "type": "ndarray"}],
)
@wraps(numpy.logaddexp)
def logaddexp(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.logaddexp(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.logaddexp2",
    name="logaddexp2",
    outputs=[{"name": "result", "type": "ndarray"}],
)
@wraps(numpy.logaddexp2)
def logaddexp2(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.logaddexp2(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.i0",
    name="i0",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.i0)
def i0(
    x: ndarray,
):  # params ['x'] [] []
    res = numpy.i0(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.sinc",
    name="sinc",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.sinc)
def sinc(
    x: ndarray,
):  # params ['x'] [] []
    res = numpy.sinc(
        x=x,
    )
    return res


@fn.NodeDecorator(
    node_id="np.signbit",
    name="signbit",
    outputs=[{"name": "result", "type": "ndarray"}],
)
@wraps(numpy.signbit)
def signbit(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.signbit(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.copysign",
    name="copysign",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.copysign)
def copysign(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.copysign(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.frexp",
    name="frexp",
    outputs=[
        {"name": "mantissa", "type": "ndarray"},
        {"name": "exponent", "type": "ndarray"},
    ],
)
@wraps(numpy.frexp)
def frexp(
    x: array_like,
    out1: Union[ndarray, None],
    out2: Union[ndarray, None],
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):
    res = numpy.frexp(
        x=x,
        out1=out1,
        out2=out2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ldexp",
    name="ldexp",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ldexp)
def ldexp(
    x1: array_like,
    x2: array_like,
    int,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.ldexp(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nextafter",
    name="nextafter",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nextafter)
def nextafter(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.nextafter(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.spacing",
    name="spacing",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.spacing)
def spacing(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.spacing(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.lcm",
    name="lcm",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.lcm)
def lcm(
    x1: int_array,
    x2: int_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting', 'order'] []
    res = numpy.lcm(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.gcd",
    name="gcd",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.gcd)
def gcd(
    x1: int_array,
    x2: int_array,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting', 'order'] []
    res = numpy.gcd(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.add",
    name="add",
    outputs=[{"name": "add", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.add)
def add(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting', 'order'] []
    res = numpy.add(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.reciprocal",
    name="reciprocal",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.reciprocal)
def reciprocal(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting'] []
    res = numpy.reciprocal(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.positive",
    name="positive",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.positive)
def positive(
    x: array_like or scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.positive(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.negative",
    name="negative",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.negative)
def negative(
    x: array_like or scalar,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.negative(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.multiply",
    name="multiply",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.multiply)
def multiply(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.multiply(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.divide",
    name="divide",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.divide)
def divide(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.divide(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.power",
    name="power",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.power)
def power(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.power(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.subtract",
    name="subtract",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.subtract)
def subtract(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.subtract(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.true_divide",
    name="true_divide",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.true_divide)
def true_divide(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.true_divide(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.floor_divide",
    name="floor_divide",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.floor_divide)
def floor_divide(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.floor_divide(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.float_power",
    name="float_power",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.float_power)
def float_power(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where'] []
    res = numpy.float_power(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fmod",
    name="fmod",
    outputs=[{"name": "y", "type": "array_like"}],
)
@wraps(numpy.fmod)
def fmod(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.fmod(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.mod",
    name="mod",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.mod)
def mod(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting', 'order'] []
    res = numpy.mod(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.modf",
    name="modf",
    outputs=[{"name": "y1", "type": "ndarray"}, {"name": "y2", "type": "ndarray"}],
)
@wraps(numpy.modf)
def modf(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    out1: Optional[ndarray] = None,
    out2: Optional[ndarray] = None,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out'] ['out1', 'out2']
    res = numpy.modf(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        out1=out1,
        out2=out2,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.remainder",
    name="remainder",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.remainder)
def remainder(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.remainder(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.divmod",
    name="divmod",
    outputs=[{"name": "out1", "type": "ndarray"}, {"name": "out2", "type": "ndarray"}],
)
@wraps(numpy.divmod)
def divmod(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    out1: Optional[ndarray] = None,
    out2: Optional[ndarray] = None,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out'] ['out1', 'out2']
    res = numpy.divmod(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        out1=out1,
        out2=out2,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.angle",
    name="angle",
    outputs=[{"name": "angle", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.angle)
def angle(
    z: array_like,
    deg: Optional[bool] = False,
):  # params ['z'] ['deg'] []
    res = numpy.angle(
        z=z,
        deg=deg,
    )
    return res


@fn.NodeDecorator(
    node_id="np.real",
    name="real",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.real)
def real(
    val: array_like,
):  # params ['val'] [] []
    res = numpy.real(
        val=val,
    )
    return res


@fn.NodeDecorator(
    node_id="np.imag",
    name="imag",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.imag)
def imag(
    val: array_like,
):  # params ['val'] [] []
    res = numpy.imag(
        val=val,
    )
    return res


@fn.NodeDecorator(
    node_id="np.conj",
    name="conj",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.conj)
def conj(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.conj(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.conjugate",
    name="conjugate",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.conjugate)
def conjugate(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting'] []
    res = numpy.conjugate(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.maximum",
    name="maximum",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.maximum)
def maximum(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.maximum(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.max",
    name="max",
    outputs=[{"name": "max", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.max)
def max(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'initial', 'where'] []
    res = numpy.max(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fmax",
    name="fmax",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.fmax)
def fmax(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.fmax(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanmax",
    name="nanmax",
    outputs=[{"name": "nanmax", "type": "ndarray"}],
)
@wraps(numpy.nanmax)
def nanmax(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'initial', 'where'] []
    res = numpy.nanmax(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.minimum",
    name="minimum",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.minimum)
def minimum(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.minimum(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.min",
    name="min",
    outputs=[{"name": "min", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.min)
def min(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'initial', 'where'] []
    res = numpy.min(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fmin",
    name="fmin",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.fmin)
def fmin(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.fmin(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanmin",
    name="nanmin",
    outputs=[{"name": "nanmin", "type": "ndarray"}],
)
@wraps(numpy.nanmin)
def nanmin(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    initial: Optional[scalar] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims', 'initial', 'where'] []
    res = numpy.nanmin(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
        initial=initial,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.convolve",
    name="convolve",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.convolve)
def convolve(
    a: ndarray,
    v: ndarray,
    mode: Optional[Literal["full", "valid", "same"]] = "full",
):  # params ['a', 'v'] ['mode'] []
    res = numpy.convolve(
        a=a,
        v=v,
        mode=mode,
    )
    return res


@fn.NodeDecorator(
    node_id="np.clip",
    name="clip",
    outputs=[{"name": "clipped_array", "type": "ndarray"}],
)
@wraps(numpy.clip)
def clip(
    a: array_like,
    a_min: array_like or None,
    a_max: array_like or None,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['a', 'a_min', 'a_max'] ['out'] []
    res = numpy.clip(
        a=a,
        a_min=a_min,
        a_max=a_max,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.sqrt",
    name="sqrt",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.sqrt)
def sqrt(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.sqrt(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.cbrt",
    name="cbrt",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.cbrt)
def cbrt(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.cbrt(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.square",
    name="square",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.square)
def square(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.square(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.absolute",
    name="absolute",
    outputs=[{"name": "absolute", "type": "ndarray"}],
)
@wraps(numpy.absolute)
def absolute(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.absolute(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.fabs",
    name="fabs",
    outputs=[{"name": "y", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.fabs)
def fabs(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.fabs(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.sign",
    name="sign",
    outputs=[{"name": "y", "type": "ndarray"}],
)
@wraps(numpy.sign)
def sign(
    x: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x'] ['out', 'where', 'casting', 'order'] []
    res = numpy.sign(
        x=x,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.heaviside",
    name="heaviside",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.heaviside)
def heaviside(
    x1: array_like,
    x2: array_like,
    # out: Optional[ndarray] = None,
    # where: Union[bool_array, bool] = True,
    # casting: casting_literal = "same_kind",
    # order: OrderKACF = "K",
    dtype: Optional[DTYPE_ENUM] = None,
    # subok: bool = True,
    # signature: Any = None,
    # extobj: Any = None,
):  # params ['x1', 'x2'] ['out', 'where', 'casting'] []
    res = numpy.heaviside(
        x1=x1,
        x2=x2,
        # out=out,
        # where=where,
        # casting=casting,
        # order=order,
        dtype=dtype_from_name(dtype),
        # subok=subok,
        # signature=signature,
        # extobj=extobj,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nan_to_num",
    name="nan_to_num",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.nan_to_num)
def nan_to_num(
    x: ndarray_or_scalar,
    copy: Optional[bool] = True,
    nan: Union[int, float] = 0.0,
    posinf: Union[int, float, None] = None,
    neginf: Union[int, float, None] = None,
):  # params ['x'] ['copy', 'nan', 'posinf', 'neginf'] []
    res = numpy.nan_to_num(
        x=x,
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )
    return res


@fn.NodeDecorator(
    node_id="np.real_if_close",
    name="real_if_close",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.real_if_close)
def real_if_close(
    a: array_like,
    tol: float = 100,
):  # params ['a'] ['tol'] []
    res = numpy.real_if_close(
        a=a,
        tol=tol,
    )
    return res


@fn.NodeDecorator(
    node_id="np.interp",
    name="interp",
    outputs=[{"name": "y", "type": "ndarray_or_number"}],
)
@wraps(numpy.interp)
def interp(
    x: array_like,
    xp: Sequence[float],
    fp: Sequence[float],
    left: Optional[float] = None,
    right: Optional[float] = None,
    period: Optional[float] = None,
):  # params ['x', 'xp', 'fp'] ['left', 'right', 'period'] []
    res = numpy.interp(
        x=x,
        xp=xp,
        fp=fp,
        left=left,
        right=right,
        period=period,
    )
    return res


@fn.NodeDecorator(
    node_id="np.ptp",
    name="ptp",
    outputs=[{"name": "ptp", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ptp)
def ptp(
    a: array_like,
    axis: Optional[axis_like] = None,
    out: array_like = None,
    keepdims: Optional[bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'keepdims'] []
    res = numpy.ptp(
        a=a,
        axis=axis,
        # out=out,
        keepdims=keepdims,
    )
    return res


@fn.NodeDecorator(
    node_id="np.percentile",
    name="percentile",
    outputs=[{"name": "percentile", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.percentile)
def percentile(
    a: ndarray,
    q: ndarray,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    method: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = None,
):  # params ['a', 'q'] ['axis', 'out'] []
    res = numpy.percentile(
        a=a,
        q=q,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanpercentile",
    name="nanpercentile",
    outputs=[{"name": "percentile", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nanpercentile)
def nanpercentile(
    a: array_like,
    q: ndarray,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    method: Optional[str] = "linear",
    keepdims: Optional[bool] = NoValue,
    interpolation: Optional[str] = None,
):  # params ['a', 'q'] ['axis', 'out'] []
    res = numpy.nanpercentile(
        a=a,
        q=q,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return res


@fn.NodeDecorator(
    node_id="np.quantile",
    name="quantile",
    outputs=[{"name": "quantile", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.quantile)
def quantile(
    a: ndarray,
    q: ndarray,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    method: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = None,
):  # params ['a', 'q'] ['axis', 'out', 'overwrite_input'] []
    res = numpy.quantile(
        a=a,
        q=q,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanquantile",
    name="nanquantile",
    outputs=[{"name": "quantile", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nanquantile)
def nanquantile(
    a: array_like,
    q: ndarray,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    method: Optional[str] = "linear",
    keepdims: Optional[bool] = NoValue,
    interpolation: Optional[str] = None,
):  # params ['a', 'q'] ['axis', 'out'] []
    res = numpy.nanquantile(
        a=a,
        q=q,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return res


@fn.NodeDecorator(
    node_id="np.median",
    name="median",
    outputs=[{"name": "median", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.median)
def median(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    keepdims: Optional[bool] = False,
):  # params ['a'] ['axis', 'out', 'overwrite_input', 'keepdims'] []
    res = numpy.median(
        a=a,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
    )
    return res


@fn.NodeDecorator(
    node_id="np.average",
    name="average",
    outputs=[
        {"name": "retval", "type": "ndarray_or_scalar"},
        {"name": "sum_of_weights", "type": "ndarray_or_scalar"},
    ],
)
@wraps(numpy.average)
def average(
    a: array_like,
    axis: Optional[axis_like] = None,
    weights: Optional[array_like] = None,
    # returned: Optional[bool] = False,
    # keepdims: Optional[bool] = NoValue,
):  # params ['a'] ['axis', 'weights', 'returned', 'keepdims'] []
    res = numpy.average(
        a=a,
        axis=axis,
        weights=weights,
        # returned=returned,
        # keepdims=keepdims,
    )
    return res


@fn.NodeDecorator(
    node_id="np.mean",
    name="mean",
    outputs=[{"name": "m", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.mean)
def mean(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims', 'where'] []
    res = numpy.mean(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.std",
    name="std",
    outputs=[{"name": "standard_deviation", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.std)
def std(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    ddof: Optional[int] = 0,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'ddof', 'keepdims', 'where'] []
    res = numpy.std(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        ddof=ddof,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.var",
    name="var",
    outputs=[{"name": "variance", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.var)
def var(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    ddof: Optional[int] = 0,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'ddof', 'keepdims', 'where'] []
    res = numpy.var(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        ddof=ddof,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanmedian",
    name="nanmedian",
    outputs=[{"name": "median", "type": "ndarray"}],
)
@wraps(numpy.nanmedian)
def nanmedian(
    a: array_like,
    axis: Optional[axis_like] = None,
    # out: Optional[ndarray] = None,
    overwrite_input: Optional[bool] = False,
    keepdims: Optional[bool] = NoValue,
):  # params ['a'] ['axis', 'out', 'overwrite_input'] []
    res = numpy.nanmedian(
        a=a,
        axis=axis,
        # out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanmean",
    name="nanmean",
    outputs=[{"name": "m", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nanmean)
def nanmean(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'keepdims', 'where'] []
    res = numpy.nanmean(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanstd",
    name="nanstd",
    outputs=[{"name": "standard_deviation", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nanstd)
def nanstd(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    ddof: Optional[int] = 0,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'ddof'] []
    res = numpy.nanstd(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        ddof=ddof,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.nanvar",
    name="nanvar",
    outputs=[{"name": "variance", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.nanvar)
def nanvar(
    a: array_like,
    axis: Optional[axis_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
    # out: Optional[ndarray] = None,
    ddof: Optional[int] = 0,
    keepdims: Optional[bool] = NoValue,
    # where: Union[bool_array, bool] = NoValue,
):  # params ['a'] ['axis', 'dtype', 'out', 'ddof'] []
    res = numpy.nanvar(
        a=a,
        axis=axis,
        dtype=dtype_from_name(dtype),
        # out=out,
        ddof=ddof,
        keepdims=keepdims,
        # where=where,
    )
    return res


@fn.NodeDecorator(
    node_id="np.corrcoef",
    name="corrcoef",
    outputs=[{"name": "R", "type": "ndarray"}],
)
@wraps(numpy.corrcoef)
def corrcoef(
    x: array_like,
    y: Optional[array_like] = None,
    rowvar: Optional[bool] = True,
    dtype: Optional[DTYPE_ENUM] = None,
):  # params ['x'] ['y', 'rowvar', 'bias', 'ddof', 'dtype'] []
    res = numpy.corrcoef(
        x=x,
        y=y,
        rowvar=rowvar,
        dtype=dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.correlate",
    name="correlate",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.correlate)
def correlate(
    a: array_like,
    v: array_like,
    mode: Optional[Literal["valid", "same", "full"]] = "valid",
):
    res = numpy.correlate(a=a, v=v, mode=mode)
    return res


@fn.NodeDecorator(
    node_id="np.cov",
    name="cov",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.cov)
def cov(
    m: array_like,
    y: Optional[array_like] = None,
    rowvar: Optional[bool] = True,
    bias: Optional[bool] = False,
    ddof: Optional[int] = None,
    fweights: Union[array_like, int, None] = None,
    aweights: Optional[array_like] = None,
    dtype: Optional[DTYPE_ENUM] = None,
):  # params ['m'] ['y', 'rowvar', 'bias', 'ddof', 'fweights'] []
    res = numpy.cov(
        m=m,
        y=y,
        rowvar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        dtype=dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.histogram",
    name="histogram",
    outputs=[
        {"name": "hist", "type": "ndarray"},
        {"name": "bin_edges", "type": "ndarray"},
    ],
)
@wraps(numpy.histogram)
def histogram(
    a: array_like,
    bins: Union[int, Union[Sequence[scalar], Sequence[str]]] = 10,
    range: Optional[Tuple[float, float]] = None,
    weights: Optional[array_like] = None,
    density: Optional[bool] = None,
):  # params ['a'] ['bins', 'range', 'density', 'weights'] []
    res = numpy.histogram(
        a=a,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
    )
    return res


@fn.NodeDecorator(
    node_id="np.histogram2d",
    name="histogram2d",
    outputs=[
        {"name": "H", "type": "ndarray"},
        {"name": "xedges", "type": "ndarray"},
        {"name": "yedges", "type": "ndarray"},
    ],
)
@wraps(numpy.histogram2d)
def histogram2d(
    x: array_like,
    y: array_like,
    bins: Union[int, ndarray, Tuple[Union[int, ndarray], Union[int, ndarray]]] = 10,
    range: Optional[array_like] = None,
    density: Optional[bool] = None,
    weights: Optional[array_like] = None,
):  # params ['x', 'y'] ['bins', 'range', 'density'] []
    res = numpy.histogram2d(
        x=x,
        y=y,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )
    return res


@fn.NodeDecorator(
    node_id="np.histogramdd",
    name="histogramdd",
    outputs=[{"name": "H", "type": "ndarray"}, {"name": "edges", "type": "list"}],
)
@wraps(numpy.histogramdd)
def histogramdd(
    sample: array_like,
    bins: Union[int, Sequence[int]] = 10,
    range: Optional[Sequence[Tuple[float, float]]] = None,
    density: Optional[bool] = None,
    weights: Optional[ndarray] = None,
):  # params ['sample'] ['bins', 'range', 'density'] []
    res = numpy.histogramdd(
        sample=sample,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )
    return res


@fn.NodeDecorator(
    node_id="np.bincount",
    name="bincount",
    outputs=[{"name": "out", "type": "int_array"}],
)
@wraps(numpy.bincount)
def bincount(
    x: bitarray,
    weights: Optional[array_like] = None,
    minlength: Optional[int] = 0,
):  # params ['x'] ['weights', 'minlength'] []
    res = numpy.bincount(
        x=x,
        weights=weights,
        minlength=minlength,
    )
    return res


@fn.NodeDecorator(
    node_id="np.histogram_bin_edges",
    name="histogram_bin_edges",
    outputs=[{"name": "bin_edges", "type": "ndarray"}],
)
@wraps(numpy.histogram_bin_edges)
def histogram_bin_edges(
    a: array_like,
    bins: Union[int, Union[Sequence[scalar], Sequence[str]]] = 10,
    range: Optional[Tuple[float, float]] = None,
    weights: Optional[array_like] = None,
):  # params ['a'] ['bins', 'range', 'weights'] []
    res = numpy.histogram_bin_edges(
        a=a,
        bins=bins,
        range=range,
        weights=weights,
    )
    return res


@fn.NodeDecorator(
    node_id="np.digitize",
    name="digitize",
    outputs=[{"name": "indices", "type": "int_array"}],
)
@wraps(numpy.digitize)
def digitize(
    x: array_like,
    bins: array_like,
    right: Optional[bool] = False,
):  # params ['x', 'bins'] ['right'] []
    res = numpy.digitize(
        x=x,
        bins=bins,
        right=right,
    )
    return res


@fn.NodeDecorator(
    node_id="np.where",
    name="where",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.where)
def where(
    condition: Union[array_like],
    x: Optional[array_like] = None,
    y: Optional[array_like] = None,
):
    res = numpy.where(
        condition=condition,
        x=x,
        y=y,
    )
    return res
