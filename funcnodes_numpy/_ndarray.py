from functools import wraps
import numpy
import funcnodes as fn
from typing import List, Literal, Optional
from ._types import ndarray, shape_like, axis_like, int_or_int_array
from ._dtypes import DTYPE_ENUM, dtype_from_name


@fn.NodeDecorator(
    node_id="np.a.all",
    name="all",
    outputs=[{"name": "out", "type": "bool_or_bool_array"}],
)
@wraps(numpy.ndarray.all)
def all(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.all(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.any",
    name="any",
    outputs=[{"name": "out", "type": "bool_or_bool_array"}],
)
@wraps(numpy.ndarray.any)
def any(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.any(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.argmax",
    name="argmax",
    outputs=[{"name": "out", "type": "int_or_int_array"}],
)
@wraps(numpy.ndarray.argmax)
def argmax(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.argmax(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.argmin",
    name="argmin",
    outputs=[{"name": "out", "type": "int_or_int_array"}],
)
@wraps(numpy.ndarray.argmin)
def argmin(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.argmin(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.argpartition",
    name="argpartition",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.argpartition)
def argpartition(
    a: ndarray,
    kth: int_or_int_array,
    axis: axis_like = -1,
):
    res = a.argpartition(kth, axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.argsort",
    name="argsort",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.argsort)
def argsort(
    a: ndarray,
    axis: Optional[axis_like] = -1,
):
    res = a.argsort(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.astype",
    name="astype",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.astype)
def astype(a: ndarray, dtype: DTYPE_ENUM):
    res = a.astype(
        dtype_from_name(dtype),
    )
    return res


@fn.NodeDecorator(
    node_id="np.a.byteswap",
    name="byteswap",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.byteswap)
def byteswap(
    a: ndarray,
):
    res = a.byteswap(False)
    return res


@fn.NodeDecorator(
    node_id="np.a.choose",
    name="choose",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.choose)
def choose(
    a: ndarray,
    choices: ndarray,
):
    res = a.choose(choices)
    return res


@fn.NodeDecorator(
    node_id="np.a.clip",
    name="clip",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.clip)
def clip(
    a: ndarray,
    a_min: Optional[float] = None,
    a_max: Optional[float] = None,
):
    res = a.clip(a_min, a_max)
    return res


@fn.NodeDecorator(
    node_id="np.a.compress",
    name="compress",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.compress)
def compress(
    a: ndarray,
    condition: ndarray,
    axis: Optional[int] = None,
):
    res = a.compress(condition, axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.conj",
    name="conj",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.conj)
def conj(
    a: ndarray,
):
    res = a.conj()
    return res


@fn.NodeDecorator(
    node_id="np.a.conjugate",
    name="conjugate",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.conjugate)
def conjugate(
    a: ndarray,
):
    res = a.conjugate()
    return res


@fn.NodeDecorator(
    node_id="np.a.copy",
    name="copy",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.copy)
def copy(
    a: ndarray,
):
    res = a.copy()
    return res


@fn.NodeDecorator(
    node_id="np.a.cumprod",
    name="cumprod",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.cumprod)
def cumprod(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.cumprod(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.cumsum",
    name="cumsum",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.cumsum)
def cumsum(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.cumsum(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.diagonal",
    name="diagonal",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.diagonal)
def diagonal(
    a: ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
):
    res = a.diagonal(offset, axis1, axis2)
    return res


@fn.NodeDecorator(
    node_id="np.a.dot",
    name="dot",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.dot)
def dot(
    a: ndarray,
    b: ndarray,
):
    res = a.dot(b)
    return res


@fn.NodeDecorator(
    node_id="np.a.fill",
    name="fill",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.fill)
def fill(
    a: ndarray,
    value: float,
):
    res = a.fill(value)
    return res


@fn.NodeDecorator(
    node_id="np.a.flatten",
    name="flatten",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.flatten)
def flatten(
    a: ndarray,
):
    res = a.flatten()
    return res


@fn.NodeDecorator(
    node_id="np.a.getfield",
    name="getfield",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.getfield)
def getfield(
    a: ndarray,
    dtype: DTYPE_ENUM,
    offset: int = 0,
):
    res = a.getfield(dtype_from_name(dtype), offset)
    return res


@fn.NodeDecorator(
    node_id="np.a.item",
    name="item",
    outputs=[{"name": "out", "type": "scalar"}],
)
@wraps(numpy.ndarray.item)
def item(
    a: ndarray,
    pos: axis_like,
):
    pos = numpy.atleast_1d(pos)
    res = a.item(*pos)
    return res


@fn.NodeDecorator(
    node_id="np.a.itemset",
    name="itemset",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.itemset)
def itemset(
    a: ndarray,
    pos: axis_like,
    value: float,
):
    pos = numpy.atleast_1d(pos)
    res = a.itemset(*pos, value)
    return res


@fn.NodeDecorator(
    node_id="np.a.max",
    name="max",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.max)
def max(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.max(
        axis,
    )
    return res


@fn.NodeDecorator(
    node_id="np.a.mean",
    name="mean",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.mean)
def mean(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.mean(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.min",
    name="min",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.min)
def min(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.min(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.newbyteorder",
    name="newbyteorder",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.newbyteorder)
def newbyteorder(
    a: ndarray,
    new_order: Literal["S", "<", ">", "=", "|"] = "S",
):
    res = a.newbyteorder(new_order)
    return res


@fn.NodeDecorator(
    node_id="np.a.nonzero",
    name="nonzero",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.nonzero)
def nonzero(
    a: ndarray,
):
    res = a.nonzero()
    return res


@fn.NodeDecorator(
    node_id="np.a.partition",
    name="partition",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.partition)
def partition(
    a: ndarray,
    kth: int_or_int_array,
    axis: axis_like = -1,
):
    res = a.partition(kth, axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.prod",
    name="prod",
    outputs=[{"name": "out", "type": "Union[float,int, ndarray]"}],
)
@wraps(numpy.ndarray.prod)
def prod(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.prod(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.ptp",
    name="ptp",
    outputs=[{"name": "out", "type": "Union[float,int, ndarray]"}],
)
@wraps(numpy.ndarray.ptp)
def ptp(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.ptp(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.put",
    name="put",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.put)
def put(
    a: ndarray,
    indices: ndarray,
    values: ndarray,
    mode: Literal["raise", "wrap", "clip"] = "raise",
):
    res = a.put(indices, values, mode)
    return res


@fn.NodeDecorator(
    node_id="np.a.ravel",
    name="ravel",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.ravel)
def ravel(
    a: ndarray,
    order: str = "C",
):
    res = a.ravel()
    return res


@fn.NodeDecorator(
    node_id="np.a.repeat",
    name="repeat",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.repeat)
def repeat(
    a: ndarray,
    repeats: int,
    axis: Optional[axis_like] = None,
):
    res = a.repeat(repeats, axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.reshape",
    name="reshape",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.reshape)
def reshape(
    a: ndarray,
    newshape: shape_like,
):
    res = a.reshape(newshape)
    return res


@fn.NodeDecorator(
    node_id="np.a.resize",
    name="resize",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.resize)
def resize(
    a: ndarray,
    new_shape: shape_like,
):
    a.resize(new_shape)
    return a


@fn.NodeDecorator(
    node_id="np.a.round",
    name="round",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.round)
def round(
    a: ndarray,
    decimals: int = 0,
):
    res = a.round(decimals)
    return res


@fn.NodeDecorator(
    node_id="np.a.searchsorted",
    name="searchsorted",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.searchsorted)
def searchsorted(
    a: ndarray,
    v: ndarray,
    side: Literal["left", "right"] = "left",
    sorter: Optional[ndarray] = None,
):
    res = a.searchsorted(v, side, sorter)
    return res


@fn.NodeDecorator(
    node_id="np.a.sort",
    name="sort",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.sort)
def sort(
    a: ndarray,
    axis: Optional[int] = -1,
):
    res = a.sort(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.squeeze",
    name="squeeze",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.squeeze)
def squeeze(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.squeeze(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.std",
    name="std",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.std)
def std(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.std(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.sum",
    name="sum",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.sum)
def sum(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.sum(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.swapaxes",
    name="swapaxes",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.swapaxes)
def swapaxes(
    a: ndarray,
    axis1: int,
    axis2: int,
):
    res = a.swapaxes(axis1, axis2)
    return res


@fn.NodeDecorator(
    node_id="np.a.take",
    name="take",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.take)
def take(
    a: ndarray,
    indices: ndarray,
    axis: Optional[int] = None,
    mode: Literal["raise", "wrap", "clip"] = "raise",
):
    res = a.take(indices, axis, mode)
    return res


@fn.NodeDecorator(
    node_id="np.a.tobytes",
    name="tobytes",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.tobytes)
def tobytes(
    a: ndarray,
):
    res = a.tobytes()
    return res


@fn.NodeDecorator(
    node_id="np.a.tolist",
    name="tolist",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.tolist)
def tolist(
    a: ndarray,
):
    res = a.tolist()
    return res


@fn.NodeDecorator(
    node_id="np.a.tostring",
    name="tostring",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.tostring)
def tostring(
    a: ndarray,
):
    res = a.tostring()
    return res


@fn.NodeDecorator(
    node_id="np.a.trace",
    name="trace",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.trace)
def trace(
    a: ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
):
    res = a.trace(offset, axis1, axis2)
    return res


@fn.NodeDecorator(
    node_id="np.a.transpose",
    name="transpose",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.transpose)
def transpose(
    a: ndarray,
    axes: Optional[List[int]] = None,
):
    res = a.transpose(*axes)
    return res


@fn.NodeDecorator(
    node_id="np.a.var",
    name="var",
    outputs=[{"name": "out", "type": "ndarray_or_scalar"}],
)
@wraps(numpy.ndarray.var)
def var(
    a: ndarray,
    axis: Optional[axis_like] = None,
):
    res = a.var(axis)
    return res


@fn.NodeDecorator(
    node_id="np.a.view",
    name="view",
    outputs=[{"name": "out", "type": "ndarray"}],
)
@wraps(numpy.ndarray.view)
def view(
    a: ndarray,
    dtype: Optional[DTYPE_ENUM] = None,
):
    res = a.view(dtype_from_name(dtype))
    return res


NODE_SHELFE = fn.Shelf(
    nodes=[
        all,
        any,
        argmax,
        argmin,
        argpartition,
        argsort,
        astype,
        byteswap,
        choose,
        clip,
        compress,
        conj,
        conjugate,
        copy,
        cumprod,
        cumsum,
        diagonal,
        dot,
        fill,
        flatten,
        getfield,
        item,
        itemset,
        max,
        mean,
        min,
        newbyteorder,
        nonzero,
        partition,
        prod,
        ptp,
        put,
        ravel,
        repeat,
        reshape,
        resize,
        round,
        searchsorted,
        sort,
        squeeze,
        std,
        sum,
        swapaxes,
        take,
        tobytes,
        tolist,
        tostring,
        trace,
        transpose,
        var,
        view,
    ],
    subshelves=[],
    name="ndarray",
    description="memeber functions of numpy.ndarray",
)
