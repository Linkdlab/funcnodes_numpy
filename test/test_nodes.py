import unittest
import funcnodes_numpy as fnp
import funcnodes as fn


def flatten_shelves(shelf: fn.Shelf):
    nodes = shelf["nodes"].copy()
    subshelves = shelf["subshelves"]
    for subshelf in subshelves:
        nodes.extend(flatten_shelves(subshelf))
    return set(nodes)


from pprint import pprint
import numpy as np


samplemap = {
    "ndarray": lambda: [
        np.array([1, 2, 3]),
        np.array([0, 1]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.arange(2 * 2 * 2).reshape(2, 2, 2),
        np.array([1, 2, 3], dtype=np.float32),
    ],
    "Union[bool, complex, float, int, ndarray, str]": lambda: [
        True,
        1j,
        1.0,
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.arange(2 * 2 * 2).reshape(2, 2, 2),
        np.datetime64("2021-01-01"),
        "str",
    ],
    "Union[None, float]": lambda: [None, 1.0],
    "Union[None, float, int]": lambda: [None, 1.0, 1],
    "Union[None, int]": lambda: [None, 1],
    "Union[None, bool]": lambda: [None, True],
    "Union[None, ndarray]": lambda: [
        None,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[None, dict]": lambda: [None, {"a": lambda: 1}],
    "Union[None, Sequence[Tuple[float, float]]]": lambda: [None, [(1.0, 2.0)]],
    "Union[None, Sequence[int]]": lambda: [None, [1, 2, 3]],
    "Union[None, Sequence[Union[float, int]]]": lambda: [None, [1, 2, 3]],
    "Union[List[int], int]": lambda: [[1, 2, 3], 1],
    "Union[List[int], None, int]": lambda: [[1, 2, 3], None, 1],
    "Union[Literal[1, 2], None]": lambda: [1, 2, None],
    "Union[None, float, int, ndarray]": lambda: [
        None,
        1.0,
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[complex, float, int, ndarray]": lambda: [
        1j,
        1.0,
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[numpy._typing._array_like._SupportsArray, numpy._typing._nested_sequence._NestedSequence, numpy._typing._nested_sequence._NestedSequence, str]": lambda: [
        np.array([1, 2, 3]),
        [[1, 2, 3]],
        [[[1, 2, 3]]],
        "str",
    ],
    "Union[int, ndarray]": lambda: [
        0,
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[Sequence[int], int]": lambda: [[1, 2, 3], 1],
    "Union[Sequence[Union[float, int]], Sequence[str], int]": lambda: [
        [1, 2, 3],
        ["a", "b", "c"],
        1,
    ],
    "Union[bytearray, bytes, memoryview, ndarray]": lambda: [
        bytearray(b"hello"),
        b"12345678",
        memoryview(b"hello"),
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[float, int, ndarray]": lambda: [
        1.0,
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
    ],
    "Union[int, ndarray, str]": lambda: [
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]).reshape(2, 2),
        "str",
    ],
    "Union[int, ndarray]": lambda: [
        1,
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.array([1, 2, 3]),
    ],
    "Union[Literal['big', 'little'], None]": lambda: ["big", "little", None],
    "Union[None, bool]": lambda: [None, True],
    "Union[None, int]": lambda: [None, 1],
    "Union[None, int, ndarray]": lambda: [
        None,
        1,
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.array([1, 2, 3]),
    ],
    "Union[None, ndarray]": lambda: [
        None,
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.array([1, 2, 3]),
    ],
    "Union[Literal['full', 'valid', 'same'], None]": lambda: [
        "full",
        "valid",
        "same",
        None,
    ],
    "Union[None, float]": lambda: [None, 1.0],
    "Union[None, Tuple[float, float]]": lambda: [None, (1.0, 2.0)],
    "Union[None, bool, complex, float, int, ndarray, str]": lambda: [
        None,
        True,
        1j,
        1.0,
        1,
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.array([1, 2, 3]),
        "str",
    ],
    "Union[None, funcnodes_numpy._dtypes.DTYPE_ENUM]": lambda: [None, "f", bool],
    "Union[List[int], None]": lambda: [[1, 2, 3], None],
    "Union[Tuple[Union[int, ndarray], Union[int, ndarray]], int, ndarray]": lambda: [
        (1, np.array([1, 2, 3])),
        np.array([1, 2, 3, 4]).reshape(2, 2),
        np.array([1, 2, 3]),
        1,
    ],
    "Union[Tuple[Union[float, int], Union[float, int]], float, int]": lambda: [
        (1.0, 2.0),
        1.0,
        1,
    ],
    "Literal['reduced', 'complete', 'r', 'raw']": lambda: [
        "reduced",
        "complete",
        "r",
        "raw",
    ],
    "int": lambda: [1, 2, 0],
    "List[ndarray]": lambda: [
        [
            np.array([3, 2, 1]),
            np.array([1, 2, 3]),
        ]
    ],
    "typing.Callable": lambda: [lambda x: x],
    "funcnodes_numpy._dtypes.DTYPE_ENUM": lambda: ["f", bool],
    "float": lambda: [1.0, 1],
    "List[Union[float, int, ndarray]]": lambda: [
        [1.0, 1, np.array([1, 2, 3, 4]).reshape(2, 2), np.array([1, 2, 3])]
    ],
    "Union[Literal['fro', 'nuc'], None, float]": lambda: ["fro", "nuc", None, 1.0],
    "List[Union[bool, complex, float, int, ndarray, str]]": lambda: [
        [
            True,
            1j,
            1.0,
            1,
            np.array([1, 2, 3, 4]).reshape(2, 2),
            np.array([1, 2, 3]),
            "str",
        ]
    ],
    "Union[float, int]": lambda: [1.0, 1],
    "bool": lambda: [True, False],
    "Union[None, str]": lambda: [None, "str"],
    "Literal['left', 'right']": lambda: ["left", "right"],
    "Union[ndarray, str]": lambda: [np.array([1, 2, 3]), "str"],
    "Sequence[ndarray]": lambda: [
        [np.array([1, 2, 3, 4]).reshape(2, 2), np.array([1, 2, 3])],
        [np.array([4, 3, 2]), np.array([1, 2, 3])],
    ],
    "Literal['L', 'U']": lambda: ["L", "U"],
    "builtins.object": lambda: [object()],
    "Literal['raise', 'wrap', 'clip']": lambda: ["raise", "wrap", "clip"],
    "Sequence[Union[bool, complex, float, int, ndarray, str]]": lambda: [
        [True, 1j, 1.0, 1, np.array([1, 2, 3]), "str"]
    ],
    "Union[List[Tuple[int, int]], Tuple[int, int], int]": lambda: [
        [(1, 2), (3, 4)],
        (1, 2),
        1,
    ],
    "Union[Literal[False, True, 'greedy', 'optimal'], None]": lambda: [
        False,
        True,
        "greedy",
        "optimal",
        None,
    ],
    "List[Union[float, int]]": lambda: [[1.0, 1], [2.0, 1]],
    "Literal['xy', 'ij']": lambda: ["xy", "ij"],
    "Literal['F', 'C', 'A', 'W', 'O', 'E']": lambda: ["F", "C", "A", "W", "O", "E"],
    "Sequence[float]": lambda: [[1.0, 2.0, 3.0]],
    "Any": lambda: [1, 1.0, "str", True],
    "Literal['S', '<', '>', '=', '|']": lambda: ["S", "<", ">", "=", "|"],
    "typing.Iterable": lambda: [[1, 2, 3]],
    "Literal['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap', 'empty']": lambda: [
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
    ],
    "str": lambda: ["str", "ii"],
    "Union[List[Union[float, int, ndarray]], None]": lambda: [
        [1.0, 1, np.array([1, 2, 3])],
        None,
    ],
}


class TestNodes(unittest.IsolatedAsyncioTestCase):
    async def test_missing_types(self):
        shelvenodes = flatten_shelves(fnp.NODE_SHELF)
        missing_types = set()
        for node in shelvenodes:
            exf = node.func.ef_funcmeta
            for ip in exf["input_params"]:
                if ip["type"] not in samplemap:
                    missing_types.add(ip["type"])
                    continue

        self.assertEqual(len(missing_types), 0, missing_types)
        for node in shelvenodes:
            exf = node.func.ef_funcmeta
            for ip in exf["input_params"]:
                assert isinstance(
                    samplemap[ip["type"]], list
                ), f"{ip['type']} not a list"

    async def test_nodes(self):
        shelvenodes = flatten_shelves(fnp.NODE_SHELF)
        ignore = [fnp.clip]
        for node in shelvenodes:
            if node in ignore:
                continue
            ins = node()
            exf = node.func.ef_funcmeta
            kwargs = {}
            args = []
            types = []
            for ip in exf["input_params"]:
                if ip["positional"]:
                    newargs = []
                    types.append(ip["type"])
                    options = samplemap[ip["type"]]()
                    for option in options:
                        if len(args) == 0:
                            newargs.append([option])
                        else:
                            for a in args:
                                newargs.append(a + [option])
                    args = newargs
                else:
                    kwargs[ip["name"]] = samplemap[ip["type"]]()[0]
            res = None
            errors = []
            if len(args) == 0:
                raise ValueError(f"len args 0 for  {node.node_name} ")
            run = False
            for a in args:
                try:
                    res = await ins.func(
                        *a,
                    )
                    run = True
                except Exception as e:
                    errors.append((str(e), a))
            if res is None:
                print(types)
                print(errors)
                raise Exception(
                    f"Failed to run {node.node_name} with types {types}:\n {errors} \n {run}"
                )


np.test("full")
