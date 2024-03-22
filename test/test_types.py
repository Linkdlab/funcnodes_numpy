import unittest
import funcnodes_numpy as fnp
from typing import Union, List, Literal, TYPE_CHECKING
from pprint import pprint
import numpy as np
import funcnodes as fn
from exposedfunctionality.function_parser.types import (
    string_to_type,
    _TYPE_GETTER,
    type_to_string,
)


def flatten_shelves(shelf: fn.Shelf) -> list[fn.Node]:
    nodes = shelf["nodes"]
    subshelves = shelf["subshelves"]
    for subshelf in subshelves:
        nodes.extend(flatten_shelves(subshelf))
    return set(nodes)


class TestTypes(unittest.TestCase):
    def test_typestrings(self):
        shelf = fnp.NODE_SHELF
        nodes = flatten_shelves(shelf)
        for node in nodes:
            ins: fn.Node = node()
            for ipname, ip in ins.inputs.items():
                try:
                    t = type_to_string(string_to_type(ip.serialize_class()["type"]))
                except Exception as e:
                    print(
                        "\n\n",
                        node.node_name,
                        ipname,
                        "\n\n",
                    )
                    raise e
            for ipname, ip in ins.outputs.items():
                try:
                    t = type_to_string(string_to_type(ip.serialize_class()["type"]))
                    print(
                        t,
                        string_to_type(ip.serialize_class()["type"]),
                        ip.serialize_class()["type"],
                    )
                except Exception as e:
                    print(_TYPE_GETTER)
                    print(
                        "\n\n",
                        node.node_name,
                        ipname,
                        "\n\n",
                    )
                    raise e

    def test_types_to_sring(self):
        for src, exp, exps in [
            (fnp.DTYPE_ENUM, fnp.DTYPE_ENUM, "funcnodes_numpy._dtypes.DTYPE_ENUM"),
            (fnp._types.scalar, Union[float, int], "Union[float, int]"),
            (
                fnp._types.number,
                Union[complex, int, float],
                "Union[complex, float, int]",
            ),
            (
                fnp._types.ndarray_or_scalar,
                Union[np.ndarray, fnp._types.scalar],
                "Union[float, int, ndarray]",
            ),
            (
                fnp._types.ndarray_or_number,
                Union[np.ndarray, fnp._types.number],
                "Union[complex, float, int, ndarray]",
            ),
            (
                fnp._types.indices_or_sections,
                Union[List[int], int],
                "Union[List[int], int]",
            ),
            (fnp._types.shape_like, Union[List[int], int], "Union[List[int], int]"),
            (fnp._types.axis_like, Union[List[int], int], "Union[List[int], int]"),
            (fnp._types.ndarray, np.ndarray, "ndarray"),
            (
                fnp._types.array_like,
                Union[bool, complex, float, int, np.ndarray, str],
                "Union[bool, complex, float, int, ndarray, str]",
            ),
            (fnp._types.int_array, np.ndarray, "ndarray"),
            (fnp._types.bool_array, np.ndarray, "ndarray"),
            (fnp._types.bitarray, np.ndarray, "ndarray"),
            (
                fnp._types.bool_or_bool_array,
                Union[bool, fnp._types.bool_array],
                "Union[bool, ndarray]",
            ),
            (
                fnp._types.int_bool_array,
                Union[fnp._types.int_array, fnp._types.bool_array],
                "ndarray",
            ),
            (
                fnp._types.int_or_int_array,
                Union[int, fnp._types.int_array],
                "Union[int, ndarray]",
            ),
            (fnp._types.real_array, np.ndarray, "ndarray"),
            (fnp._types.matrix, np.ndarray, "ndarray"),
            (fnp._types.OrderCF, Literal[None, "C", "F"], "Literal[None, 'C', 'F']"),
            (
                fnp._types.OrderKACF,
                Literal[None, "K", "A", "C", "F"],
                "Literal[None, 'K', 'A', 'C', 'F']",
            ),
            (
                fnp._types.OrderACF,
                Literal[None, "A", "C", "F"],
                "Literal[None, 'A', 'C', 'F']",
            ),
            (
                fnp._types.buffer_like,
                Union[bytes, bytearray, memoryview, np.ndarray],
                "Union[bytearray, bytes, memoryview, ndarray]",
            ),
            (
                fnp._types.str_array,
                np._ArrayLikeStr_co if TYPE_CHECKING else np._typing._ArrayLikeStr_co,
                "Union[numpy._typing._array_like._SupportsArray, numpy._typing._nested_sequence._NestedSequence, numpy._typing._nested_sequence._NestedSequence, str]",
            ),
            (fnp._types.NoValue, np._NoValue, "<no value>"),
            (
                fnp._types.casting_literal,
                Literal["no", "equiv", "safe", "same_kind", "unsafe"],
                "Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']",
            ),
        ]:
            self.assertEqual(type_to_string(src), type_to_string(exp), (src, exp))
            self.assertEqual(type_to_string(src), exps)

    def test_enum_options(self):
        node = fnp.astype()
        self.assertEqual(
            node.get_input("dtype").value_options["options"],
            {i.name: i.value for i in fnp.DTYPE_ENUM},
        )

    def test_literal_options(self):
        node = fnp.newbyteorder()
        self.assertEqual(
            node.get_input("new_order").value_options["options"],
            ["S", "<", ">", "=", "|"],
        )
