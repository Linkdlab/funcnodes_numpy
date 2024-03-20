import unittest
import funcnodes_numpy as fnp
from pprint import pprint
import numpy as np
import funcnodes as fn


def flatten_shelves(shelf: fn.Shelf):
    nodes = shelf["nodes"].copy()
    subshelves = shelf["subshelves"]
    for subshelf in subshelves:
        nodes.extend(flatten_shelves(subshelf))
    return set(nodes)


def get_module_nodes(module):
    nodes = [getattr(module, node) for node in dir(module)]
    nodes = [
        node for node in nodes if isinstance(node, type) and issubclass(node, fn.Node)
    ]
    return nodes


class TestGeneral(unittest.IsolatedAsyncioTestCase):
    def test_main_shelf(self):
        shelf = fnp.NODE_SHELFE
        self.assertEqual(shelf["name"], "numpy")
        self.assertEqual(len(shelf["nodes"]), 0)
        self.assertEqual(len(shelf["subshelves"]), 15)

    def test_all_nodes(self):
        nodes = get_module_nodes(fnp)
        self.assertEqual(len(nodes), 299)
        for node in nodes:
            print(node.node_name)

        shelvenodes = flatten_shelves(fnp.NODE_SHELFE)
        missing_shelvenodes = set(nodes) - (set(shelvenodes))
        self.assertEqual(
            len(missing_shelvenodes), 0, [n.node_name for n in missing_shelvenodes]
        )

        self.assertEqual(len(shelvenodes), 329)

    async def test_ndarray_shelve(self):
        shelf = fnp._ndarray.NODE_SHELFE
        shelve_nodes = flatten_shelves(shelf)
        module_nodes = get_module_nodes(fnp._ndarray)
        self.assertEqual(len(shelve_nodes), len(module_nodes))
        self.assertEqual(len(shelve_nodes), 51)

    async def test_linalg_shelve(self):
        shelf = fnp._linalg.NODE_SHELFE
        shelve_nodes = flatten_shelves(shelf)
        module_nodes = get_module_nodes(fnp._linalg)
        self.assertEqual(len(shelve_nodes), len(module_nodes))
        self.assertEqual(len(shelve_nodes), 20)

    async def test_emath_shelve(self):
        shelf = fnp._lib.EMATH_NODE_SHELFE
        shelve_nodes = flatten_shelves(shelf)
        module_nodes = get_module_nodes(fnp._lib.scimath)
        self.assertEqual(len(shelve_nodes), len(module_nodes))
        self.assertEqual(len(shelve_nodes), 9)

    async def test_core_shelve(self):
        module_nodes = get_module_nodes(fnp._core)
        self.assertEqual(len(module_nodes), 253)
