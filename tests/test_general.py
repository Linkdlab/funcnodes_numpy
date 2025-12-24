import pytest
import funcnodes_numpy as fnp

import funcnodes as fn


def get_module_nodes(module):
    nodes = [getattr(module, node) for node in dir(module)]
    nodes = [
        node for node in nodes if isinstance(node, type) and issubclass(node, fn.Node)
    ]
    return nodes


def test_main_shelf():
    shelf = fnp.NODE_SHELF
    assert shelf.name == "numpy"
    assert len(shelf.nodes) == 0
    assert len(shelf.subshelves) == 16


def test_all_nodes():
    nodes = get_module_nodes(fnp)
    exp = 307
    exp_shelfnodes = 374
    if fnp.np_version["major_int"] < 2:
        exp -= 1
        exp_shelfnodes -= 1
    if fnp.np_version["major_int"] >= 2 and fnp.np_version["minor_int"] >= 2:
        exp += 3
        exp_shelfnodes += 3
    assert len(nodes) + 1 == exp
    for node in nodes:
        print(node.node_name)

    shelvenodes, _ = fn.flatten_shelf(fnp.NODE_SHELF)
    missing_shelvenodes = set(nodes) - (set(shelvenodes))
    assert len(missing_shelvenodes) == 1, [n.node_name for n in missing_shelvenodes]
    assert len(shelvenodes) == exp_shelfnodes


@pytest.mark.asyncio
async def test_ndarray_shelve():
    shelf = fnp._ndarray.NODE_SHELF
    shelve_nodes, _ = fn.flatten_shelf(shelf)
    module_nodes = get_module_nodes(fnp._ndarray)
    assert len(shelve_nodes) == len(module_nodes)
    assert len(shelve_nodes) == 49


@pytest.mark.asyncio
async def test_linalg_shelve():
    shelf = fnp._linalg.NODE_SHELF
    shelve_nodes, _ = fn.flatten_shelf(shelf)
    module_nodes = get_module_nodes(fnp._linalg)
    assert len(shelve_nodes) == len(module_nodes)
    assert len(shelve_nodes) == 20


@pytest.mark.asyncio
async def test_emath_shelve():
    shelf = fnp._lib.EMATH_NODE_SHELF
    shelve_nodes, _ = fn.flatten_shelf(shelf)
    module_nodes = get_module_nodes(fnp._lib.scimath)
    assert len(shelve_nodes) == len(module_nodes)
    assert len(shelve_nodes) == 9


@pytest.mark.asyncio
async def test_core_shelve():
    exp_nodes = 261
    if fnp.np_version["major_int"] < 2:
        exp_nodes -= 1
    if fnp.np_version["major_int"] >= 2 and fnp.np_version["minor_int"] >= 2:
        exp_nodes += 3
    module_nodes = get_module_nodes(fnp._core)
    assert len(module_nodes) == exp_nodes
