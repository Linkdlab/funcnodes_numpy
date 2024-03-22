import unittest
import funcnodes_numpy as fnp
import funcnodes as fn
import numpy as np
import inspect


def flatten_shelves(shelf: fn.Shelf):
    nodes = shelf["nodes"].copy()
    subshelves = shelf["subshelves"]
    for subshelf in subshelves:
        nodes.extend(flatten_shelves(subshelf))
    return set(nodes)


class TestNumpyLib(unittest.TestCase):
    def test_ufunc_completness(self):
        from numpy import _core as np_core
        from numpy._core import umath

        nodes = flatten_shelves(fnp.NODE_SHELF)
        node_names = [node.node_name for node in nodes]
        all_funcs = fnp.ufuncs.get_numpy_ufucs()
        print(all_funcs)
        self.assertIn("add", all_funcs)
        self.assertIn("subtract", all_funcs)
        self.assertIn("sin", all_funcs)
        self.assertIn("cos", all_funcs)
        self.assertIn("sqrt", all_funcs)
        self.assertIn("exp", all_funcs)

        for f in all_funcs:
            self.assertIn(f, node_names)

        for f in all_funcs:
            for node in nodes:
                if node.node_name == f:
                    srcf = inspect.getsourcefile(inspect.unwrap(node.func))
                    self.assertTrue(
                        srcf.endswith("ufuncs.py"),
                        f"souce file for {f} is {srcf} not ufuncs.py",
                    )

        inifilecode = None
        newcode = ""
        for f in all_funcs:
            for node in nodes:
                if node.node_name == f:
                    if inspect.getsourcefile(inspect.unwrap(node.func)).endswith(
                        "funcnodes_numpy\\funcnodes_numpy\\_core\\__init__.py"
                    ):
                        if inifilecode is None:
                            with open(
                                inspect.getsourcefile(inspect.unwrap(node.func))
                            ) as f:
                                inifilecode = f.read()

                        src = inspect.getsource(node.func)
                        if src not in inifilecode:
                            continue
                        inifilecode = inifilecode.replace(src, "")
                        newcode += src + "\n"
                        break

        with open("test.py", "w") as f:
            f.write(newcode)

        with open("test2.py", "w") as f:
            f.write(inifilecode)

    def test_calls(self):
        import inspect

        print(inspect.getsource(testf))


from exposedfunctionality import controlled_wrapper as wraps


@wraps(np.e, wrapper_attribute="__fnwrapped__")
def testf():
    return 1
