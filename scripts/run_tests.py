"""Run both unittest classes and zero-argument function regression tests."""
from pathlib import Path
import importlib
import inspect
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
suite=unittest.TestSuite()
for path in sorted((ROOT/'tests').glob('test_*.py')):
    module=importlib.import_module('tests.'+path.stem)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    for name, function in inspect.getmembers(module,inspect.isfunction):
        if name.startswith('test_') and function.__module__==module.__name__:
            suite.addTest(unittest.FunctionTestCase(function))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
