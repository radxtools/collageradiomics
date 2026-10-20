"""Run the extension integration suite with Slicer --python-script."""

import sys
import traceback
import unittest
from pathlib import Path

import slicer

repository = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(repository / "module"),
    str(repository / "slicer/collageradiomics/CollageRadiomicsSlicer"),
]

try:
    from CollageRadiomicsSlicer import CollageRadiomicsSlicerTest

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CollageRadiomicsSlicerTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    slicer.util.exit(0 if result.wasSuccessful() else 1)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
