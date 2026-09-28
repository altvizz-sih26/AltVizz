import h5py
import numpy as np
from pathlib import Path

CLS_DIR = Path("gamus_test/classes")

files = sorted(CLS_DIR.glob("*_CLS.h5"))

print(f"Found {len(files)} class files.\n")

for path in files[:5]:
    with h5py.File(path, "r") as f:
        cls = f["image"][()]

    values, counts = np.unique(cls, return_counts=True)

    print(f"=== {path.name} ===")

    for value, count in zip(values, counts):
        percentage = (count / cls.size) * 100
        print(f"Class {int(value)}: {count:,} pixels ({percentage:.2f}%)")

    print()