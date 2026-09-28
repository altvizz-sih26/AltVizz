import h5py

sample_id = "DC_03_26"

files = {
    "RGB":    f"gamus_test/images/{sample_id}_RGB.h5",
    "AGL":    f"gamus_test/heights/{sample_id}_AGL.h5",
    "CLS":    f"gamus_test/classes/{sample_id}_CLS.h5",
}

for label, path in files.items():
    print(f"=== {label}: {path} ===")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  key='{name}'  shape={obj.shape}  dtype={obj.dtype}  "
                      f"min={obj[()].min():.3f}  max={obj[()].max():.3f}")
        f.visititems(show)
    print()