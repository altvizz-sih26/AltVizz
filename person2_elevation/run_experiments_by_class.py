import numpy as np
from pathlib import Path
from load_gamus import load_gamus_sample, discover_complete_samples
from fit_methods import fit_global_linear, fit_terrain_wise, predict_terrain_wise
from metrics import compute_metrics

DEPTH_DIR = Path("person2_elevation/GAMUS_depth")
SPLIT_PATH = Path("person2_elevation/gamus_split.npz")

CLASS_NAMES = {1: "ground", 2: "low_vegetation", 3: "building", 4: "water", 5: "road", 6: "tree"}

def load_depth(sample_id):
    return np.load(DEPTH_DIR / f"{sample_id}_RGB_depth.npy")

def run_by_class():
    sample_ids = discover_complete_samples()
    if not sample_ids:
        print("No complete samples yet.")
        return

    split = np.load(SPLIT_PATH)

    all_depth_train, all_agl_train, all_class_train = [], [], []
    all_depth_test, all_agl_test, all_class_test = [], [], []

    for sample_id in sample_ids:
        _, agl, class_map = load_gamus_sample(sample_id)
        depth = load_depth(sample_id)
        train_mask = split[f"{sample_id}_train"]
        test_mask = split[f"{sample_id}_test"]

        all_depth_train.append(depth[train_mask]); all_agl_train.append(agl[train_mask]); all_class_train.append(class_map[train_mask])
        all_depth_test.append(depth[test_mask]);   all_agl_test.append(agl[test_mask]);   all_class_test.append(class_map[test_mask])

    depth_tr = np.concatenate(all_depth_train); agl_tr = np.concatenate(all_agl_train); cls_tr = np.concatenate(all_class_train)
    depth_te = np.concatenate(all_depth_test);   agl_te = np.concatenate(all_agl_test);   cls_te = np.concatenate(all_class_test)

    print(f"Samples used: {len(sample_ids)} | Train pixels: {len(depth_tr)} | Test pixels: {len(depth_te)}\n")

    for name, model_type, degree in [
        ("Test 2: Terrain-wise Linear", "linear", 1),
        ("Test 3: Terrain-wise Huber", "huber", 1),
        ("Test 4: Terrain-wise Polynomial (deg 2)", "linear", 2),
    ]:
        print(f"=== {name} ===")
        models = fit_terrain_wise(depth_tr, agl_tr, cls_tr, model_type=model_type, degree=degree)
        pred_all = predict_terrain_wise(depth_te, cls_te, models)

        for class_id in sorted(CLASS_NAMES):
            cls_mask = cls_te == class_id
            if cls_mask.sum() == 0:
                continue
            result = compute_metrics(pred_all[cls_mask], agl_te[cls_mask])
            print(f"  {CLASS_NAMES[class_id]:>15}: corr={result['correlation']:.3f}  "
                  f"mae={result['mae']:.2f}  rmse={result['rmse']:.2f}  n={result['n']}")
        print()

if __name__ == "__main__":
    run_by_class()