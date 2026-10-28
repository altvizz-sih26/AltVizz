import sys, os, torch
path = sys.argv[1]
print("file:", path, "| size MB:", round(os.path.getsize(path)/1e6, 1))
ckpt = torch.load(path, map_location="cpu")
print("top-level keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
for k in ("epoch", "best_metric", "val_metrics", "metrics", "args"):
    if isinstance(ckpt, dict) and k in ckpt:
        print(k, "=", ckpt[k])
state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
keys = list(state.keys())
print("n weights:", len(keys))
print("first keys:", keys[:3])
print("last keys:", keys[-3:])