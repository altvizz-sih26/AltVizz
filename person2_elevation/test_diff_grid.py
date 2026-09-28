import numpy as np
from diff_grid import compute_difference_grid, summarize_difference_grid

shape = (10, 10)

elev_before = np.full(shape, 100.0, dtype=np.float32)
elev_after = elev_before.copy()
elev_after[0:3, 0:3] -= 5.0     # real change: 5m drop
elev_after[5:8, 5:8] += 0.1     # noise-level: 0.1m shift

class_before = np.zeros(shape, dtype=int)
class_after = class_before.copy()
class_after[0:3, 0:3] = 1       # class label changed where real change happened

conf_before = np.full(shape, 0.9, dtype=np.float32)
conf_after = np.full(shape, 0.9, dtype=np.float32)

result = compute_difference_grid(elev_before, elev_after, class_before, class_after,
                                  conf_before, conf_after, max_margin=2.0)

print("Real 5m change region (expect all True):")
print(result["significant_change"][0:3, 0:3])

print("\n0.1m noise region (expect all False):")
print(result["significant_change"][5:8, 5:8])

print("\nClass-changed region (expect all True):")
print(result["class_changed"][0:3, 0:3])

print("\nSummary:", summarize_difference_grid(result))