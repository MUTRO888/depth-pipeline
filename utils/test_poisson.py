import numpy as np
from poisson import normal_to_height

# Normal shape is (H, W, 3).
# Error was: operands could not be broadcast together with shapes(3412,2560)(2560,3)
# This typically happens if `normals_map = predictions.squeeze().cpu().numpy().transpose(1, 2, 0)` failed to squeeze the batch dim, 
# or if predictions was e.g. (1, H, W) instead of (1, 3, H, W).
# Marigold Normals typically outputs (1, 3, H, W).
# Wait, if output is (H, W, 3), Nx = map[..., 0] is (H, W).
# Let's see the error carefully: shapes(3412,2560)(2560,3)
# Ah! Zx = -Nx / Nz
# If Nx has shape (H, W) and Nz has shape (W, 3)? Wait.
# If normal_map has shape (3412, 2560) .. wait, if squeezing (1, 3, 3412, 2560) results in (3, 3412, 2560)
# Then transpose(1, 2, 0) gives (3412, 2560, 3).
# Nx = normal_map[..., 0] -> (3412, 2560).
# Ny = normal_map[..., 1] -> (3412, 2560).
# Nz = normal_map[..., 2] -> (3412, 2560).
# Then Zx = -Nx / Nz -> (3412, 2560).
# div[:, :-1] += Zx[:, 1:] - Zx[:, :-1]
# div is (3412, 2560). div[:, :-1] is (3412, 2559). Zx[:, 1:] is (3412, 2559).

# Let's test with dummy data:
dummy = np.random.rand(3412, 2560, 3)
try:
    res = normal_to_height(dummy)
    print("Success:", res.shape)
except Exception as e:
    print("Error:", str(e))
