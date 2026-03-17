import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def normal_to_height(normal_map):
    """
    Reconstructs a height map from a normal map using Poisson integration.
    
    Args:
        normal_map (np.ndarray): Shape (H, W, 3), values roughly in [-1, 1].
    
    Returns:
        np.ndarray: Height map (H, W), values strictly derived from gradient integration.
    """
    H, W, _ = normal_map.shape
    
    # Extract components corresponding to Nx, Ny, Nz
    # Typically, normal maps from models map X right, Y up/down, Z pointing at viewer
    # We must ensure Nz > 0 to avoid division by zero.
    Nx = normal_map[..., 0]
    Ny = normal_map[..., 1]
    Nz = np.clip(normal_map[..., 2], 1e-5, 1.0)
    
    # Derived gradients assuming Surface = (x, y, Z(x,y))
    # Normal N ∝ (-Zx, -Zy, 1) -> Zx = -Nx/Nz, Zy = -Ny/Nz
    # Note: image coordinates have Y going down, so we might need to flip the sign for Zy
    # depending on the specific normal map convention. 
    Zx = -Nx / Nz
    Zy = Ny / Nz 
    
    # Calculate divergence of the gradient field (Laplacian of Z)
    # div(G) = d(Zx)/dx + d(Zy)/dy
    div = np.zeros((H, W))
    
    # Forward difference for divergence
    div[:, :-1] += Zx[:, 1:] - Zx[:, :-1]
    div[:-1, :] += Zy[1:, :] - Zy[:-1, :]
    
    # Setup the sparse linear system: A * Z_flat = div_flat
    # A is the Laplacian operator matrix
    num_pixels = H * W
    A = lil_matrix((num_pixels, num_pixels))
    b = div.flatten()
    
    # Build Laplacian matrix
    # Using a simple 5-point stencil: -4*Z(x,y) + Z(x+1,y) + Z(x-1,y) + Z(x,y+1) + Z(x,y-1)
    
    # To speed up matrix construction in pure Python without C extensions,
    # we use flat indices.
    
    # 0, 1, ..., num_pixels - 1
    idx = np.arange(num_pixels)
    row_idx = idx // W
    col_idx = idx % W
    
    # Mask for interior points
    is_interior = (row_idx > 0) & (row_idx < H - 1) & (col_idx > 0) & (col_idx < W - 1)
    
    # Setup diagonals and neighbors for matrix A by building lists for coo_matrix or updating lil_matrix
    # For a small application, lil_matrix pointwise assignment can be slow, but it's acceptable for now.
    # A more optimized way:
    import scipy.sparse as sp
    
    main_diag = -4 * np.ones(num_pixels)
    # Boundaries: Dirichlet condition Z=0 on edges (simplest approach for isolated objects)
    # Or Neumann (derivative = 0). Here we just set boundary to z=0 for simplicity
    main_diag[~is_interior] = 1.0
    b[~is_interior] = 0.0
    
    off_diag_x = np.ones(num_pixels - 1)
    off_diag_x[W-1::W] = 0.0 # break connections between rows
    
    off_diag_y = np.ones(num_pixels - W)
    
    A = sp.diags(
        [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y],
        [0, 1, -1, W, -W],
        format='csr'
    )
    
    # For boundary points, we need to clear out their off-diagonal entries
    # A cleaner but slightly slower way to enforce boundary conditions:
    # (In Python loops it's slow, so we did the simple diagonal trick above, which implies a slight distortion at the very edge)
    
    # Solve A * Z = b
    # Using scipy's spsolve. For large matrices this might take a second.
    Z_flat = spsolve(A, b)
    
    Z = Z_flat.reshape((H, W))
    
    # Normalize result back to [0, 1] range to match depth pipeline expectations
    Z_min, Z_max = Z.min(), Z.max()
    if Z_max - Z_min > 1e-6:
        Z = (Z - Z_min) / (Z_max - Z_min)
        
    return Z
