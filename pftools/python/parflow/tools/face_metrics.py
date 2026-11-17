"""Prepare static reference-sphere face scales for a meter-background grid.

DXScale is on y faces, DYScale on x faces. Angles are degrees; arrays use
ParFlow's (z, y, x) convention. Keep the resulting PFBs undistributed.
"""

import numpy as np


def reference_sphere_face_scales(nx, ny, nz, south, dlon, dlat, dx, dy,
                                 radius=6371000.0):
    """Return (DXScale, DYScale) for the thin-layer face-metric approximation.

    Cell-center distances use the average of opposite face-edge lengths.
    The resulting horizontal area is trapezoidal, not the exact sine integral.
    All layers share the same horizontal metrics; DZ/z_mult remain independent.
    """
    if any(int(n) != n or n <= 0 for n in (nx, ny, nz)):
        raise ValueError("nx, ny and nz must be positive integers")
    nx, ny, nz = int(nx), int(ny), int(nz)
    if not np.all(np.isfinite([south, dlon, dlat, dx, dy, radius])):
        raise ValueError("geometry values must be finite")
    if min(dlon, dlat, dx, dy, radius) <= 0:
        raise ValueError("angular spacing, meter spacing and radius must be positive")
    if south <= -90 or south + ny * dlat >= 90 or nx * dlon > 360:
        raise ValueError("grid must exclude poles and span at most 360 degrees")
    phi = np.deg2rad(south + np.arange(ny + 1) * dlat)
    xs = radius * np.deg2rad(dlon) * np.cos(phi) / dx
    dx_scale = np.broadcast_to(xs[None, :, None], (nz, ny + 1, nx)).copy()
    dy_scale = np.full((nz, ny, nx + 1), radius * np.deg2rad(dlat) / dy)
    return dx_scale, dy_scale


def write_face_scales(dx_filename, dy_filename, dx_scale, dy_scale,
                      dx, dy, dz, x=0.0, y=0.0, z=0.0):
    """Write the two whole-domain PFBs (no distribution or input mutation)."""
    from .io import write_pfb

    xs, ys = np.asarray(dx_scale, dtype=np.float64), np.asarray(dy_scale, dtype=np.float64)
    if xs.ndim != 3 or ys.ndim != 3 or not (
        xs.shape[0] == ys.shape[0] and xs.shape[1] == ys.shape[1] + 1
        and xs.shape[2] + 1 == ys.shape[2]
    ):
        raise ValueError("expected DXScale=(nz,ny+1,nx), DYScale=(nz,ny,nx+1)")
    if any(not np.all(np.isfinite(a) & (a > 0)) for a in (xs, ys)):
        raise ValueError("all scales must be finite and positive")
    for filename, values in ((dx_filename, xs), (dy_filename, ys)):
        write_pfb(str(filename), values, x=x, y=y, z=z,
                  dx=dx, dy=dy, dz=dz, dist=False)
