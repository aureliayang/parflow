# Experimental horizontal face metrics

Branch `face-metric-pfb` retains the reference-sphere implementation and adds a
mutually exclusive input mode. ComputationalGrid coordinates and spacings remain
**meters**, including the spacings passed to PFMG. Angles are only used by the
external Python preparation helper. Existing spherical mode is not changed.

```python
run.Solver.Experimental.SphericalAccumulation = False
run.Solver.FaceMetrics.Enabled = True
run.Solver.FaceMetrics.DXScale.FileName = "dx_scale.pfb"
run.Solver.FaceMetrics.DYScale.FileName = "dy_scale.pfb"
```

The two PFBs contain finite, strictly positive, dimensionless scales:

| File | Geometric location | PFB NX,NY,NZ | NumPy shape |
| --- | --- | --- | --- |
| DXScale | east-west edge lengths on south/north faces | NX,NY+1,NZ | (NZ,NY+1,NX) |
| DYScale | north-south edge lengths on west/east faces | NX+1,NY,NZ | (NZ,NY,NX+1) |

These are edge length scales, **not** permeability, flux multipliers, or
center-to-center distances. Files have zero-based global indices. Keep both
files **undistributed**: do not call run.dist/pfdist on them. The experimental
reader reads each whole PFB on each MPI rank once, retaining only local storage
and halos. Multiple file subgrids are supported. Physical exterior halos use
constant extension. This favors small verification cases over scalable input
throughput; no rank-local face-file distribution is implemented yet.

Face vectors use grids grown by one in the relevant direction, like the
velocity grids. Ownership/lifetime is in ProblemData, and cleanup frees both
vectors and their grids. Thickness halos use the existing MPI exchange.

## Reconstruction

For a cell, let xs,xn be its south/north DX scales and yw,ye its west/east DY
scales. Let dx0,dy0 be background spacings and t=DZ*z_mult be actual thickness:

- center distances: dx=dx0*(xs+xn)/2; dy=dy0*(yw+ye)/2;
- west/east areas: dy0*yw*t, dy0*ye*t;
- south/north areas: dx0*xs*t, dx0*xn*t;
- horizontal area: dx*dy; volume: dx*dy*t.

All-one fields recover Cartesian geometry. Residual and Jacobian share the same
C helper. Positive-face integrated fluxes are computed once and applied with
opposite signs, including when face metrics vary. Metrics are static, independent
of pressure; slopes and other physical inputs retain their existing meanings.

This is an **orthogonal face-scale approximation**, not a general curvilinear
metric tensor. For reference-sphere fields, center dx uses the average of the
north/south cosine values. Horizontal area uses trapezoidal latitude quadrature,
not the exact sine integral used by SphericalAccumulation. Its area ratio to the
exact integral is (dphi/2)*cot(dphi/2), approximately 1-dphi^2/12. At 0.1 degrees
the relative area difference is about 2.54e-7. Do not claim the two modes are
bitwise equivalent. Arbitrary independent 3-D scales need not describe a
geometrically realizable mesh.

Scope: unrefined index grid; underground Richards flow, storage/source terms,
FluxConst side/bottom boundaries, and top SeepageFace. Other boundary laws,
overland routing, wells and other geometry-dependent modules are not converted.
Prefer all-zero FluxConst around sides/bottom and SeepageFace on top for tests.

## Preparing files in Python

```python
import numpy as np
from parflow.tools.face_metrics import reference_sphere_face_scales, write_face_scales

nx, ny, nz = 10, 20, 10
radius = 6371000.0
south, dlon, dlat = 34.0, 1/12, 1/12
midlat = south + ny*dlat/2
# Fixed meter background representing the center latitude of this region.
dx0 = radius*np.cos(np.deg2rad(midlat))*np.deg2rad(dlon)
dy0 = radius*np.deg2rad(dlat)
xs, ys = reference_sphere_face_scales(nx, ny, nz, south, dlon, dlat, dx0, dy0, radius)
write_face_scales("dx_scale.pfb", "dy_scale.pfb", xs, ys, dx0, dy0, 10.0)
# Cartesian control: write np.ones_like(xs), np.ones_like(ys) to different files.
# Set ComputationalGrid.DX=dx0, DY=dy0, DZ=10, and meter-valued Lower/Upper.
```

Use separate output directories for each case. This change does not create or
overwrite simulation cases or start/push any production runs.

## Validation

`python3 test/python/face_metrics_geometry.py` compiles the actual C PFB reader
and geometry helper with address/undefined-behavior sanitizers. It checks all-one
and nonuniform fields, multiple subgrids, partition halos, physical halos,
invalid values/dimensions, and the legacy spherical formula. This is not an MPI
end-to-end solver convergence test. A full ParFlow build/run remains required on
a machine with the normal dependencies.

### Cached geometry access

Initialization stores the two horizontal distances, four side areas, horizontal
area and cell volume in eight cell-centered vectors with identical layouts.
Residual and Jacobian assembly acquire the needed data pointers at the start of
each subgrid loop. Each cell computes one index using the geometry subvector
(not the pressure index, whose ghost layout may differ), and formulas read the
cached arrays directly. There is no static pointer cache, per-cell subgrid-key
check or intermediate geometry struct on the FaceMetrics path. The original
reference-sphere calculation remains available when FaceMetrics is disabled.

### Original-algorithm compatibility

FaceMetrics retains the original Cartesian vertical density-derivative formulas,
symmetric boundary preparation and FluxBC residual/Jacobian correction algebra.
Only their distances, face areas and volumes use the geometry arrays. In
particular, FluxBC again applies `cp += op; op = 0`, and there are no temporary
`flux_diag_x/y/z` vectors or alternative vertical-derivative formulas. This
retains the original Jacobian approximations rather than introducing separate
derivative fixes as part of the geometry feature.

An extracted-code comparison against the original master (`7958b8af`) passed
35,000 all-one checks for internal flux/Jacobian terms, all six residual FluxBC
faces and all six symmetric boundary faces, including variable density and
layer thickness. This is algebra-level validation, not proof of identical MPI
solver trajectories; a complete all-one/control run is still required.
