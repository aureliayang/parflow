# Experimental PFMG box transfer

This branch adds a runtime choice for ordinary PFMG. Both choices use the same
matrix coefficient calculations and solver settings; PFMGOctree and SMG retain
their existing paths. The default is the original point insertion.

Tcl:

```tcl
pfset Solver.Linear.Preconditioner PFMG
pfset Solver.Linear.Preconditioner.PFMG.HypreTransfer Point
# Change Point to Box for the comparison run.
```

Python:

```python
run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PFMG.HypreTransfer = 'Box'
```

Box mode packs owned cells (excluding ParFlow ghost cells) in HYPRE's public API
order: stencil entry, x, y, z, from fastest to slowest. Each local Subgrid supplies
one matrix box and one RHS box. Symmetric and nonsymmetric matrices, including
the existing overland-flow coefficient substitutions, use the same arithmetic
as Point mode. Solution retrieval remains point-wise in both modes.

Build using the existing working HYPRE-CUDA configuration. This option does not
enable CUDA by itself. CoLM is not required. No new solver algorithm is introduced.

First run a small case with identical inputs, decomposition, and solver settings
in Point and Box modes. Compare pressure and saturation outputs within the case's
accepted tolerance, and compare convergence/iteration counts. Include an overland
case if that is used in production, and test more than one MPI rank.

Then compare the large box case. ParFlow's existing `HYPRE_Copies` timer includes
matrix/RHS transfer, assembly, and solution retrieval; `PFMG` measures the solve
call. Record these and total runtime. Packing/allocation overhead is included in
the comparison, and a speedup is not guaranteed.

An isolated C harness exercised the actual transfer routines against mock HYPRE
insertion functions. Point and Box produced identical coefficient/RHS streams
for two subgrids with ghost cells and nonzero origins, symmetric/nonsymmetric
matrices, and with/without overland coefficients. Address/undefined-behavior
sanitizers passed. This checks packing, not HYPRE integration or convergence.

The shared grid setup also now registers every local Subgrid inside the loop
(previously only the last extent was registered) and clears the grid handle
correctly after destruction. These fixes are needed for valid multi-subgrid setup.

The implementation has not yet been validated with a full HYPRE-CUDA build or a
Point/Box solution comparison. Those checks must be performed in the working
server environment before treating this branch as validated.
