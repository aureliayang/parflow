# ParFlow-CoLM Coupling

ParFlow-CoLM is a renewed coupling effort that connects ParFlow's
three-dimensional variably saturated groundwater and overland-flow
solver with the updated water and energy modules of CoLM. Following the
original ParFlow-CLM coupling framework, CoLM provides land-surface net
water and energy fluxes to ParFlow as source/sink terms, while ParFlow
returns soil moisture and pressure head to CoLM to close the coupled
water and energy balance.

CoLM is the latest generation of the Common Land Model developed and
maintained by the Common Land Model team at Sun Yat-sen University, led
by Prof. Yongjiu Dai. This ParFlow-CoLM coupling was developed in
collaboration with the Common Land Model team.

The current ParFlow-CoLM effort is described in:

[Yang, C., Sun, A., Zhang, S., Dai, Y., Kollet, S., & Maxwell, R. (2026). 20 years of trials and insights: bridging legacy and next generation in ParFlow and Land Surface Model Coupling. Geoscientific Model Development, 19(5), 1849-1866.](https://doi.org/10.5194/gmd-19-1849-2026)

## Enabling CoLM

CoLM is selected through the land surface model key:

```python
run.Solver.LSM = "CoLM"
```

or in TCL:

```tcl
pfset Solver.LSM CoLM
```

The coupled configuration requires ParFlow to be built with land surface
model support, currently enabled through the existing CLM build option:

```shell
cmake ../parflow -DPARFLOW_HAVE_CLM=ON ...
```

The example in `pfsimulator/colm/examples/colm_version` shows a compact
ParFlow-CoLM setup. It sets `Solver.LSM = "CoLM"` while continuing to use
the existing `Solver.CLM.*` ParFlow keys for shared land-model coupling
controls such as forcing, output, restart, and root-zone settings.

## Main CoLM Input Files

The example uses two primary CoLM input files:

- `CoLM_nlfile.nml`
- `CoLM_readin.dat`

Additional files in the example directory provide meteorological forcing,
snow optics parameters, ParFlow terrain and subsurface inputs, and run
scripts.

### `CoLM_nlfile.nml`

`CoLM_nlfile.nml` is the CoLM run-control Fortran namelist read during
CoLM initialization. In the example, it contains the `nl_colm` namelist
and sets basic run-control options:

- `DEF_simulation_time%start_year`
- `DEF_simulation_time%start_month`
- `DEF_simulation_time%start_day`
- `DEF_simulation_time%start_sec`
- `DEF_WRST_FREQ`
- `DEF_hotstart`

For example:

```fortran
&nl_colm
   DEF_simulation_time%start_year   = 2010
   DEF_simulation_time%start_month  = 1
   DEF_simulation_time%start_day    = 1
   DEF_simulation_time%start_sec    = 0
   DEF_WRST_FREQ = 'DAILY'
   DEF_hotstart = .false.
/
```

The current coupling reads this file as `CoLM_nlfile.nml` from the run
directory.

### `CoLM_readin.dat`

`CoLM_readin.dat` is a formatted coupling-side table that provides CoLM
patch metadata and vertically resolved soil properties for each ParFlow
horizontal grid cell. The reader loops over the global ParFlow `x` and
`y` grid and reads the row associated with each cell.

The current reader expects each row in the following order:

1. `i`
2. `j`
3. `patchclass`
4. `patchlonr`
5. `patchlatr`
6. `int_soil_grav_l(1:8)`
7. `int_soil_sand_l(1:8)`
8. `int_soil_clay_l(1:8)`
9. `int_soil_oc_l(1:8)`
10. `int_soil_bd_l(1:8)`

`patchclass` identifies the CoLM land-cover or patch-type class, while
`patchlonr` and `patchlatr` provide the patch longitude and latitude.
The five vertically resolved soil-property groups represent coarse
fragments, sand, clay, soil organic carbon, and fine-earth bulk density,
respectively.

The table provides eight levels of soil texture and physical property
data. In the ParFlow-CoLM configuration described by Yang et al. (2026)
and used by this example, CoLM uses a 10-layer soil column, while the
legacy ParFlow-CLM coupling uses a 4-layer soil column. Only eight
soil-property levels are provided in `CoLM_readin.dat` because the
coupling maps these data onto the 10 CoLM soil layers by reusing the
same properties for layers 1 and 2, and again for layers 9 and 10. Thus,
layers 1 and 2 share identical soil properties, layers 9 and 10 share
identical soil properties, and the remaining layers use the
corresponding entries from the eight-level input table.

In the example, the file contains 30 rows, matching the 6 by 5
horizontal grid used by `unname_test.py`.

## Example Directory

The example directory is:

```text
pfsimulator/colm/examples/colm_version
```

Important files include:

- `unname_test.py`: ParFlow Python input script for the coupled example.
- `CoLM_nlfile.nml`: CoLM namelist.
- `CoLM_readin.dat`: CoLM patch metadata and vertically resolved
  soil-property table.
- `station0.txt`: 1-D meteorological forcing used by
  `Solver.CLM.MetForcing = "1D"`.
- `snicar_par.dat`: snow optics parameter file used by CoLM snow physics.
- `unname.slopex.pfb`, `unname.slopey.pfb`, `unname.manning.pfb`, and
  `unname.subsur.pfb`: ParFlow terrain, Manning's coefficient, and
  subsurface inputs used by the example.
