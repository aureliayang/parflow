/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

/*****************************************************************************
*
* Module for initializing the problem structure.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"
#include <string.h>


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  PFModule  *geometries;
  PFModule  *domain;
  PFModule  *permeability;
  PFModule  *porosity;
  PFModule  *wells;
  PFModule *reservoirs;
  PFModule  *bc_pressure;
  PFModule  *specific_storage;  //sk
  PFModule  *x_slope;  //sk
  PFModule  *y_slope;  //sk
  PFModule  *wc_x;
  PFModule  *wc_y;
  PFModule  *mann;  //sk
  PFModule  *dz_mult;   //RMM
  PFModule  *FBx;   //RMM
  PFModule  *FBy;   //RMM
  PFModule  *FBz;   //RMM

  PFModule  *real_space_z;
  int site_data_not_formed;

  /* InitInstanceXtra arguments */
  Problem   *problem;
  Grid      *grid;
  Grid      *grid2d;
  double    *temp_data;
} InstanceXtra;


/* All ranks must reject invalid geometry together, before halo communication. */
static void CheckSphericalGeometry(int invalid, const char *message)
{
  amps_Invoice invoice = amps_NewInvoice("%i", &invalid);
  amps_AllReduce(amps_CommWorld, invoice, amps_Max);
  amps_FreeInvoice(invoice);
  if (invalid)
    InputError("%s%s\n", message, "");
}

/* Read an UNDISTRIBUTED PFB on every rank, including local halos. This avoids
 * applying cell-grid pfdist layouts to the overlapping face grids. */
static unsigned long long FacePFBWord(FILE *file, int bytes, int *invalid)
{
  unsigned long long word = 0;
  for (int b = 0; b < bytes; b++)
  {
    int c = fgetc(file);
    if (c == EOF) { *invalid = 1; return 0; }
    word = (word << 8) | (unsigned char)c;
  }
  return word;
}

static Vector *ReadFaceMetric(Grid *grid, char *filename, int axis)
{
  SubgridArray *all = NewSubgridArray();
  int is, invalid = 0;
  ForSubgridI(is, GridAllSubgrids(grid))
  {
    Subgrid *sg = DuplicateSubgrid(SubgridArraySubgrid(GridAllSubgrids(grid), is));
    if (axis == 0) SubgridNX(sg)++;
    else SubgridNY(sg)++;
    AppendSubgrid(sg, all);
  }
  Grid *fg = NewGrid(GetGridSubgrids(all), all);
  /* NewVectorType constructs communication packages even for read-only fields. */
  CreateComputePkgs(fg);
  Vector *v = NewVectorType(fg, 1, 2, vector_cell_centered);
  InitVectorAll(v, NAN);
  FILE *file = fopen(filename, "rb");
  CheckSphericalGeometry(file == NULL, "Cannot open face metric PFB (use a shared, undistributed file)");
  for (int b = 0; b < 3; b++) FacePFBWord(file, 8, &invalid);
  int dims[3];
  for (int b = 0; b < 3; b++) dims[b] = (int)FacePFBWord(file, 4, &invalid);
  int expected[3] = {BackgroundNX(GlobalsBackground) + (axis == 0),
                     BackgroundNY(GlobalsBackground) + (axis == 1),
                     BackgroundNZ(GlobalsBackground)};
  for (int b = 0; b < 3; b++)
  {
    invalid |= dims[b] != expected[b];
    FacePFBWord(file, 8, &invalid);
  }
  int blocks = (int)FacePFBWord(file, 4, &invalid);
  invalid |= blocks <= 0;
  CheckSphericalGeometry(invalid, "Face metric PFB has an invalid header or dimensions");
  for (int b = 0; b < blocks; b++)
  {
    int origin[3], size[3];
    for (int d = 0; d < 3; d++) origin[d] = (int)FacePFBWord(file, 4, &invalid);
    for (int d = 0; d < 3; d++) size[d] = (int)FacePFBWord(file, 4, &invalid);
    for (int d = 0; d < 3; d++)
    {
      int refinement = (int)FacePFBWord(file, 4, &invalid);
      invalid |= origin[d] < 0 || size[d] <= 0 || size[d] > dims[d]
                 || origin[d] > dims[d] - size[d] || refinement != 0;
    }
    CheckSphericalGeometry(invalid, "Face metric PFB contains an invalid or refined subgrid");
    for (int k = origin[2]; k < origin[2] + size[2]; k++)
      for (int j = origin[1]; j < origin[1] + size[1]; j++)
        for (int i = origin[0]; i < origin[0] + size[0]; i++)
        {
          unsigned long long bits = FacePFBWord(file, 8, &invalid);
          double value;
          memcpy(&value, &bits, sizeof(value));
          invalid |= !isfinite(value) || value <= 0.0;
          ForSubgridI(is, GridSubgrids(fg))
          {
            Subvector *s = VectorSubvector(v, is);
            if (i >= SubvectorIX(s) && i < SubvectorIX(s) + SubvectorNX(s)
                && j >= SubvectorIY(s) && j < SubvectorIY(s) + SubvectorNY(s)
                && k >= SubvectorIZ(s) && k < SubvectorIZ(s) + SubvectorNZ(s))
              SubvectorData(s)[SubvectorEltIndex(s, i, j, k)] = value;
          }
        }
  }
  invalid |= fgetc(file) != EOF;
  fclose(file);
  CheckSphericalGeometry(invalid, "Face metric PFB is truncated, has trailing data, or nonpositive/nonfinite scales");
  /* Fill physical exterior halos by constant extension. Inter-rank halos were
   * read directly from the same global file, including edge/corner values. */
  ForSubgridI(is, GridSubgrids(fg))
  {
    Subvector *s = VectorSubvector(v, is);
    for (int k = SubvectorIZ(s); k < SubvectorIZ(s) + SubvectorNZ(s); k++)
      for (int j = SubvectorIY(s); j < SubvectorIY(s) + SubvectorNY(s); j++)
        for (int i = SubvectorIX(s); i < SubvectorIX(s) + SubvectorNX(s); i++)
        {
          int ci = pfmin(pfmax(i, 0), dims[0] - 1);
          int cj = pfmin(pfmax(j, 0), dims[1] - 1);
          int ck = pfmin(pfmax(k, 0), dims[2] - 1);
          double val = SubvectorData(s)[SubvectorEltIndex(s, ci, cj, ck)];
          SubvectorData(s)[SubvectorEltIndex(s, i, j, k)] = val;
          invalid |= !isfinite(val) || val <= 0.0;
        }
  }
  CheckSphericalGeometry(invalid, "Face metric PFB does not cover the local grid and halo");
  return v;
}

static void PrepareFaceMetrics(ProblemData *pd)
{
  NameArray names = NA_NewNameArray("False True");
  char *key = "Solver.FaceMetrics.Enabled";
  int enabled = NA_NameToIndexExitOnError(names, GetStringDefault(key, "False"), key);
  NA_FreeNameArray(names);
  if (!enabled) return;
  names = NA_NewNameArray("False True");
  key = "Solver.Experimental.SphericalAccumulation";
  int spherical = NA_NameToIndexExitOnError(names, GetStringDefault(key, "False"), key);
  NA_FreeNameArray(names);
  CheckSphericalGeometry(spherical, "FaceMetrics and SphericalAccumulation are mutually exclusive");
  Background *bg = GlobalsBackground;
  int invalid = BackgroundIX(bg) != 0 || BackgroundIY(bg) != 0 || BackgroundIZ(bg) != 0
                || !isfinite(BackgroundDX(bg)) || BackgroundDX(bg) <= 0.0
                || !isfinite(BackgroundDY(bg)) || BackgroundDY(bg) <= 0.0;
  Grid *grid = VectorGrid(ProblemDataZmult(pd));
  int is;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    invalid |= SubgridRX(sg) || SubgridRY(sg) || SubgridRZ(sg);
  }
  CheckSphericalGeometry(invalid, "FaceMetrics requires a zero-origin unrefined index grid and positive finite meter spacing");
  pd->face_dx_scale = ReadFaceMetric(grid, GetString("Solver.FaceMetrics.DXScale.FileName"), 1);
  pd->face_dy_scale = ReadFaceMetric(grid, GetString("Solver.FaceMetrics.DYScale.FileName"), 0);
}

/* Validate reference-sphere geometry and prepare actual layer-thickness halos. */
static void PrepareSphericalGeometry(ProblemData *problem_data)
{
  Grid *grid = VectorGrid(ProblemDataZmult(problem_data));
  Background *bg = GlobalsBackground;
  NameArray spherical_na = NA_NewNameArray("False True");
  char *spherical_key = "Solver.Experimental.SphericalAccumulation";
  char *spherical_name = GetStringDefault(spherical_key, "False");
  int enabled = NA_NameToIndexExitOnError(spherical_na, spherical_name, spherical_key);
  NA_FreeNameArray(spherical_na);
  int is, i, j, k;
  int invalid = 0;
  if (!enabled && !problem_data->face_dx_scale)
    return;

  double reference_radius = 0.0;
  if (enabled)
  {
    reference_radius = GetDouble("Solver.SphericalGeometry.ReferenceRadius");
    CheckSphericalGeometry(!isfinite(reference_radius) || reference_radius <= 0.0,
                         "Solver.SphericalGeometry.ReferenceRadius must be finite and positive (meters)");
  }
  double dx = BackgroundDX(bg);
  double dy = BackgroundDY(bg);
  int ilo = BackgroundIX(bg), ihi = ilo + BackgroundNX(bg) - 1;
  int jlo = BackgroundIY(bg), jhi = jlo + BackgroundNY(bg) - 1;
  int klo = BackgroundIZ(bg), khi = klo + BackgroundNZ(bg) - 1;
  if (enabled)
  {
    double south = BackgroundY(bg) + jlo * dy;
    double north = BackgroundY(bg) + (jhi + 1) * dy;
    invalid = !isfinite(dx) || !isfinite(dy) || dx <= 0.0 || dy <= 0.0
              || !isfinite(south) || !isfinite(north) || south <= -90.0 || north >= 90.0
              || dx * BackgroundNX(bg) > 360.0;
    ForSubgridI(is, GridSubgrids(grid))
    {
      Subgrid *sg = GridSubgrid(grid, is);
      invalid |= SubgridRX(sg) != 0 || SubgridRY(sg) != 0 || SubgridRZ(sg) != 0;
    }
    CheckSphericalGeometry(invalid,
                           "Spherical geometry requires an unrefined grid, positive angular spacing in degrees, "
                           "longitude span <= 360 degrees and latitude boundaries strictly between -90 and 90 degrees");

  }

  Vector *thickness = NewVectorType(grid, 1, 3, vector_cell_centered);
  InitVectorAll(thickness, NAN);

  /* Keep a fully exchanged multiplier copy while constructing geometric halos. */
  invalid = 0;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    Subvector *ts = VectorSubvector(thickness, is);
    Subvector *zs = VectorSubvector(ProblemDataZmult(problem_data), is);
    double *td = SubvectorData(ts), *zd = SubvectorData(zs);
    double dz = SubgridDZ(sg);
    for (k = SubgridIZ(sg); k < SubgridIZ(sg) + SubgridNZ(sg); k++)
      for (j = SubgridIY(sg); j < SubgridIY(sg) + SubgridNY(sg); j++)
        for (i = SubgridIX(sg); i < SubgridIX(sg) + SubgridNX(sg); i++)
        {
          int it = SubvectorEltIndex(ts, i, j, k);
          int iz = SubvectorEltIndex(zs, i, j, k);
          td[it] = zd[iz];
          invalid |= !isfinite(td[it]) || td[it] <= 0.0
                     || !isfinite(dz) || dz <= 0.0;
        }
  }
  CheckSphericalGeometry(invalid, "Spherical layer thickness must be finite and positive");

  VectorUpdateCommHandle *handle = InitVectorUpdate(thickness, VectorUpdateGodunov);
  FinalizeVectorUpdate(handle);

  /* Extend actual thickness into the physical-boundary halo. No radius field is needed. */
  invalid = 0;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    Subvector *ts = VectorSubvector(thickness, is);
    Subvector *zs = VectorSubvector(ProblemDataZmult(problem_data), is);
    double *td = SubvectorData(ts), *zd = SubvectorData(zs);
    for (k = SubgridIZ(sg) - 1; k <= SubgridIZ(sg) + SubgridNZ(sg); k++)
      for (j = SubgridIY(sg) - 1; j <= SubgridIY(sg) + SubgridNY(sg); j++)
        for (i = SubgridIX(sg) - 1; i <= SubgridIX(sg) + SubgridNX(sg); i++)
        {
          int ci = pfmin(pfmax(i, ilo), ihi);
          int cj = pfmin(pfmax(j, jlo), jhi);
          int ck = pfmin(pfmax(k, klo), khi);
          int it = SubvectorEltIndex(ts, ci, cj, ck);
          double multiplier = td[it];
          zd[SubvectorEltIndex(zs, i, j, k)] = multiplier;
          invalid |= !isfinite(multiplier) || multiplier <= 0.0;
        }
  }
  CheckSphericalGeometry(invalid, "Spherical thickness ghost construction failed");
  FreeVector(thickness);
  ProblemDataReferenceRadius(problem_data) = reference_radius;
}

/*--------------------------------------------------------------------------
 * SetProblemData
 *--------------------------------------------------------------------------*/

void          SetProblemData(
                             ProblemData *problem_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule      *geometries = (instance_xtra->geometries);
  PFModule      *domain = (instance_xtra->domain);
  PFModule      *permeability = (instance_xtra->permeability);
  PFModule      *porosity = (instance_xtra->porosity);
  PFModule      *wells = (instance_xtra->wells);
  PFModule      *reservoirs = (instance_xtra->reservoirs);
  PFModule      *bc_pressure = (instance_xtra->bc_pressure);
  PFModule      *specific_storage = (instance_xtra->specific_storage);    //sk
  PFModule      *x_slope = (instance_xtra->x_slope);         //sk
  PFModule      *y_slope = (instance_xtra->y_slope);         //sk
  PFModule      *wc_x = (instance_xtra->wc_x);
  PFModule      *wc_y = (instance_xtra->wc_y);
  PFModule      *mann = (instance_xtra->mann);            //sk
  PFModule      *dz_mult = (instance_xtra->dz_mult);                //rmm
  PFModule      *FBx = (instance_xtra->FBx);                //rmm
  PFModule      *FBy = (instance_xtra->FBy);                //rmm
  PFModule      *FBz = (instance_xtra->FBz);                //rmm

  PFModule      *real_space_z = (instance_xtra->real_space_z);

  /* Note: the order in which these modules are called is important */

  if ((instance_xtra->site_data_not_formed))
  {
    PFModuleInvokeType(GeometriesInvoke, geometries, (problem_data));
    PFModuleInvokeType(DomainInvoke, domain, (problem_data));

    PFModuleInvokeType(SubsrfSimInvoke, permeability,
                       (problem_data,
                        ProblemDataPermeabilityX(problem_data),
                        ProblemDataPermeabilityY(problem_data),
                        ProblemDataPermeabilityZ(problem_data),
                        ProblemDataNumSolids(problem_data),
                        ProblemDataSolids(problem_data),
                        ProblemDataGrSolids(problem_data)));
    PFModuleInvokeType(PorosityInvoke, porosity,
                       (problem_data,
                        ProblemDataPorosity(problem_data),
                        ProblemDataNumSolids(problem_data),
                        ProblemDataSolids(problem_data),
                        ProblemDataGrSolids(problem_data)));
    PFModuleInvokeType(SpecStorageInvoke, specific_storage,                   //sk
                       (problem_data,
                        ProblemDataSpecificStorage(problem_data)));
    PFModuleInvokeType(SlopeInvoke, x_slope,                   //sk
                       (problem_data,
                        ProblemDataTSlopeX(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(SlopeInvoke, y_slope,                   //sk
                       (problem_data,
                        ProblemDataTSlopeY(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(ChannelWidthInvoke, wc_x,
                       (problem_data,
                        ProblemDataChannelWidthX(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(ChannelWidthInvoke, wc_y,
                       (problem_data,
                        ProblemDataChannelWidthY(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(ManningsInvoke, mann,                   //sk
                       (problem_data,
                        ProblemDataMannings(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(dzScaleInvoke, dz_mult,                   //RMM
                       (problem_data,
                        ProblemDataZmult(problem_data)));

    /* Prepare reference-sphere geometry only after actual layer thicknesses. */
    PrepareFaceMetrics(problem_data);
    PrepareSphericalGeometry(problem_data);


    PFModuleInvokeType(realSpaceZInvoke, real_space_z,
                       (problem_data,
                        ProblemDataRealSpaceZ(problem_data)));

    PFModuleInvokeType(FBxInvoke, FBx,                   //RMM
                       (problem_data,
                        ProblemDataFBx(problem_data)));
    PFModuleInvokeType(FByInvoke, FBy,                   //RMM
                       (problem_data,
                        ProblemDataFBy(problem_data)));
    PFModuleInvokeType(FBzInvoke, FBz,                   //RMM
                       (problem_data,
                        ProblemDataFBz(problem_data)));


    (instance_xtra->site_data_not_formed) = 0;
  }
  PFModuleInvokeType(WellPackageInvoke, wells, (problem_data));
  PFModuleInvokeType(ReservoirPackageInvoke, reservoirs, (problem_data));
  PFModuleInvokeType(BCPressurePackageInvoke, bc_pressure, (problem_data));
}


/*--------------------------------------------------------------------------
 * SetProblemDataInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SetProblemDataInitInstanceXtra(
                                          Problem *problem,
                                          Grid *   grid,
                                          Grid *   grid2d,
                                          double * temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
    (instance_xtra->problem) = problem;

  /*-----------------------------------------------------------------------
   * Initialize data associated with `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
    (instance_xtra->grid) = grid;

  /*-----------------------------------------------------------------------
   * Initialize data associated with `grid2d'
   *-----------------------------------------------------------------------*/

  if (grid2d != NULL)
    (instance_xtra->grid2d) = grid2d;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
    (instance_xtra->temp_data) = temp_data;

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->geometries) =
      PFModuleNewInstanceType(GeometriesInitInstanceXtraInvoke, ProblemGeometries(problem), (grid));
    (instance_xtra->domain) =
      PFModuleNewInstanceType(DomainInitInstanceXtraInvoke, ProblemDomain(problem), (grid));
    (instance_xtra->permeability) =
      PFModuleNewInstanceType(SubsrfSimInitInstanceXtraInvoke, ProblemPermeability(problem), (grid, temp_data));
    (instance_xtra->porosity) =
      PFModuleNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              ProblemPorosity(problem), (grid, temp_data));
    (instance_xtra->specific_storage) =                                       //sk
                                        PFModuleNewInstance(ProblemSpecStorage(problem), ());
    (instance_xtra->x_slope) =                                       //sk
                               PFModuleNewInstanceType(SlopeInitInstanceXtraInvoke,
                                                       ProblemXSlope(problem), (grid, grid2d));
    (instance_xtra->y_slope) =                                       //sk
                               PFModuleNewInstanceType(SlopeInitInstanceXtraInvoke,
                                                       ProblemYSlope(problem), (grid, grid2d));

    (instance_xtra->wc_x) =
      PFModuleNewInstanceType(ChannelWidthInitInstanceXtraInvoke,
                              ProblemXChannelWidth(problem), (grid, grid2d));
    (instance_xtra->wc_y) =
      PFModuleNewInstanceType(ChannelWidthInitInstanceXtraInvoke,
                              ProblemYChannelWidth(problem), (grid, grid2d));

    (instance_xtra->mann) =                                       //sk
                            PFModuleNewInstanceType(ManningsInitInstanceXtraInvoke,
                                                    ProblemMannings(problem), (grid, grid2d));
    (instance_xtra->dz_mult) =                                      //RMM
                               PFModuleNewInstance(ProblemdzScale(problem), ());

    (instance_xtra->FBx) =                                      //RMM
                           PFModuleNewInstance(ProblemFBx(problem), ());
    (instance_xtra->FBy) =                                      //RMM
                           PFModuleNewInstance(ProblemFBy(problem), ());
    (instance_xtra->FBz) =                                      //RMM
                           PFModuleNewInstance(ProblemFBz(problem), ());

    (instance_xtra->real_space_z) =
      PFModuleNewInstance(ProblemRealSpaceZ(problem), ());

    (instance_xtra->site_data_not_formed) = 1;

    (instance_xtra->wells) =
      PFModuleNewInstance(ProblemWellPackage(problem), ());

    (instance_xtra->reservoirs) =
      PFModuleNewInstance(ProblemReservoirPackage(problem), ());

    (instance_xtra->bc_pressure) =
      PFModuleNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              ProblemBCPressurePackage(problem), (problem));
  }
  else
  {
    PFModuleReNewInstanceType(GeometriesInitInstanceXtraInvoke,
                              (instance_xtra->geometries), (grid));
    PFModuleReNewInstanceType(DomainInitInstanceXtraInvoke,
                              (instance_xtra->domain), (grid));
    PFModuleReNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              (instance_xtra->permeability),
                              (grid, temp_data));
    PFModuleReNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              (instance_xtra->porosity),
                              (grid, temp_data));
    PFModuleReNewInstance((instance_xtra->specific_storage), ());        //sk
    PFModuleReNewInstanceType(SlopeInitInstanceXtraInvoke,
                              (instance_xtra->x_slope), (grid, grid2d));        //sk
    PFModuleReNewInstanceType(SlopeInitInstanceXtraInvoke,
                              (instance_xtra->y_slope), (grid, grid2d));        //sk
    PFModuleReNewInstanceType(ChannelWidthInitInstanceXtraInvoke,
                              (instance_xtra->wc_x), (grid, grid2d));
    PFModuleReNewInstanceType(ChannelWidthInitInstanceXtraInvoke,
                              (instance_xtra->wc_y), (grid, grid2d));
    PFModuleReNewInstanceType(ManningsInitInstanceXtraInvoke,
                              (instance_xtra->mann), (grid, grid2d));        //sk
    PFModuleReNewInstance((instance_xtra->dz_mult), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBx), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBy), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBz), ());        //RMM

    PFModuleReNewInstance((instance_xtra->real_space_z), ());
    PFModuleReNewInstance((instance_xtra->wells), ());
    PFModuleReNewInstance((instance_xtra->reservoirs), ());
    PFModuleReNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              (instance_xtra->bc_pressure), (problem));
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SetProblemDataFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->bc_pressure);
    PFModuleFreeInstance(instance_xtra->wells);
    PFModuleFreeInstance(instance_xtra->reservoirs);

    PFModuleFreeInstance(instance_xtra->geometries);
    PFModuleFreeInstance(instance_xtra->domain);
    PFModuleFreeInstance(instance_xtra->porosity);
    PFModuleFreeInstance(instance_xtra->permeability);
    PFModuleFreeInstance(instance_xtra->specific_storage);       //sk
    PFModuleFreeInstance(instance_xtra->x_slope);       //sk
    PFModuleFreeInstance(instance_xtra->y_slope);       //sk
    PFModuleFreeInstance(instance_xtra->wc_x);
    PFModuleFreeInstance(instance_xtra->wc_y);
    PFModuleFreeInstance(instance_xtra->mann);       //sk
    PFModuleFreeInstance(instance_xtra->dz_mult);       // RMM
    PFModuleFreeInstance(instance_xtra->FBx);       // RMM
    PFModuleFreeInstance(instance_xtra->FBy);       // RMM
    PFModuleFreeInstance(instance_xtra->FBz);       // RMM

    PFModuleFreeInstance(instance_xtra->real_space_z);
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetProblemDataNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SetProblemDataNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;


#if 0
  public_xtra = ctalloc(PublicXtra, 1);
#endif
  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SetProblemDataFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetProblemDataSizeOfTempData
 *--------------------------------------------------------------------------*/

int       SetProblemDataSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;


  /* set `sz' to max of each of the called modules */
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->geometries));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->porosity));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->permeability));

  return sz;
}
