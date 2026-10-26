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

/* Static geometry for the experimental spherical groundwater path.
 * Input contains the whole rectangular grid, including inactive cells, but no
 * ghosts. Three allocated halo layers permit the existing Godunov exchange
 * (27-point core plus extended faces); the flow loops need only the first.
 * This also fills corner/edge halos, unlike VectorUpdateAll's six-face stencil.
 */
static void ReadSphericalRadius(ProblemData *problem_data)
{
  Grid *grid = VectorGrid(ProblemDataZmult(problem_data));
  Background *bg = GlobalsBackground;
  int enabled = GetIntDefault("Solver.Experimental.SphericalAccumulation", 0);
  int is, i, j, k;
  int invalid = enabled != 0 && enabled != 1;
  CheckSphericalGeometry(invalid, "Solver.Experimental.SphericalAccumulation must be 0 or 1");
  if (!enabled)
    return;

  char *filename = GetString("Solver.SphericalGeometry.RadiusCenter.FileName");
  double dx = BackgroundDX(bg);
  double dy = BackgroundDY(bg);
  int ilo = BackgroundIX(bg), ihi = ilo + BackgroundNX(bg) - 1;
  int jlo = BackgroundIY(bg), jhi = jlo + BackgroundNY(bg) - 1;
  int klo = BackgroundIZ(bg), khi = klo + BackgroundNZ(bg) - 1;
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

  /* Check the global dimensions before ReadPFBinary writes subgrid data.
   * Use the same distributed-file opening convention as ReadPFBinary. */
  amps_File file = amps_FFopen(amps_CommWorld, filename, "rb", 0);
  invalid = file == NULL;
  CheckSphericalGeometry(invalid, "Cannot open spherical radius PFB (distribute it for the configured process grid)");
  if (amps_Rank(amps_CommWorld) == 0)
  {
    double origin[3];
    int dimensions[3];
    amps_ReadDouble(file, origin, 3);
    amps_ReadInt(file, dimensions, 3);
    invalid = dimensions[0] != BackgroundNX(bg) || dimensions[1] != BackgroundNY(bg)
              || dimensions[2] != BackgroundNZ(bg);
  }
  amps_FFclose(file);
  CheckSphericalGeometry(invalid, "Spherical radius PFB dimensions must equal ComputationalGrid.NX, NY, NZ");

  Vector *radius = NewVectorType(grid, 1, 3, vector_cell_centered);
  Vector *thickness = NewVectorType(grid, 1, 3, vector_cell_centered);
  InitVectorAll(radius, NAN);
  InitVectorAll(thickness, NAN);
  ReadPFBinary(filename, radius);

  /* Keep a fully exchanged multiplier copy while constructing geometric halos. */
  invalid = 0;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    Subvector *rs = VectorSubvector(radius, is);
    Subvector *ts = VectorSubvector(thickness, is);
    Subvector *zs = VectorSubvector(ProblemDataZmult(problem_data), is);
    double *rd = SubvectorData(rs), *td = SubvectorData(ts), *zd = SubvectorData(zs);
    double dz = SubgridDZ(sg);
    for (k = SubgridIZ(sg); k < SubgridIZ(sg) + SubgridNZ(sg); k++)
      for (j = SubgridIY(sg); j < SubgridIY(sg) + SubgridNY(sg); j++)
        for (i = SubgridIX(sg); i < SubgridIX(sg) + SubgridNX(sg); i++)
        {
          int ir = SubvectorEltIndex(rs, i, j, k);
          int it = SubvectorEltIndex(ts, i, j, k);
          int iz = SubvectorEltIndex(zs, i, j, k);
          td[it] = zd[iz];
          invalid |= !isfinite(rd[ir]) || !isfinite(td[it]) || td[it] <= 0.0
                     || !isfinite(dz) || dz <= 0.0 || rd[ir] <= 0.5 * dz * td[it];
        }
  }
  CheckSphericalGeometry(invalid, "Radius PFB must contain finite positive center radii in meters in every grid cell; layer thickness must be finite and positive");

  VectorUpdateCommHandle *handle = InitVectorUpdate(radius, VectorUpdateGodunov);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(thickness, VectorUpdateGodunov);
  FinalizeVectorUpdate(handle);

  /* Check adjacent radial centers before extrapolating physical-boundary halos.
   * All columns must be contiguous and k must increase radially outward. */
  invalid = 0;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    Subvector *rs = VectorSubvector(radius, is);
    Subvector *ts = VectorSubvector(thickness, is);
    double *rd = SubvectorData(rs), *td = SubvectorData(ts);
    double dz = SubgridDZ(sg);
    for (k = SubgridIZ(sg); k < SubgridIZ(sg) + SubgridNZ(sg); k++)
      for (j = SubgridIY(sg); j < SubgridIY(sg) + SubgridNY(sg); j++)
        for (i = SubgridIX(sg); i < SubgridIX(sg) + SubgridNX(sg); i++)
          if (k < khi)
          {
            int ir = SubvectorEltIndex(rs, i, j, k);
            int iu = SubvectorEltIndex(rs, i, j, k + 1);
            int it = SubvectorEltIndex(ts, i, j, k);
            int itu = SubvectorEltIndex(ts, i, j, k + 1);
            double expected = 0.5 * dz * (td[it] + td[itu]);
            double separation = rd[iu] - rd[ir];
            invalid |= !isfinite(separation) || separation <= 0.0
                       || fabs(separation - expected) > 1.0e-6 + 1.0e-8 * expected;
          }
  }
  CheckSphericalGeometry(invalid, "Radius centers must increase with k and adjacent radial layers must touch (center spacing equals the mean of their thicknesses)");

  /* Fill the one-cell shell outside the global rectangular box. Horizontally
   * extend the nearest column; radially extend by its boundary-cell thickness.
   * Interior inactive cells keep their explicitly supplied PFB geometry. */
  invalid = 0;
  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *sg = GridSubgrid(grid, is);
    Subvector *rs = VectorSubvector(radius, is);
    Subvector *ts = VectorSubvector(thickness, is);
    Subvector *zs = VectorSubvector(ProblemDataZmult(problem_data), is);
    double *rd = SubvectorData(rs), *td = SubvectorData(ts), *zd = SubvectorData(zs);
    double dz = SubgridDZ(sg);
    for (k = SubgridIZ(sg) - 1; k <= SubgridIZ(sg) + SubgridNZ(sg); k++)
      for (j = SubgridIY(sg) - 1; j <= SubgridIY(sg) + SubgridNY(sg); j++)
        for (i = SubgridIX(sg) - 1; i <= SubgridIX(sg) + SubgridNX(sg); i++)
        {
          int ci = pfmin(pfmax(i, ilo), ihi);
          int cj = pfmin(pfmax(j, jlo), jhi);
          int ck = pfmin(pfmax(k, klo), khi);
          int ir = SubvectorEltIndex(rs, i, j, k);
          int it = SubvectorEltIndex(ts, ci, cj, ck);
          int ic = SubvectorEltIndex(rs, ci, cj, ck);
          double multiplier = td[it];
          rd[ir] = rd[ic] + (k - ck) * dz * multiplier;
          zd[SubvectorEltIndex(zs, i, j, k)] = multiplier;
          invalid |= !isfinite(rd[ir]) || !isfinite(multiplier) || multiplier <= 0.0
                     || rd[ir] <= 0.5 * dz * multiplier;
        }
  }
  CheckSphericalGeometry(invalid, "Spherical radius/thickness ghost construction failed");
  FreeVector(thickness);
  ProblemDataRadiusCenter(problem_data) = radius;
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

    /* Static spherical geometry is loaded only after actual layer thicknesses. */
    ReadSphericalRadius(problem_data);


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
