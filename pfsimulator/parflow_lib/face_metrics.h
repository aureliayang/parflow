/* Experimental orthogonal face metrics. Included after parflow.h. */
#ifndef PF_FACE_METRICS_H
#define PF_FACE_METRICS_H

typedef struct {
  double dx, dy, west, east, south, north, horizontal, volume;
} PFHorizontalGeometry;

static inline double PFFaceScale(Vector *v, int is, int i, int j, int k)
{
  Subvector *s = VectorSubvector(v, is);
  return SubvectorData(s)[SubvectorEltIndex(s, i, j, k)];
}

static inline PFHorizontalGeometry PFGetHorizontalGeometry(
  ProblemData *pd, int is, int i, int j, int k,
  double dx, double dy, double thickness, double dlambda, double phi0,
  double dphi, int refinement)
{
  PFHorizontalGeometry g;
  if (pd->face_geom_dx)
  {
    /* FaceMetrics geometry is static for a run: read the values prepared at
     * initialization instead of recomputing face averages and areas here. */
    g.dx = PFFaceScale(pd->face_geom_dx, is, i, j, k);
    g.dy = PFFaceScale(pd->face_geom_dy, is, i, j, k);
    g.west = PFFaceScale(pd->face_geom_west, is, i, j, k);
    g.east = PFFaceScale(pd->face_geom_east, is, i, j, k);
    g.south = PFFaceScale(pd->face_geom_south, is, i, j, k);
    g.north = PFFaceScale(pd->face_geom_north, is, i, j, k);
    g.horizontal = PFFaceScale(pd->face_geom_horizontal, is, i, j, k);
    g.volume = PFFaceScale(pd->face_geom_volume, is, i, j, k);
    return g;
  }
  if (pd->face_dx_scale)
  {
    double xs = PFFaceScale(pd->face_dx_scale, is, i, j, k);
    double xn = PFFaceScale(pd->face_dx_scale, is, i, j + 1, k);
    double yw = PFFaceScale(pd->face_dy_scale, is, i, j, k);
    double ye = PFFaceScale(pd->face_dy_scale, is, i + 1, j, k);
    g.dx = dx * 0.5 * (xs + xn);
    g.dy = dy * 0.5 * (yw + ye);
    g.west = dy * yw * thickness;
    g.east = dy * ye * thickness;
    g.south = dx * xs * thickness;
    g.north = dx * xn * thickness;
    g.horizontal = g.dx * g.dy;
  }
  else
  {
    double r = ProblemDataReferenceRadius(pd);
    double factor = pow(2.0, refinement);
    double dp = dphi / factor, dl = dlambda / factor;
    double ps = phi0 + j * dp, pn = ps + dp, pc = 0.5 * (ps + pn);
    g.dx = r * cos(pc) * dl;
    g.dy = r * dp;
    g.west = g.east = r * dp * thickness;
    g.south = r * cos(ps) * dl * thickness;
    g.north = r * cos(pn) * dl * thickness;
    g.horizontal = r * r * dl * 2.0 * cos(pc) * sin(0.5 * dp);
  }
  g.volume = g.horizontal * thickness;
  return g;
}
#endif
