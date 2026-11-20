/* Experimental orthogonal face metrics. Included after parflow.h. */
#ifndef PF_FACE_METRICS_H
#define PF_FACE_METRICS_H

typedef struct {
  double dx, dy, west, east, south, north, horizontal, volume;
} PFHorizontalGeometry;

typedef struct {
  Vector *key;
  int is;
  Subvector *sub[8];
  double *data[8];
} PFFaceMetricAccessCache;
static PFFaceMetricAccessCache pf_face_metric_cache = {0};

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
    if (pf_face_metric_cache.key != pd->face_geom_dx || pf_face_metric_cache.is != is)
    {
      pf_face_metric_cache.key = pd->face_geom_dx;
      pf_face_metric_cache.is = is;
      pf_face_metric_cache.sub[0] = VectorSubvector(pd->face_geom_dx, is);
      pf_face_metric_cache.sub[1] = VectorSubvector(pd->face_geom_dy, is);
      pf_face_metric_cache.sub[2] = VectorSubvector(pd->face_geom_west, is);
      pf_face_metric_cache.sub[3] = VectorSubvector(pd->face_geom_east, is);
      pf_face_metric_cache.sub[4] = VectorSubvector(pd->face_geom_south, is);
      pf_face_metric_cache.sub[5] = VectorSubvector(pd->face_geom_north, is);
      pf_face_metric_cache.sub[6] = VectorSubvector(pd->face_geom_horizontal, is);
      pf_face_metric_cache.sub[7] = VectorSubvector(pd->face_geom_volume, is);
      pf_face_metric_cache.data[0] = SubvectorData(pf_face_metric_cache.sub[0]);
      pf_face_metric_cache.data[1] = SubvectorData(pf_face_metric_cache.sub[1]);
      pf_face_metric_cache.data[2] = SubvectorData(pf_face_metric_cache.sub[2]);
      pf_face_metric_cache.data[3] = SubvectorData(pf_face_metric_cache.sub[3]);
      pf_face_metric_cache.data[4] = SubvectorData(pf_face_metric_cache.sub[4]);
      pf_face_metric_cache.data[5] = SubvectorData(pf_face_metric_cache.sub[5]);
      pf_face_metric_cache.data[6] = SubvectorData(pf_face_metric_cache.sub[6]);
      pf_face_metric_cache.data[7] = SubvectorData(pf_face_metric_cache.sub[7]);
    }
    int fm_idx = SubvectorEltIndex(pf_face_metric_cache.sub[0], i, j, k);
    g.dx = pf_face_metric_cache.data[0][fm_idx];
    g.dy = pf_face_metric_cache.data[1][fm_idx];
    g.west = pf_face_metric_cache.data[2][fm_idx];
    g.east = pf_face_metric_cache.data[3][fm_idx];
    g.south = pf_face_metric_cache.data[4][fm_idx];
    g.north = pf_face_metric_cache.data[5][fm_idx];
    g.horizontal = pf_face_metric_cache.data[6][fm_idx];
    g.volume = pf_face_metric_cache.data[7][fm_idx];
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
