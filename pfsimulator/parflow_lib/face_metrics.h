/* Experimental orthogonal face metrics. Included after parflow.h. */
#ifndef PF_FACE_METRICS_H
#define PF_FACE_METRICS_H

typedef struct {
  double dx, dy, west, east, south, north, horizontal, volume;
} PFHorizontalGeometry;

#ifdef __CUDACC__
#define PF_FACE_METRIC_HD __host__ __device__
#else
#define PF_FACE_METRIC_HD
#endif

/* Retained for the non-FaceMetrics spherical fallback. The active FaceMetrics
 * BoxLoop path reads cached arrays directly. */
static inline PF_FACE_METRIC_HD double PFFaceScale(Vector *v, int is, int i, int j, int k)
{
#ifdef __CUDA_ARCH__
  (void)v; (void)is; (void)i; (void)j; (void)k;
  return 1.0;
#else
  Subvector *s = VectorSubvector(v, is);
  return SubvectorData(s)[SubvectorEltIndex(s, i, j, k)];
#endif
}

static inline PF_FACE_METRIC_HD PFHorizontalGeometry PFGetHorizontalGeometry(
  ProblemData *pd, int is, int i, int j, int k,
  double dx, double dy, double thickness, double dlambda, double phi0,
  double dphi, int refinement)
{
  PFHorizontalGeometry g;
#ifdef __CUDA_ARCH__
  (void)pd; (void)is; (void)i; (void)j; (void)k;
  (void)dlambda; (void)phi0; (void)dphi; (void)refinement;
  g.dx = dx;
  g.dy = dy;
  g.west = g.east = dy * thickness;
  g.south = g.north = dx * thickness;
  g.horizontal = dx * dy;
  g.volume = g.horizontal * thickness;
  return g;
#else
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
#endif
}

#undef PF_FACE_METRIC_HD
#endif
