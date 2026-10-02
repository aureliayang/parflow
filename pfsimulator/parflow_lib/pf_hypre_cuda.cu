/* CUDA staging for the public HYPRE StructVector Box interface. */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

__global__ void PackBoxKernel(const double *source, double *values,
                              int source_index, int nx_source, int ny_source,
                              int nx, int ny, int nz)
{
  size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)nx * (size_t)ny * (size_t)nz;
  if (cell >= count)
  {
    return;
  }

  int i = (int)(cell % (size_t)nx);
  int j = (int)((cell / (size_t)nx) % (size_t)ny);
  int k = (int)(cell / ((size_t)nx * (size_t)ny));
  size_t source_offset = (size_t)source_index
                       + (size_t)i
                       + (size_t)nx_source * (size_t)j
                       + (size_t)nx_source * (size_t)ny_source * (size_t)k;
  values[cell] = source[source_offset];
}

__global__ void UnpackBoxKernel(const double *values, double *destination,
                                int destination_index, int nx_destination,
                                int ny_destination, int nx, int ny, int nz)
{
  size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)nx * (size_t)ny * (size_t)nz;
  if (cell >= count)
  {
    return;
  }

  int i = (int)(cell % (size_t)nx);
  int j = (int)((cell / (size_t)nx) % (size_t)ny);
  int k = (int)(cell / ((size_t)nx * (size_t)ny));
  size_t destination_offset = (size_t)destination_index
                            + (size_t)i
                            + (size_t)nx_destination * (size_t)j
                            + (size_t)nx_destination * (size_t)ny_destination * (size_t)k;
  destination[destination_offset] = values[cell];
}

static void CheckCuda(cudaError_t error, const char *where)
{
  if (error != cudaSuccess)
  {
    fprintf(stderr, "ParFlow HYPRE CUDA staging failed at %s: %s\n",
            where, cudaGetErrorString(error));
    exit(1);
  }
}

static int BlockCount(int nx, int ny, int nz)
{
  size_t count = (size_t)nx * (size_t)ny * (size_t)nz;
  return (int)((count + 255u) / 256u);
}

}  // namespace

extern "C" void HypreCudaPackBoxValues(const double *source, double *values,
                                        int source_index, int nx_source,
                                        int ny_source, int nx, int ny, int nz)
{
  if (nx <= 0 || ny <= 0 || nz <= 0)
  {
    return;
  }
  PackBoxKernel<<<BlockCount(nx, ny, nz), 256>>>(
      source, values, source_index, nx_source, ny_source, nx, ny, nz);
  CheckCuda(cudaGetLastError(), "PackBoxKernel launch");
  CheckCuda(cudaStreamSynchronize(0), "PackBoxKernel synchronize");
}

extern "C" void HypreCudaUnpackBoxValues(const double *values,
                                          double *destination,
                                          int destination_index,
                                          int nx_destination,
                                          int ny_destination,
                                          int nx, int ny, int nz)
{
  if (nx <= 0 || ny <= 0 || nz <= 0)
  {
    return;
  }
  UnpackBoxKernel<<<BlockCount(nx, ny, nz), 256>>>(
      values, destination, destination_index, nx_destination,
      ny_destination, nx, ny, nz);
  CheckCuda(cudaGetLastError(), "UnpackBoxKernel launch");
  CheckCuda(cudaStreamSynchronize(0), "UnpackBoxKernel synchronize");
}


__global__ void PackMatrixBoxKernel(
    const double *cp, const double *wp, const double *ep,
    const double *sop, const double *np, const double *lp,
    const double *up, const double *cp_c, const double *wp_c,
    const double *ep_c, const double *sop_c, const double *np_c,
    const double *top, double *values,
    int source_index, int nx_source, int ny_source,
    int c_index, int nx_c, int ny_c,
    int top_index, int nx_top,
    int nx, int ny, int nz, int stencil_size,
    int symmetric, int overland)
{
  size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)nx * (size_t)ny * (size_t)nz;
  if (cell >= count)
  {
    return;
  }

  int i = (int)(cell % (size_t)nx);
  int j = (int)((cell / (size_t)nx) % (size_t)ny);
  int k = (int)(cell / ((size_t)nx * (size_t)ny));
  size_t src = (size_t)source_index
             + (size_t)i
             + (size_t)nx_source * (size_t)j
             + (size_t)nx_source * (size_t)ny_source * (size_t)k;
  size_t out = cell * (size_t)stencil_size;

  if (!overland)
  {
    if (symmetric)
    {
      values[out + 0] = cp[src];
      values[out + 1] = ep[src];
      values[out + 2] = np[src];
      values[out + 3] = up[src];
    }
    else
    {
      values[out + 0] = cp[src];
      values[out + 1] = wp[src];
      values[out + 2] = ep[src];
      values[out + 3] = sop[src];
      values[out + 4] = np[src];
      values[out + 5] = lp[src];
      values[out + 6] = up[src];
    }
    return;
  }

  int top_k = (int)top[(size_t)top_index + (size_t)i
                       + (size_t)nx_top * (size_t)j];
  if (symmetric)
  {
    if (top_k == k)
    {
      size_t csrc = (size_t)c_index
                  + (size_t)i
                  + (size_t)nx_c * (size_t)j
                  + (size_t)nx_c * (size_t)ny_c * (size_t)k;
      values[out + 0] = cp_c[csrc];
      values[out + 1] = ep[src];
      values[out + 2] = np[src];
      values[out + 3] = up[src];
    }
    else
    {
      values[out + 0] = cp[src];
      values[out + 1] = ep[src];
      values[out + 2] = np[src];
      values[out + 3] = up[src];
    }
  }
  else
  {
    if (top_k == k)
    {
      size_t csrc = (size_t)c_index
                  + (size_t)i
                  + (size_t)nx_c * (size_t)j
                  + (size_t)nx_c * (size_t)ny_c * (size_t)k;
      size_t t = (size_t)top_index + (size_t)i
               + (size_t)nx_top * (size_t)j;
      values[out + 0] = cp_c[csrc];
      values[out + 1] = ((int)top[t - 1] == top_k) ? wp_c[csrc] : wp[src];
      values[out + 2] = ((int)top[t + 1] == top_k) ? ep_c[csrc] : ep[src];
      values[out + 3] = ((int)top[t - nx_top] == top_k) ? sop_c[csrc] : sop[src];
      values[out + 4] = ((int)top[t + nx_top] == top_k) ? np_c[csrc] : np[src];
      values[out + 5] = lp[src];
      values[out + 6] = up[src];
    }
    else
    {
      values[out + 0] = cp[src];
      values[out + 1] = wp[src];
      values[out + 2] = ep[src];
      values[out + 3] = sop[src];
      values[out + 4] = np[src];
      values[out + 5] = lp[src];
      values[out + 6] = up[src];
    }
  }
}

extern "C" void HypreCudaPackMatrixBoxValues(
    const double *cp, const double *wp, const double *ep,
    const double *sop, const double *np, const double *lp, const double *up,
    const double *cp_c, const double *wp_c, const double *ep_c,
    const double *sop_c, const double *np_c,
    const double *top, double *values,
    int source_index, int nx_source, int ny_source,
    int c_index, int nx_c, int ny_c,
    int top_index, int nx_top,
    int nx, int ny, int nz, int stencil_size,
    int symmetric, int overland)
{
  if (nx <= 0 || ny <= 0 || nz <= 0)
  {
    return;
  }
  PackMatrixBoxKernel<<<BlockCount(nx, ny, nz), 256>>>(
      cp, wp, ep, sop, np, lp, up, cp_c, wp_c, ep_c, sop_c, np_c, top,
      values, source_index, nx_source, ny_source, c_index, nx_c, ny_c,
      top_index, nx_top, nx, ny, nz, stencil_size, symmetric, overland);
  CheckCuda(cudaGetLastError(), "PackMatrixBoxKernel launch");
  CheckCuda(cudaStreamSynchronize(0), "PackMatrixBoxKernel synchronize");
}
