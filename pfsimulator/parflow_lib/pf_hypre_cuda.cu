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
