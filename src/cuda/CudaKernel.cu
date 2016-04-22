
#include "CudaKernel.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void cudaKernel(float *arrayIn, float *arrayOut, int numElems) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) {
    return;
  }

  float d = arrayIn[index];
  arrayOut[index] = d*d;
}

void CudaKernel::RunKernel(void) {
  const unsigned ARRAY_SIZE = 128;

  float* h_in = new float[ARRAY_SIZE];
  float* h_out = new float[ARRAY_SIZE];

  for (unsigned i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = static_cast<float>(i);
  }

  float* d_in;
  float* d_out;

  cudaMalloc(&d_in, ARRAY_SIZE * sizeof(float));
  cudaMalloc(&d_out, ARRAY_SIZE * sizeof(float));

  int threadsPerBlock = 256;
  int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;

  cudaMemcpy(d_in, h_in, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, ARRAY_SIZE);
  cudaMemcpy(h_out, d_out, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < ARRAY_SIZE; i++) {
    cout << i << " : " << h_out[i] << endl;
  }

  cudaFree(d_out);
  cudaFree(d_in);

  delete[] h_out;
  delete[] h_in;
}
