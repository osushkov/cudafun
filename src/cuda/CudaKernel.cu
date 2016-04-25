
#include "CudaKernel.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>

#include "../common/Timer.hpp"
#include "../math/MatrixView.hpp"

using namespace std;
using namespace math;

static MatrixView uploadToDevice(const MatrixView &mv);
static MatrixView downloadFromDevice(const MatrixView &mv);
static MatrixView allocHostView(unsigned rows, unsigned cols);
static void releaseHostView(MatrixView &view);

__global__ void multiplyKernel0(MatrixView a, MatrixView b, MatrixView out) {
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  if (outRow >= out.rows || outCol >= out.cols) {
    return;
  }

  float r = 0.0f;
  for (int i = 0; i < a.cols; i++) {
    r += a.data[i + outRow * a.cols] * b.data[outCol + i * b.cols];
  }
  out.data[outCol + outRow * out.cols] = r;
}

// Use shared memory.
__global__ void multiplyKernel1(MatrixView a, MatrixView b, MatrixView out) {
  extern __shared__ float buf[];

  const int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  const int outCol = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (a.cols + blockDim.x - 1) / blockDim.x;

  float *aChunk = (float *) buf;
  float *bChunk = (float *) &buf[blockDim.x * blockDim.y];

  float value = 0.0f;
  for (int i = 0; i < numChunks; i++) {
    const int chunkOffset = i * blockDim.x;
    const int chunkIndex = threadIdx.x + threadIdx.y * blockDim.x;

    const int aRow = outRow;
    const int aCol = chunkOffset + threadIdx.x;

    const int bRow = chunkOffset + threadIdx.y;
    const int bCol = outCol;

    if (aCol < a.cols) {
      aChunk[chunkIndex] = a.data[aCol + aRow * a.cols];
    }
    if (bRow < b.rows) {
      bChunk[chunkIndex] = b.data[bCol + bRow * b.cols];
    }

    __syncthreads();

    if (outRow < out.rows && outCol < out.cols) {
      int chunkLim = min(blockDim.x, a.cols - chunkOffset);
      for (int j = 0; j < chunkLim; j++) {
        value += aChunk[j + threadIdx.y * blockDim.x] * bChunk[threadIdx.x + j * blockDim.x];
      }
    }
    __syncthreads();
  }

  if (outRow < out.rows && outCol < out.cols) {
    out.data[outCol + outRow * out.cols] = value;
  }
}

math::Matrix CudaKernel::Multiply(const std::vector<math::Matrix> &matrixArray) {
  assert(!matrixArray.empty());
  if (matrixArray.size() == 1) {
    return matrixArray[0];
  }

  // Threads per block in X and Y dimensions.
  int tpbX = 16;
  int tpbY = 16;

  MatrixView d_cur = uploadToDevice(matrixArray[0].GetView());
  for (unsigned i = 1; i < matrixArray.size(); i++) {
    MatrixView d_next = uploadToDevice(matrixArray[i].GetView());
    MatrixView d_out = allocHostView(d_cur.rows, d_next.cols);

    // Blocks per grid in X and Y dimensions.
    int bpgX = (d_out.cols + tpbX - 1) / tpbX;
    int bpgY = (d_out.rows + tpbY - 1) / tpbY;

    size_t sharedMemSize = 2 * tpbX * tpbY * sizeof(float);

    multiplyKernel1<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1), sharedMemSize>>>(d_cur, d_next, d_out);

    releaseHostView(d_next);
    releaseHostView(d_cur);
    d_cur = d_out;
  }

  MatrixView h_result = downloadFromDevice(d_cur);
  releaseHostView(d_cur);

  math::Matrix result(h_result);
  MatrixView::ReleaseMatrixView(h_result);

  return result;

  // for (const auto& m : matrixArray) {
  //
  // }
  //
  // MatrixView h_A = viewFromMatrix(a);
  // MatrixView h_B = viewFromMatrix(b);
  // MatrixView h_R = newView(h_A.rows, h_B.cols);
  //
  // MatrixView d_A = uploadToDevice(h_A);
  // MatrixView d_B = uploadToDevice(h_B);
  //
  // MatrixView d_R = h_R;
  // cudaMalloc(&(d_R.data), h_R.rows * h_R.cols * sizeof(float));
  //
  // // Threads per block in X and Y dimensions.
  // int tpbX = 32;
  // int tpbY = 32;
  //
  // // Blocks per grid in X and Y dimensions.
  // int bpgX = (h_R.cols + tpbX - 1) / tpbX;
  // int bpgY = (h_R.rows + tpbY - 1) / tpbY;
  //
  //   size_t sharedMemSize = 2 * tpbX * tpbY * sizeof(float);
  //
  // Timer timer;
  // timer.Start();
  // for (unsigned i = 0; i < 10; i++) {
  //   multiplyKernel1<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1), sharedMemSize>>>(d_A, d_B, d_R);
  // }
  // cudaMemcpy(h_R.data, d_R.data, /*h_R.rows * h_R.cols * */sizeof(float), cudaMemcpyDeviceToHost);
  //
  // timer.Stop();
  // cout << "elapsed time: " << timer.GetNumElapsedSeconds() << endl;
  //
  // EMatrix result = matrixFromView(h_R);
  //
  // cudaFree(d_R.data);
  // cudaFree(d_B.data);
  // cudaFree(d_A.data);
  //
  // releaseView(h_R);
  // releaseView(h_B);
  // releaseView(h_A);
  //
  // return result;
}



MatrixView uploadToDevice(const MatrixView &h_mv) {
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  MatrixView d_mv = h_mv;
  cudaMalloc(&(d_mv.data), dataSize);
  cudaMemcpy(d_mv.data, h_mv.data, dataSize, cudaMemcpyHostToDevice);
  return d_mv;
}

MatrixView downloadFromDevice(const MatrixView &d_mv) {
  MatrixView h_mv = MatrixView::CreateMatrixView(d_mv.rows, d_mv.cols);
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  cudaMemcpy(h_mv.data, d_mv.data, dataSize, cudaMemcpyDeviceToHost);
  return h_mv;
}

MatrixView allocHostView(unsigned rows, unsigned cols) {
  assert(rows > 0 && cols > 0);

  MatrixView d_mv;
  d_mv.rows = rows;
  d_mv.cols = cols;
  cudaMalloc(&(d_mv.data), rows * cols * sizeof(float));
  return d_mv;
}

void releaseHostView(MatrixView &view) {
  cudaFree(view.data);
  view.data = nullptr;
}
