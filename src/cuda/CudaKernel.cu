
#include "CudaKernel.hpp"

// #define CUDA_API_PER_THREAD_DEFAULT_STREAM

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

// Threads per block in X and Y dimensions.
static constexpr int tpbX = 16;
static constexpr int tpbY = 16;

static MatrixView uploadToDevice(const MatrixView &mv, cudaStream_t &stream);
static MatrixView downloadFromDevice(const MatrixView &mv, cudaStream_t &stream);
static MatrixView uploadToDevice(const MatrixView &mv);
static MatrixView downloadFromDevice(const MatrixView &mv);
static MatrixView allocHostView(unsigned rows, unsigned cols);
static void releaseDeviceView(MatrixView &view);

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

void *CudaKernel::AllocPinned(size_t amount) {
  void* result;
  cudaHostAlloc(&result, amount, cudaHostAllocPortable);
  return result;
}

void CudaKernel::FreePinned(void *ptr) {
  cudaFreeHost(ptr);
}

std::vector<math::Matrix> CudaKernel::Multiply(
    const std::vector<std::vector<math::Matrix>> &dataArray) {

  std::vector<math::Matrix> result;
  result.reserve(dataArray.size());

  for (const auto& matrixArray : dataArray) {
    MatrixView d_cur = uploadToDevice(matrixArray[0].GetView());
    for (unsigned i = 1; i < matrixArray.size(); i++) {
      MatrixView d_next = uploadToDevice(matrixArray[i].GetView());
      MatrixView d_out = allocHostView(d_cur.rows, d_next.cols);

      // Blocks per grid in X and Y dimensions.
      int bpgX = (d_out.cols + tpbX - 1) / tpbX;
      int bpgY = (d_out.rows + tpbY - 1) / tpbY;

      size_t sharedMemSize = 2 * tpbX * tpbY * sizeof(float);

      multiplyKernel1<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1), sharedMemSize>>>(d_cur, d_next, d_out);

      releaseDeviceView(d_next);
      releaseDeviceView(d_cur);
      d_cur = d_out;
    }

    MatrixView h_result = downloadFromDevice(d_cur);
    releaseDeviceView(d_cur);

    result.push_back(math::Matrix(h_result));
    MatrixView::ReleaseMatrixView(h_result);
  }

  return result;
}

math::Matrix CudaKernel::MultiplyMT(const std::vector<math::Matrix> &matrixArray) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // MatrixView d_cur = uploadToDevice(matrixArray[0].GetView(), stream);
  // MatrixView d_next = uploadToDevice(matrixArray[i].GetView(), stream);
  // MatrixView d_tmp = allocHostView(d_cur.rows, d_next.cols);
  unsigned rows = matrixArray[0].Rows();
  unsigned cols = matrixArray[0].Cols();
  const unsigned dataSize = rows * cols * sizeof(float);

  MatrixView d_cur = allocHostView(rows, cols);
  MatrixView d_next = allocHostView(rows, cols);
  MatrixView d_out = allocHostView(rows, cols);

  cudaMemcpyAsync(d_cur.data, matrixArray[0].GetView().data, dataSize, cudaMemcpyHostToDevice, stream);

  for (unsigned i = 1; i < matrixArray.size(); i++) {
    cudaMemcpyAsync(d_next.data, matrixArray[i].GetView().data, dataSize, cudaMemcpyHostToDevice, stream);

    // Blocks per grid in X and Y dimensions.
    int bpgX = (d_out.cols + tpbX - 1) / tpbX;
    int bpgY = (d_out.rows + tpbY - 1) / tpbY;

    size_t sharedMemSize = 2 * tpbX * tpbY * sizeof(float);

    multiplyKernel1<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1), sharedMemSize, stream>>>(d_cur, d_next, d_out);

    float *tmp = d_cur.data;
    d_cur.data = d_out.data;
    d_out.data = tmp;
  }

  MatrixView h_result = downloadFromDevice(d_cur, stream);
  releaseDeviceView(d_cur);
  releaseDeviceView(d_next);
  releaseDeviceView(d_out);

  math::Matrix threadResult(h_result);
  MatrixView::ReleaseMatrixView(h_result);
  cudaStreamDestroy(stream);

  return threadResult;
}

MatrixView uploadToDevice(const MatrixView &h_mv, cudaStream_t &stream) {
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  MatrixView d_mv = h_mv;
  cudaMalloc(&(d_mv.data), dataSize);
  cudaMemcpyAsync(d_mv.data, h_mv.data, dataSize, cudaMemcpyHostToDevice, stream);
  return d_mv;
}

MatrixView downloadFromDevice(const MatrixView &d_mv, cudaStream_t &stream) {
  MatrixView h_mv = MatrixView::CreateMatrixView(d_mv.rows, d_mv.cols);
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  cudaMemcpyAsync(h_mv.data, d_mv.data, dataSize, cudaMemcpyDeviceToHost, stream);
  return h_mv;
}

MatrixView uploadToDevice(const MatrixView &h_mv) {
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  MatrixView d_mv = h_mv;
  cudaMalloc(&(d_mv.data), dataSize);
  cudaMemcpyAsync(d_mv.data, h_mv.data, dataSize, cudaMemcpyHostToDevice);
  return d_mv;
}

MatrixView downloadFromDevice(const MatrixView &d_mv) {
  MatrixView h_mv = MatrixView::CreateMatrixView(d_mv.rows, d_mv.cols);
  unsigned dataSize = h_mv.rows * h_mv.cols * sizeof(float);
  cudaMemcpyAsync(h_mv.data, d_mv.data, dataSize, cudaMemcpyDeviceToHost);
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

void releaseDeviceView(MatrixView &view) {
  cudaFree(view.data);
  view.data = nullptr;
}
