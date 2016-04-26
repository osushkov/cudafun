
#pragma once

#include "../math/Matrix.hpp"
#include <vector>

namespace CudaKernel {
void *AllocPinned(size_t amount);
void FreePinned(void *ptr);

std::vector<math::Matrix> Multiply(const std::vector<std::vector<math::Matrix>> &matrixArray);
math::Matrix MultiplyMT(const std::vector<math::Matrix> &matrixArray);
}
