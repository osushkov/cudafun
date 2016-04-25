
#pragma once

#include "../math/Matrix.hpp"
#include <vector>

namespace CudaKernel {
math::Matrix Multiply(const std::vector<math::Matrix> &matrixArray);
}
