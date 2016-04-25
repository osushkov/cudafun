
#include "MatrixView.hpp"
#include <algorithm>
#include <cassert>

using namespace math;

MatrixView MatrixView::CreateMatrixView(unsigned rows, unsigned cols) {
  assert(rows > 0 && cols > 0);

  MatrixView result;
  result.rows = rows;
  result.cols = cols;
  result.data = new float[rows * cols];

  return result;
}

MatrixView MatrixView::CreateMatrixViewZeroed(unsigned rows, unsigned cols) {
  assert(rows > 0 && cols > 0);

  MatrixView result = CreateMatrixView(rows, cols);
  std::fill(result.data, result.data + rows * cols, 0.0f);
  return result;
}

void MatrixView::ReleaseMatrixView(MatrixView &view) {
  assert(view.data != nullptr);
  delete[] view.data;
  view.data = nullptr;
}
