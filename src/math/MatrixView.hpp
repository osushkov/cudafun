
#pragma once

namespace math {
struct MatrixView {
  unsigned rows;
  unsigned cols;
  float *data;

  static MatrixView CreateMatrixView(unsigned rows, unsigned cols);
  static MatrixView CreateMatrixViewZeroed(unsigned rows, unsigned cols);
  static void ReleaseMatrixView(MatrixView &view);
};
}
