
#include "Matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

using namespace math;

Matrix::Matrix(unsigned rows, unsigned cols) : rows(rows), cols(cols), data(rows * cols) {
  assert(rows > 0 && cols > 0);
}

Matrix::Matrix(const MatrixView &mv) : rows(mv.rows), cols(mv.cols), data(rows * cols) {
  assert(rows > 0 && cols > 0);
  unsigned length = rows * cols;
  for (unsigned i = 0; i < length; i++) {
    data[i] = mv.data[i];
  }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols), data(other.data) {}
Matrix::Matrix(Matrix &&other) : rows(other.rows), cols(other.cols), data(std::move(other.data)) {}

Matrix &Matrix::operator=(const Matrix &other) {
  this->rows = other.rows;
  this->cols = other.cols;
  this->data = other.data;
  return *this;
}

float Matrix::operator()(unsigned r, unsigned c) const {
  assert(r < rows && c < cols);
  return data[c + r * cols];
}

float &Matrix::operator()(unsigned r, unsigned c) {
  assert(r < rows && c < cols);
  return data[c + r * cols];
}

float Matrix::operator()(unsigned r) const { return (*this)(r, 0); }

float &Matrix::operator()(unsigned r) { return (*this)(r, 0); }

Matrix Matrix::operator*(const Matrix &o) const {
  assert(cols == o.rows);

  Matrix result(rows, o.cols);

  for (unsigned r = 0; r < result.rows; r++) {
    for (unsigned c = 0; c < result.cols; c++) {
      float &v = result(r, c);
      v = 0.0f;

      for (unsigned i = 0; i < cols; i++) {
        v += (*this)(r, i) * o(i, c);
      }
    }
  }

  return result;
}

Matrix Matrix::operator+(const Matrix &o) const {
  assert(rows == o.rows && cols == o.cols);
  Matrix result(*this);
  result += o;
  return result;
}

Matrix Matrix::operator-(const Matrix &o) const {
  assert(rows == o.rows && cols == o.cols);
  Matrix result(*this);
  result -= o;
  return result;
}

Matrix Matrix::operator*(float s) const {
  Matrix result(*this);
  result *= s;
  return result;
}

Matrix Matrix::operator/(float s) const {
  Matrix result(*this);
  result /= s;
  return result;
}

Matrix &Matrix::operator+=(const Matrix &o) {
  assert(rows == o.rows && cols == o.cols);
  for (unsigned i = 0; i < data.size(); i++) {
    data[i] += o.data[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &o) {
  assert(rows == o.rows && cols == o.cols);
  for (unsigned i = 0; i < data.size(); i++) {
    data[i] -= o.data[i];
  }
  return *this;
}

Matrix &Matrix::operator*=(float s) {
  for (unsigned i = 0; i < data.size(); i++) {
    data[i] *= s;
  }
  return *this;
}

Matrix &Matrix::operator/=(float s) {
  for (unsigned i = 0; i < data.size(); i++) {
    data[i] /= s;
  }
  return *this;
}

unsigned Matrix::Rows(void) const { return rows; }

unsigned Matrix::Cols(void) const { return cols; }

MatrixView Matrix::GetView(void) const {
  MatrixView result = MatrixView::CreateMatrixView(rows, cols);
  result.data = &(const_cast<FloatStore *>(&data)->operator[](0));
  return result;
}

Matrix &Matrix::Fill(float val) {
  for (unsigned i = 0; i < data.size(); i++) {
    data[i] = val;
  }
  return *this;
}

Matrix &Matrix::SetIdentity(void) {
  assert(rows == cols);

  Fill(0.0f);
  for (unsigned i = 0; i < rows; i++) {
    (*this)(i, i) = 1.0f;
  }

  return *this;
}

bool Matrix::IsColumnVector(void) const { return cols == 1; }

float Matrix::InnerProduct(const Matrix &other) const {
  assert(rows == other.rows && cols == other.cols);

  float result = 0.0f;
  for (unsigned r = 0; r < rows; r++) {
    for (unsigned c = 0; c < cols; c++) {
      result += (*this)(r, c) * other(r, c);
    }
  }
  return result;
}

Matrix Matrix::OuterProduct(const Matrix &other) const {
  assert(IsColumnVector() && other.IsColumnVector());

  Matrix result(rows, other.rows);
  for (unsigned r = 0; r < result.rows; r++) {
    for (unsigned c = 0; c < result.cols; c++) {
      unsigned x = (*this)(r);
      unsigned y = other(c);
      result(r, c) = x * y;
    }
  }
  return result;
}

void Matrix::Apply(std::function<float(float)> transform) {
  for (unsigned r = 0; r < rows; r++) {
    for (unsigned c = 0; c < cols; c++) {
      (*this)(r, c) = transform((*this)(r, c));
    }
  }
}

std::ostream &operator<<(std::ostream &stream, const Matrix &m) {
  for (unsigned r = 0; r < m.Rows(); r++) {
    for (unsigned c = 0; c < m.Cols(); c++) {
      stream << m(r, c) << "\t";
    }
    stream << std::endl;
  }
  return stream;
}
