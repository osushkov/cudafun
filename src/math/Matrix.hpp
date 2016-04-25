
#pragma once

#include "MatrixView.hpp"
#include <functional>
#include <iostream>
#include <vector>

namespace math {

// A relatively simple matrix class optimised for interfacing with CUDA and
// built for neural network tasks. Not as fast on the CPU as say the Eigen library
// for multiplication (Eigen uses SIMD instructions I'm pretty sure), but simple and
// a knowable memory layout for the data.
class Matrix {
  unsigned rows;
  unsigned cols;
  std::vector<float> data; // row major storage of data.

public:
  Matrix(unsigned rows, unsigned cols);
  Matrix(const MatrixView &mv);

  Matrix(const Matrix &other);
  Matrix(Matrix &&other);
  Matrix &operator=(const Matrix &other);

  float operator()(unsigned r, unsigned c) const;
  float &operator()(unsigned r, unsigned c);

  // Useful for column vectors, assumes c = 0.
  float operator()(unsigned r) const;
  float &operator()(unsigned r);

  Matrix operator*(const Matrix &o) const;
  Matrix operator+(const Matrix &o) const;
  Matrix operator-(const Matrix &o) const;
  Matrix operator*(float s) const;
  Matrix operator/(float s) const;

  // Matrix &operator*=(const Matrix &o);
  Matrix &operator+=(const Matrix &o);
  Matrix &operator-=(const Matrix &o);
  Matrix &operator*=(float s);
  Matrix &operator/=(float s);

  unsigned Rows(void) const;
  unsigned Cols(void) const;

  // This is useful for getting a simple datastructure view of the matrix data that doesnt
  // rely on the implementation of the base Matrix class. Useful for passing to CUDA kernels
  // that may not be happy taking a pointer to a fancy C++ class.
  // WARNING: the lifetime of the view returned from this method should be strictly less than the
  // lifetime of this object. Additionally, do not free/release the memory of the view.
  MatrixView GetView(void) const;

  Matrix &Fill(float val);
  Matrix &SetIdentity(void);

  bool IsColumnVector(void) const;

  // InnerProduct assumes both matrices are of the same dimensionality.
  float InnerProduct(const Matrix &other) const;

  // OuterProduct assumes both matrices are column vectors, but not necessarily the same size.
  Matrix OuterProduct(const Matrix &other) const;

  void Apply(std::function<float(float)> transform);
};
}
std::ostream &operator<<(std::ostream &stream, const math::Matrix &m);
