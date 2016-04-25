
#include "TestData.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "common/Util.hpp"
#include "cuda/CudaKernel.hpp"
#include "math/Matrix.hpp"

math::Matrix processDataOnHost(const vector<math::Matrix> &data) {
  assert(!data.empty());

  math::Matrix result = data[0];
  for (unsigned i = 1; i < data.size(); i++) {
    result = result * data[i];
  }
  return result;
}

math::Matrix randomMatrix(unsigned dim) {
  math::Matrix result(dim, dim);
  result.SetIdentity();

  double range = 1.0 / 33.0;
  for (unsigned i = 0; i < dim; i++) {
    for (unsigned j = 0; j < dim; j++) {
      result(i, j) = Util::RandInterval(-range, range);
    }
  }
  return result;
}

vector<math::Matrix> generateData(unsigned num, unsigned dim) {
  vector<math::Matrix> result;
  result.reserve(num);
  for (unsigned i = 0; i < num; i++) {
    result.push_back(randomMatrix(dim));
  }
  return result;
}

int main(int argc, char **argv) {
  Timer timer;

  vector<math::Matrix> data = generateData(100, 1000);

  timer.Start();
  math::Matrix deviceResult = CudaKernel::Multiply(data);
  timer.Stop();
  cout << "result: " << timer.GetNumElapsedSeconds() << endl; // << deviceResult << endl;

  timer.Start();
  math::Matrix hostResult = processDataOnHost(data);
  timer.Stop();
  cout << "result: " << timer.GetNumElapsedSeconds() << endl; // << hostResult << endl;

  math::Matrix diffM = deviceResult - hostResult;
  float diff = 0.0f;
  for (unsigned r = 0; r < diffM.Rows(); r++) {
    for (unsigned c = 0; c < diffM.Cols(); c++) {
      diff += diffM(r, c) * diffM(r, c);
    }
  }

  cout << "delta: " << sqrtf(diff) << endl;

  // Timer timer;
  // auto testData = TestData::GenerateData(1000, 10, 1000);
  //
  // // cout << "host processing elapsed time: " << timer.GetNumElapsedSeconds() << endl;
  //
  // EMatrix A = randomMatrix(10000);
  // EMatrix B = randomMatrix(10000);
  //
  // // EMatrix R = CudaKernel::Multiply0(A, B);
  //
  // timer.Start();
  // for (unsigned i = 0; i < 10; i++) {
  //   EMatrix R = A * B;
  // }
  // timer.Stop();
  // cout << "elapsed time: " << timer.GetNumElapsedSeconds() << endl;

  // EMatrix diff = A * B - R;
  // float delta = 0.0f;
  // float maxd = 0.0f;
  // float maxv = 0.0f;
  // for (unsigned r = 0; r < diff.rows(); r++) {
  //   for (unsigned c = 0; c < diff.cols(); c++) {
  //     if (fabs(diff(r, c)) > maxd) {
  //       maxd = fabs(diff(r, c));
  //       maxv = R(r, c);
  //     }
  //     delta += diff(r, c) * diff(r, c);
  //   }
  // }
  // delta = sqrtf(delta);
  // cout << "delta: " << delta << " " << maxd << ":" << maxv << endl;

  // cout << (A * B) << endl << endl;
  // cout << R << endl << endl;
  // cout << diff << endl;
  return 0;
}
