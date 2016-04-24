
#include "TestData.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "common/Util.hpp"
#include "cuda/CudaKernel.hpp"

vector<EVector> processDataOnHost(const pair<vector<EMatrix>, vector<EVector>> &data) {
  vector<EVector> result;
  result.reserve(data.second.size());

  for (const auto &dv : data.second) {
    result.push_back(TestData::Process(data.first, dv));
  }

  return result;
}

EMatrix randomMatrix(unsigned dim) {
  EMatrix result(dim, dim);
  int n = 0;
  for (unsigned i = 0; i < dim; i++) {
    for (unsigned j = 0; j < dim; j++) {
      result(i, j) = Util::RandInterval(-2.0, 2.0);
    }
  }
  return result;
}

int main(int argc, char **argv) {
  Timer timer;
  auto testData = TestData::GenerateData(1000, 10, 1000);

  timer.Start();
  // vector<EVector> hostResult = processDataOnHost(testData);
  timer.Stop();

  // cout << "host processing elapsed time: " << timer.GetNumElapsedSeconds() << endl;

  EMatrix A = randomMatrix(1000);
  EMatrix B = randomMatrix(1000);

  EMatrix R = CudaKernel::Multiply0(A, B);

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
