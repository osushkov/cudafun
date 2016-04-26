
#include "TestData.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "common/Util.hpp"
#include "cuda/CudaKernel.hpp"
#include "math/Matrix.hpp"
#include <future>
#include <thread>

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

vector<vector<math::Matrix>> generateData(unsigned batches, unsigned num, unsigned dim) {
  vector<vector<math::Matrix>> result;
  result.reserve(batches);

  for (unsigned i = 0; i < batches; i++) {
    vector<math::Matrix> batch;
    batch.reserve(num);
    for (unsigned j = 0; j < num; j++) {
      batch.push_back(randomMatrix(dim));
    }
    result.push_back(batch);
  }

  return result;
}

std::vector<math::Matrix> hostMultiplyMT(const std::vector<std::vector<math::Matrix>> &dataArray) {
  std::vector<std::future<math::Matrix>> tasks(dataArray.size());

  unsigned index = 0;
  for (const auto &matrixArray : dataArray) {
    tasks[index++] = std::async(std::launch::async,
                                [&matrixArray]() { return CudaKernel::MultiplyMT(matrixArray); });
    cout << "started " << index << endl;
  }

  std::vector<math::Matrix> result;
  result.reserve(dataArray.size());

  for (auto &task : tasks) {
    result.push_back(task.get());
  }

  return result;
}

int main(int argc, char **argv) {
  Timer timer;

  vector<vector<math::Matrix>> data = generateData(4, 100, 500);

  timer.Start();
  cout << "doing work: " << endl;
  auto deviceResult = hostMultiplyMT(data);
  // auto deviceResult = CudaKernel::Multiply(data);
  timer.Stop();
  cout << "result: " << timer.GetNumElapsedSeconds() << endl; // << deviceResult << endl;

  // timer.Start();
  // math::Matrix hostResult = processDataOnHost(data);
  // timer.Stop();
  // cout << "result: " << timer.GetNumElapsedSeconds() << endl; // << hostResult << endl;

  return 0;
}
