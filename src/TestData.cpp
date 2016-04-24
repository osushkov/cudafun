
#include "TestData.hpp"
#include "common/Util.hpp"

static EVector randomVector(unsigned dim) {
  EVector result(dim);
  for (unsigned i = 0; i < dim; i++) {
    result(i) = Util::RandInterval(-10.0, 10.0);
  }
  return result;
}

static EMatrix randomMatrix(unsigned dim) {
  EMatrix result(dim, dim);
  for (unsigned i = 0; i < dim; i++) {
    for (unsigned j = 0; j < dim; j++) {
      result(i, j) = Util::RandInterval(-1.0, 1.0);
    }
  }
  return result;
}

pair<std::vector<EMatrix>, std::vector<EVector>>
TestData::GenerateData(unsigned dim, unsigned pipelineLength, unsigned samples) {

  pair<std::vector<EMatrix>, std::vector<EVector>> result;
  result.first.reserve(pipelineLength);
  result.second.reserve(samples);

  for (unsigned i = 0; i < pipelineLength; i++) {
    result.first.push_back(randomMatrix(dim));
  }

  for (unsigned i = 0; i < samples; i++) {
    result.second.push_back(randomVector(dim));
  }

  return result;
}

EVector TestData::Process(const std::vector<EMatrix> &pipeline, const EVector &sample) {
  EVector result = sample;
  for (const auto &pm : pipeline) {
    result = pm * result;
  }
  return result;
}
