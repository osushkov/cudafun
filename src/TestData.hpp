
#pragma once

#include "common/Common.hpp"
#include "common/Math.hpp"

namespace TestData {
pair<std::vector<EMatrix>, std::vector<EVector>> GenerateData(unsigned dim, unsigned pipelineLength,
                                                              unsigned samples);
EVector Process(const std::vector<EMatrix> &pipeline, const EVector &sample);
}
