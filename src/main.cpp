
#include "common/Common.hpp"
#include "cuda/CudaKernel.hpp"

int main(int argc, char **argv) {
  CudaKernel::RunKernel();
  return 0;
}
