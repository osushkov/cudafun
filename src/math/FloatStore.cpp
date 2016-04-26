
#include "FloatStore.hpp"
#include "../cuda/CudaKernel.hpp"
#include <cassert>
#include <cstring>

FloatStore::FloatStore(unsigned length) : FloatStore(length, true) {}

FloatStore::FloatStore(unsigned length, bool pinned) : length(length), isPinned(pinned) {
  if (isPinned) {
    data = (float *)CudaKernel::AllocPinned(length * sizeof(float));
  } else {
    data = new float[length];
  }
}

FloatStore::FloatStore(const FloatStore &other) : length(other.length), isPinned(other.isPinned) {
  if (isPinned) {
    data = (float *)CudaKernel::AllocPinned(length * sizeof(float));
  } else {
    data = new float[length];
  }

  std::memcpy(data, other.data, length * sizeof(float));
}

FloatStore::FloatStore(FloatStore &&other)
    : length(other.length), isPinned(other.isPinned), data(other.data) {
  other.data = nullptr;
}

FloatStore::~FloatStore() {
  if (isPinned) {
    CudaKernel::FreePinned(data);
  } else {
    delete[] data;
  }
}

FloatStore &FloatStore::operator=(const FloatStore &other) {
  if (data != nullptr) {
    if (isPinned) {
      CudaKernel::FreePinned(data);
    } else {
      delete[] data;
    }
  }

  length = other.length;
  isPinned = other.isPinned;

  if (isPinned) {
    data = (float *)CudaKernel::AllocPinned(length * sizeof(float));
  } else {
    data = new float[length];
  }

  std::memcpy(data, other.data, length * sizeof(float));
  return *this;
}

float FloatStore::operator[](unsigned index) const {
  assert(index < length);
  return data[index];
}

float &FloatStore::operator[](unsigned index) {
  assert(index < length);
  return data[index];
}

unsigned FloatStore::size(void) const { return length; }
