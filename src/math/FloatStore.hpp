#pragma once

class FloatStore {
  unsigned length;
  bool isPinned;
  float *data;

public:
  FloatStore(unsigned length);
  FloatStore(unsigned length, bool pinned);
  FloatStore(const FloatStore &other);
  FloatStore(FloatStore &&other);
  ~FloatStore();

  FloatStore &operator=(const FloatStore &other);

  float operator[](unsigned index) const;
  float &operator[](unsigned index);

  unsigned size(void) const;
};
