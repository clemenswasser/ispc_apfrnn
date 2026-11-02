#pragma once

// Minimal reader for the legacy VTK POINTS block written by learnsph.
// learnsph writes binary big-endian doubles; float files are also accepted.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace apfrnn {
namespace bench {

struct Cloud {
  std::vector<float> xyz;
  std::size_t size() const { return xyz.size() / 3; }
};

inline uint64_t byte_swap64(uint64_t value) {
  return ((value & 0x00000000000000ffULL) << 56) |
         ((value & 0x000000000000ff00ULL) << 40) |
         ((value & 0x0000000000ff0000ULL) << 24) |
         ((value & 0x00000000ff000000ULL) << 8) |
         ((value & 0x000000ff00000000ULL) >> 8) |
         ((value & 0x0000ff0000000000ULL) >> 24) |
         ((value & 0x00ff000000000000ULL) >> 40) |
         ((value & 0xff00000000000000ULL) >> 56);
}

inline uint32_t byte_swap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24) | ((value & 0x0000ff00U) << 8) |
         ((value & 0x00ff0000U) >> 8) | ((value & 0xff000000U) >> 24);
}

inline Cloud read_vtk_points(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }

  std::string line;
  long long count = 0;
  std::string type;
  while (std::getline(input, line)) {
    if (line.rfind("POINTS ", 0) == 0) {
      char type_buffer[16] = {};
      std::sscanf(line.c_str(), "POINTS %lld %15s", &count, type_buffer);
      type = type_buffer;
      break;
    }
  }

  if (count < 0 || (type != "double" && type != "float")) {
    std::fprintf(stderr, "unsupported VTK POINTS header in %s: %s\n",
                 path.c_str(), line.c_str());
    std::exit(1);
  }

  Cloud cloud;
  cloud.xyz.resize(static_cast<std::size_t>(count) * 3);
  if (type == "double") {
    std::vector<uint64_t> bits(static_cast<std::size_t>(count) * 3);
    input.read(reinterpret_cast<char *>(bits.data()),
               static_cast<std::streamsize>(bits.size() * sizeof(uint64_t)));
    if (!input) {
      std::fprintf(stderr, "short VTK POINTS block in %s\n", path.c_str());
      std::exit(1);
    }
    for (std::size_t i = 0; i < bits.size(); ++i) {
      const uint64_t value = byte_swap64(bits[i]);
      double coordinate = 0.0;
      std::memcpy(&coordinate, &value, sizeof(double));
      cloud.xyz[i] = static_cast<float>(coordinate);
    }
  } else {
    std::vector<uint32_t> bits(static_cast<std::size_t>(count) * 3);
    input.read(reinterpret_cast<char *>(bits.data()),
               static_cast<std::streamsize>(bits.size() * sizeof(uint32_t)));
    if (!input) {
      std::fprintf(stderr, "short VTK POINTS block in %s\n", path.c_str());
      std::exit(1);
    }
    for (std::size_t i = 0; i < bits.size(); ++i) {
      const uint32_t value = byte_swap32(bits[i]);
      float coordinate = 0.0f;
      std::memcpy(&coordinate, &value, sizeof(float));
      cloud.xyz[i] = coordinate;
    }
  }

  return cloud;
}

}  // namespace bench
}  // namespace apfrnn
