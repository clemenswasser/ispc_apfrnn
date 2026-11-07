#pragma once

// Replays neighbor-search traces captured from a simulation by
// scripts/capture_traces.sh. A trace holds the positions of every search in a
// run of consecutive searches, including particles entering and leaving.
// Playback runs forward and then backward, so motion never jumps.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace apfrnn::bench {

class Trace {
public:
  struct Header {
    char magic[8];
    float radius;
    int32_t sets;
    int32_t pair_count;
    int32_t frames;
    int32_t is_static[8];
    int32_t pairs[16][2];
  };

  struct Points {
    const float *xyz = nullptr;
    int count = 0;
  };

  explicit Trace(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      return;
    }
    struct stat info {};
    if (fstat(fd, &info) == 0 &&
        static_cast<std::size_t>(info.st_size) >= sizeof(Header)) {
      size_ = static_cast<std::size_t>(info.st_size);
      void *map = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
      if (map != MAP_FAILED) {
        data_ = static_cast<const char *>(map);
      }
    }
    close(fd);
    if (data_ == nullptr) {
      return;
    }
    std::memcpy(&header_, data_, sizeof(Header));
    if (std::memcmp(header_.magic, "APFRNNT1", 8) != 0 || header_.frames < 2) {
      return;
    }
    index();
    valid_ = true;
  }

  ~Trace() {
    if (data_ != nullptr) {
      munmap(const_cast<char *>(data_), size_);
    }
  }

  Trace(const Trace &) = delete;
  Trace &operator=(const Trace &) = delete;

  bool valid() const { return valid_; }
  const Header &header() const { return header_; }
  int frames() const { return header_.frames; }
  // Searches in one forward and backward pass.
  int cycle() const { return 2 * header_.frames - 2; }

  // Positions of a set at a step of the forward and backward playback.
  Points points(int set, int step) const {
    return points_[static_cast<std::size_t>(frame_of(step))]
                  [static_cast<std::size_t>(set)];
  }

  int frame_of(int step) const {
    const int position = step % cycle();
    return position < header_.frames ? position : cycle() - position;
  }

  bool resizes() const {
    for (const std::vector<Points> &frame : points_) {
      for (std::size_t set = 0; set < frame.size(); ++set) {
        if (frame[set].count != points_[0][set].count) {
          return true;
        }
      }
    }
    return false;
  }

  // The largest point count of the first set over the trace.
  int max_points() const {
    int largest = 0;
    for (const std::vector<Points> &frame : points_) {
      largest = std::max(largest, frame[0].count);
    }
    return largest;
  }

private:
  void index() {
    const char *cursor = data_ + sizeof(Header);
    points_.resize(static_cast<std::size_t>(header_.frames));
    std::vector<Points> current(static_cast<std::size_t>(header_.sets));
    for (std::vector<Points> &frame : points_) {
      for (Points &points : current) {
        int32_t count = 0;
        int32_t stored = 0;
        std::memcpy(&count, cursor, sizeof(count));
        std::memcpy(&stored, cursor + sizeof(count), sizeof(stored));
        cursor += sizeof(count) + sizeof(stored);
        points.count = count;
        if (stored != 0) {
          points.xyz = reinterpret_cast<const float *>(cursor);
          cursor += static_cast<std::size_t>(count) * 3 * sizeof(float);
        }
      }
      frame = current;
    }
  }

  const char *data_ = nullptr;
  std::size_t size_ = 0;
  Header header_{};
  bool valid_ = false;
  std::vector<std::vector<Points>> points_;
};

inline std::vector<std::filesystem::path>
trace_files(const std::string &directory) {
  std::vector<std::filesystem::path> files;
  std::error_code error;
  for (const auto &entry :
       std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() == ".trace") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

} // namespace apfrnn::bench
