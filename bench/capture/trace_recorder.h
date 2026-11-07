#pragma once

// Records the positions a solver hands to the neighbor search, once per
// search. scripts/capture_traces.sh patches calls to this into a copy of
// learnsph; bench/trace.hpp reads the result. Positions are stored as float.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct TraceFileHeader {
  char magic[8] = {'A', 'P', 'F', 'R', 'N', 'N', 'T', '1'};
  float radius = 0.0f;
  int32_t sets = 0;
  int32_t pair_count = 0;
  int32_t frames = 0;
  int32_t is_static[8] = {};
  int32_t pairs[16][2] = {};
};

class TraceRecorder {
public:
  void arm(const std::string &path, int frames) {
    file_ = std::fopen(path.c_str(), "wb");
    remaining_ = frames;
    written_ = 0;
    last_ = {};
    if (file_ != nullptr) {
      TraceFileHeader header;
      std::fwrite(&header, sizeof(header), 1, file_);
    }
  }

  bool armed() const { return file_ != nullptr; }

  void set_points(int set, const double *xyz, std::size_t n) {
    sources_[set] = {xyz, n};
  }

  void set_static(int set, bool value) { static_[set] = value; }

  void set_active(int query, int target, bool active) {
    if (active) {
      pairs_.insert({query, target});
    } else {
      pairs_.erase({query, target});
    }
  }

  void clear_active() { pairs_.clear(); }

  void frame(double radius) {
    if (file_ == nullptr) {
      return;
    }
    radius_ = static_cast<float>(radius);
    for (const auto &[set, source] : sources_) {
      write_set(set, source.first, source.second);
    }
    ++written_;
    if (--remaining_ == 0) {
      finish();
    }
  }

private:
  void write_set(int set, const double *xyz, std::size_t n) {
    std::vector<float> &points = scratch_;
    points.resize(n * 3);
    for (std::size_t i = 0; i < n * 3; ++i) {
      points[i] = static_cast<float>(xyz[i]);
    }
    std::vector<float> &last = last_[set];
    const int32_t count = static_cast<int32_t>(n);
    const int32_t stored = written_ == 0 || points != last ? 1 : 0;
    std::fwrite(&count, sizeof(count), 1, file_);
    std::fwrite(&stored, sizeof(stored), 1, file_);
    if (stored != 0) {
      std::fwrite(points.data(), sizeof(float), points.size(), file_);
      last = points;
    }
  }

  void finish() {
    TraceFileHeader header;
    header.radius = radius_;
    header.sets = static_cast<int32_t>(sources_.size());
    header.frames = written_;
    for (const auto &[set, value] : static_) {
      if (set >= 0 && set < 8) {
        header.is_static[set] = value ? 1 : 0;
      }
    }
    for (const auto &[query, target] : pairs_) {
      if (header.pair_count < 16) {
        header.pairs[header.pair_count][0] = query;
        header.pairs[header.pair_count][1] = target;
        ++header.pair_count;
      }
    }
    std::fseek(file_, 0, SEEK_SET);
    std::fwrite(&header, sizeof(header), 1, file_);
    std::fclose(file_);
    file_ = nullptr;
  }

  std::FILE *file_ = nullptr;
  int remaining_ = 0;
  int written_ = 0;
  float radius_ = 0.0f;
  std::map<int, std::pair<const double *, std::size_t>> sources_;
  std::map<int, bool> static_;
  std::set<std::pair<int, int>> pairs_;
  std::map<int, std::vector<float>> last_;
  std::vector<float> scratch_;
};

inline TraceRecorder &trace_recorder() {
  static TraceRecorder recorder;
  return recorder;
}
