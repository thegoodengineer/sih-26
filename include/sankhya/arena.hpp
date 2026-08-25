// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bump-pointer arena for node-local memory.
//
// Why this exists: branch-and-cut allocates and frees a swarm of short-lived arrays per
// node - the ratio-test candidate list, the cut-separation working set, the propagation
// queue. Those all die together when the node is popped. An arena turns thousands of
// malloc/free pairs per node into one pointer bump and one reset, which is both faster and
// far friendlier to the per-thread work-stealing pool in Phase 7, where allocator
// contention is a real cost.
//
// The arena hands out RAW, TRIVIALLY DESTRUCTIBLE storage only. It never runs destructors.
// That restriction is enforced at compile time by allocate<T>().
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace sankhya {

class Arena {
 public:
  /// `block_bytes` is the size of each underlying block. An allocation larger than a block
  /// gets a dedicated oversized block, so no request ever fails for being too big.
  explicit Arena(std::size_t block_bytes = 64 * 1024) : block_bytes_(block_bytes) {}

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = default;
  Arena& operator=(Arena&&) = default;

  /// Raw allocation with explicit alignment.
  [[nodiscard]] void* allocate_bytes(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return nullptr;
    std::size_t offset = align_up(used_, alignment);
    if (blocks_.empty() || offset + bytes > current_capacity_) {
      add_block(std::max(bytes + alignment, block_bytes_));
      offset = align_up(used_, alignment);
    }
    void* p = blocks_.back().get() + offset;
    used_ = offset + bytes;
    live_bytes_ += bytes;
    return p;
  }

  /// Uninitialised storage for `count` objects of type T.
  template <typename T>
  [[nodiscard]] T* allocate(std::size_t count) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Arena never runs destructors; use a container for non-trivial types");
    return static_cast<T*>(allocate_bytes(count * sizeof(T), alignof(T)));
  }

  /// Storage for `count` objects of type T, value-initialised to zero.
  template <typename T>
  [[nodiscard]] T* allocate_zeroed(std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T>, "allocate_zeroed writes raw zero bytes");
    T* p = allocate<T>(count);
    if (p != nullptr) std::memset(static_cast<void*>(p), 0, count * sizeof(T));
    return p;
  }

  /// Release everything back to the arena, KEEPING the largest block for reuse. Pointers
  /// handed out before the reset are dangling afterwards.
  void reset() noexcept {
    if (blocks_.size() > 1) {
      // Keep only the last (largest) block so a steady-state node solve stops allocating.
      auto keep = std::move(blocks_.back());
      const std::size_t keep_capacity = current_capacity_;
      blocks_.clear();
      blocks_.push_back(std::move(keep));
      current_capacity_ = keep_capacity;
      total_reserved_ = keep_capacity;
    }
    used_ = 0;
    live_bytes_ = 0;
  }

  /// Drop every block.
  void release() noexcept {
    blocks_.clear();
    used_ = 0;
    current_capacity_ = 0;
    live_bytes_ = 0;
    total_reserved_ = 0;
  }

  /// Bytes currently handed out (excluding alignment padding).
  [[nodiscard]] std::size_t live_bytes() const noexcept { return live_bytes_; }

  /// Total bytes held in blocks. The number to watch for a memory-growth regression.
  [[nodiscard]] std::size_t reserved_bytes() const noexcept { return total_reserved_; }

  [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }

 private:
  static std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  void add_block(std::size_t bytes) {
    blocks_.push_back(std::make_unique<std::byte[]>(bytes));
    current_capacity_ = bytes;
    total_reserved_ += bytes;
    used_ = 0;
  }

  std::size_t block_bytes_;
  std::vector<std::unique_ptr<std::byte[]>> blocks_;
  std::size_t used_ = 0;
  std::size_t current_capacity_ = 0;
  std::size_t live_bytes_ = 0;
  std::size_t total_reserved_ = 0;
};

}  // namespace sankhya
