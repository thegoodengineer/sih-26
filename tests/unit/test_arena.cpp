// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Arena tests.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/arena.hpp"

namespace sankhya {
namespace {

TEST(Arena, ZeroSizedAllocationReturnsNull) {
  Arena arena;
  EXPECT_EQ(arena.allocate<double>(0), nullptr);
  EXPECT_EQ(arena.live_bytes(), 0u);
}

TEST(Arena, AllocationsAreDistinctAndWritable) {
  Arena arena(1024);
  double* a = arena.allocate<double>(10);
  double* b = arena.allocate<double>(10);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);

  for (int i = 0; i < 10; ++i) {
    a[i] = static_cast<double>(i);
    b[i] = static_cast<double>(-i);
  }
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(a[i], static_cast<double>(i));
    EXPECT_DOUBLE_EQ(b[i], static_cast<double>(-i));
  }
}

TEST(Arena, RespectsAlignment) {
  Arena arena(4096);
  // Interleave types with different alignments; a bump allocator that forgets to round up
  // produces a misaligned pointer here, which is UB the sanitizer build will catch.
  (void)arena.allocate<char>(1);
  auto* p64 = arena.allocate<std::int64_t>(4);
  (void)arena.allocate<char>(3);
  auto* pd = arena.allocate<double>(4);

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p64) % alignof(std::int64_t), 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(pd) % alignof(double), 0u);
}

TEST(Arena, GrowsBeyondOneBlock) {
  Arena arena(256);
  std::vector<double*> pointers;
  for (int i = 0; i < 64; ++i) {
    double* p = arena.allocate<double>(16);  // 128 bytes each
    ASSERT_NE(p, nullptr);
    for (int k = 0; k < 16; ++k) p[k] = static_cast<double>(i);
    pointers.push_back(p);
  }
  EXPECT_GT(arena.block_count(), 1u);
  // Earlier allocations must survive later block additions.
  for (int i = 0; i < 64; ++i) {
    for (int k = 0; k < 16; ++k) {
      ASSERT_DOUBLE_EQ(pointers[static_cast<std::size_t>(i)][k], static_cast<double>(i));
    }
  }
}

TEST(Arena, HandlesAllocationLargerThanTheBlockSize) {
  Arena arena(64);
  double* p = arena.allocate<double>(1000);  // 8000 bytes into 64-byte blocks
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 1000; ++i) p[i] = static_cast<double>(i);
  EXPECT_DOUBLE_EQ(p[999], 999.0);
}

TEST(Arena, AllocateZeroedIsZero) {
  Arena arena(1024);
  // Dirty the arena first so that a missing memset shows up as garbage, not as luck.
  double* dirty = arena.allocate<double>(32);
  for (int i = 0; i < 32; ++i) dirty[i] = 12345.0;
  arena.reset();

  double* p = arena.allocate_zeroed<double>(32);
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 32; ++i) EXPECT_DOUBLE_EQ(p[i], 0.0);
}

TEST(Arena, ResetKeepsOneBlockAndClearsLiveBytes) {
  Arena arena(256);
  for (int i = 0; i < 32; ++i) (void)arena.allocate<double>(16);
  EXPECT_GT(arena.block_count(), 1u);
  EXPECT_GT(arena.live_bytes(), 0u);

  arena.reset();
  EXPECT_EQ(arena.block_count(), 1u);
  EXPECT_EQ(arena.live_bytes(), 0u);

  // A reset arena must still be usable, and reusing the kept block must not allocate more.
  const std::size_t reserved_after_reset = arena.reserved_bytes();
  double* p = arena.allocate<double>(4);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(arena.reserved_bytes(), reserved_after_reset);
}

TEST(Arena, ReleaseDropsEverything) {
  Arena arena(256);
  (void)arena.allocate<double>(100);
  arena.release();
  EXPECT_EQ(arena.block_count(), 0u);
  EXPECT_EQ(arena.reserved_bytes(), 0u);
  EXPECT_EQ(arena.live_bytes(), 0u);

  double* p = arena.allocate<double>(2);
  ASSERT_NE(p, nullptr);
  p[0] = 1.0;
  EXPECT_DOUBLE_EQ(p[0], 1.0);
}

TEST(Arena, RepeatedResetDoesNotGrowMemory) {
  // The branch-and-cut steady state: allocate a node's working set, pop the node, repeat.
  // Reserved bytes must plateau instead of climbing, or Phase 7 leaks a block per node.
  Arena arena(4096);
  for (int i = 0; i < 8; ++i) {
    (void)arena.allocate<double>(400);
    arena.reset();
  }
  const std::size_t plateau = arena.reserved_bytes();
  for (int i = 0; i < 200; ++i) {
    (void)arena.allocate<double>(400);
    arena.reset();
  }
  EXPECT_EQ(arena.reserved_bytes(), plateau);
}

}  // namespace
}  // namespace sankhya
