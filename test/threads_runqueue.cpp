// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS
#include <cstdlib>
#include <memory>
#include "main.h"
#include <Eigen/ThreadPool>

// Visual studio doesn't implement a rand_r() function since its
// implementation of rand() is already thread safe
int rand_reentrant(unsigned int* s) {
#if EIGEN_COMP_MSVC_STRICT
  EIGEN_UNUSED_VARIABLE(s);
  return rand();
#else
  return rand_r(s);
#endif
}

void test_basic_runqueue() {
  RunQueue<int, 4> q;
  // Check empty state.
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PopFront());
  std::vector<int> stolen;
  VERIFY_IS_EQUAL(0u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(0u, stolen.size());
  // Push one front, pop one front.
  VERIFY_IS_EQUAL(0, q.PushFront(1));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(1, q.PopFront());
  VERIFY_IS_EQUAL(0u, q.Size());
  // Push front to overflow.
  VERIFY_IS_EQUAL(0, q.PushFront(2));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(3));
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(4));
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(5));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(6, q.PushFront(6));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(5, q.PopFront());
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(4, q.PopFront());
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(3, q.PopFront());
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(2, q.PopFront());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PopFront());
  // Push one back, pop one back.
  VERIFY_IS_EQUAL(0, q.PushBack(7));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(7, stolen[0]);
  VERIFY_IS_EQUAL(0u, q.Size());
  stolen.clear();
  // Push back to overflow.
  VERIFY_IS_EQUAL(0, q.PushBack(8));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(9));
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(10));
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(11));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(12, q.PushBack(12));
  VERIFY_IS_EQUAL(4u, q.Size());
  // Pop back in halves.
  VERIFY_IS_EQUAL(2u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(2u, stolen.size());
  VERIFY_IS_EQUAL(10, stolen[0]);
  VERIFY_IS_EQUAL(11, stolen[1]);
  VERIFY_IS_EQUAL(2u, q.Size());
  stolen.clear();
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(9, stolen[0]);
  VERIFY_IS_EQUAL(1u, q.Size());
  stolen.clear();
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(8, stolen[0]);
  stolen.clear();
  VERIFY_IS_EQUAL(0u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(0u, stolen.size());
  // Empty again.
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(1));
  VERIFY_IS_EQUAL(0, q.PushFront(2));
  VERIFY_IS_EQUAL(0, q.PushFront(3));
  VERIFY_IS_EQUAL(1, q.PopBack());
  VERIFY_IS_EQUAL(2, q.PopBack());
  VERIFY_IS_EQUAL(3, q.PopBack());
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
}

// Counts move constructions, to detect when a std::vector holding it reallocates. Move assignments are not counted:
// the queue resets each slot it pops with one.
struct CountedWork {
  static int moves;
  int value;
  CountedWork(int v = 0) : value(v) {}
  CountedWork(CountedWork&& other) noexcept : value(other.value) { moves++; }
  CountedWork& operator=(CountedWork&& other) noexcept {
    value = other.value;
    return *this;
  }
};
int CountedWork::moves = 0;

// PopBackHalf stores back_ only after taking its last element, so a std::bad_alloc from the result vector in between
// would leave a hole in the queue. The vector must grow before the first element is taken: each element taken is
// move-constructed exactly once, and the elements already in the vector only if it grew.
void test_pop_back_half_grows_result_first() {
  RunQueue<CountedWork, 16> q;
  std::vector<CountedWork> stolen;
  for (int round = 0; round < 3; round++) {
    for (int i = 1; i <= 16; i++) VERIFY_IS_EQUAL(0, q.PushFront(CountedWork(i)).value);
    unsigned expected = 8;
    while (!q.Empty()) {
      const std::size_t size = stolen.size();
      const std::size_t capacity = stolen.capacity();
      CountedWork::moves = 0;
      VERIFY_IS_EQUAL(expected, q.PopBackHalf(&stolen));
      VERIFY_IS_EQUAL(size + expected, stolen.size());
      VERIFY_IS_EQUAL(static_cast<std::size_t>(CountedWork::moves),
                      expected + (stolen.capacity() == capacity ? 0 : size));
      expected = (expected + 1) / 2;
    }
    VERIFY_IS_EQUAL(0u, q.Size());
  }
  int count[17] = {0};
  for (const CountedWork& w : stolen) count[w.value]++;
  for (int i = 1; i <= 16; i++) VERIFY_IS_EQUAL(3, count[i]);
}

// Empty tests that the queue is not claimed to be empty when is is in fact not.
// Emptiness property is crucial part of thread pool blocking scheme,
// so we go to great effort to ensure this property. We create a queue with
// 1 element and then push 1 element (either front or back at random) and pop
// 1 element (either front or back at random). So queue always contains at least
// 1 element, but otherwise changes chaotically. Another thread constantly tests
// that the queue is not claimed to be empty.
void test_empty_runqueue() {
  RunQueue<int, 4> q;
  q.PushFront(1);
  std::atomic<bool> done(false);
  std::thread mutator([&q, &done]() {
    unsigned rnd = 0;
    std::vector<int> stolen;
    for (int i = 0; i < 1 << 18; i++) {
      if (rand_reentrant(&rnd) % 2)
        VERIFY_IS_EQUAL(0, q.PushFront(1));
      else
        VERIFY_IS_EQUAL(0, q.PushBack(1));
      if (rand_reentrant(&rnd) % 2)
        VERIFY_IS_EQUAL(1, q.PopFront());
      else {
        for (;;) {
          if (q.PopBackHalf(&stolen) == 1) {
            stolen.clear();
            break;
          }
          VERIFY_IS_EQUAL(0u, stolen.size());
        }
      }
    }
    done = true;
  });
  while (!done) {
    VERIFY(!q.Empty());
    int size = q.Size();
    VERIFY_GE(size, 1);
    VERIFY_LE(size, 2);
  }
  VERIFY_IS_EQUAL(1, q.PopFront());
  mutator.join();
}

// Stress is a chaotic random test.
// One thread (owner) calls PushFront/PopFront, other threads call PushBack/
// PopBack. Ensure that we don't crash, deadlock, and all sanity checks pass.
void test_stress_runqueue() {
  static const int kEvents = 1 << 18;
  RunQueue<int, 8> q;
  std::atomic<int64_t> total(0);
  std::vector<std::unique_ptr<std::thread>> threads;
  threads.emplace_back(new std::thread([&q, &total]() {
    int64_t sum = 0;
    int pushed = 1;
    int popped = 1;
    while (pushed < kEvents || popped < kEvents) {
      if (pushed < kEvents) {
        if (q.PushFront(pushed) == 0) {
          sum += pushed;
          pushed++;
        }
      }
      if (popped < kEvents) {
        int v = q.PopFront();
        if (v != 0) {
          sum -= v;
          popped++;
        }
      }
    }
    total += sum;
  }));
  for (int i = 0; i < 2; i++) {
    threads.emplace_back(new std::thread([&q, &total]() {
      int64_t sum = 0;
      for (int j = 1; j < kEvents; j++) {
        if (q.PushBack(j) == 0) {
          sum += j;
          continue;
        }
        EIGEN_THREAD_YIELD();
        j--;
      }
      total += sum;
    }));
    threads.emplace_back(new std::thread([&q, &total]() {
      int64_t sum = 0;
      std::vector<int> stolen;
      for (int j = 1; j < kEvents;) {
        if (q.PopBackHalf(&stolen) == 0) {
          EIGEN_THREAD_YIELD();
          continue;
        }
        while (!stolen.empty() && j < kEvents) {
          int v = stolen.back();
          stolen.pop_back();
          VERIFY_IS_NOT_EQUAL(v, 0);
          sum += v;
          j++;
        }
      }
      while (!stolen.empty()) {
        int v = stolen.back();
        stolen.pop_back();
        VERIFY_IS_NOT_EQUAL(v, 0);
        while ((v = q.PushBack(v)) != 0) EIGEN_THREAD_YIELD();
      }
      total -= sum;
    }));
  }
  for (size_t i = 0; i < threads.size(); i++) threads[i]->join();
  VERIFY(q.Empty());
  VERIFY(total.load() == 0);
}

// A Work whose move leaves the source owning a copy, as libc++'s std::function does for a small target.
struct CopyOnMoveWork {
  CopyOnMoveWork() = default;
  explicit CopyOnMoveWork(const std::shared_ptr<int>& o) : owner(o) {}
  CopyOnMoveWork(CopyOnMoveWork&& other) : owner(other.owner) {}
  CopyOnMoveWork& operator=(CopyOnMoveWork&& other) {
    owner = other.owner;
    return *this;
  }
  std::shared_ptr<int> owner;
};

// A popped element must not stay alive in its slot of the queue: what it owns would be released only when the slot
// is reused, and a thread pool task can wait for that release.
void test_pop_releases_element() {
  RunQueue<CopyOnMoveWork, 4> q;
  const std::shared_ptr<int> owner = std::make_shared<int>(0);

  q.PushFront(CopyOnMoveWork(owner));
  VERIFY(q.PopFront().owner == owner);
  VERIFY(owner.use_count() == 1);

  q.PushBack(CopyOnMoveWork(owner));
  VERIFY(q.PopBack().owner == owner);
  VERIFY(owner.use_count() == 1);

  for (int i = 0; i < 4; ++i) q.PushBack(CopyOnMoveWork(owner));
  std::vector<CopyOnMoveWork> stolen;
  VERIFY_IS_EQUAL(2u, q.PopBackHalf(&stolen));
  stolen.clear();
  VERIFY(owner.use_count() == 3);
  q.Flush();
  VERIFY(owner.use_count() == 1);
}

// A Work that, like CopyOnMoveWork, stays alive in the slot it is moved from, and whose destructor pushes a new element
// onto the front of its queue, as the destructor of a closure that schedules a task from its worker thread does.
struct PushOnDestroyWork;
using PushOnDestroyQueue = RunQueue<PushOnDestroyWork, 4>;
struct PushOnDestroyState {
  PushOnDestroyQueue* queue = nullptr;
  int pushes_left = 0;
  int accepted = 0;
};
struct PushOnDestroyWork {
  PushOnDestroyWork() = default;
  explicit PushOnDestroyWork(PushOnDestroyState* s) : state(s) {}
  PushOnDestroyWork(PushOnDestroyWork&& other) : state(other.state), marker(other.marker) {}
  PushOnDestroyWork& operator=(PushOnDestroyWork&& other) {
    release();
    state = other.state;
    marker = other.marker;
    return *this;
  }
  ~PushOnDestroyWork() { release(); }
  void release() {
    PushOnDestroyState* s = state;
    state = nullptr;
    if (s == nullptr || s->pushes_left == 0) return;
    --s->pushes_left;
    PushOnDestroyWork pushed;
    pushed.marker = true;
    // A queue that does not accept the element returns it.
    const PushOnDestroyWork rejected = s->queue->PushFront(std::move(pushed));
    if (!rejected.marker) ++s->accepted;
  }
  PushOnDestroyState* state = nullptr;
  bool marker = false;
};

// PopFront() destroys what the move left in the slot. That destructor can push onto the front of the same queue,
// which must stay consistent: it holds the pushed element exactly if it accepted it.
void test_pop_front_with_pushing_destructor() {
  PushOnDestroyQueue q;
  PushOnDestroyState state;
  state.queue = &q;
  q.PushFront(PushOnDestroyWork(&state));
  VERIFY_IS_EQUAL(1u, q.Size());
  // Only the copy left in the slot pushes, once.
  state.pushes_left = 1;
  PushOnDestroyWork popped = q.PopFront();
  VERIFY_IS_EQUAL(0, state.pushes_left);
  VERIFY(popped.state == &state);
  VERIFY_IS_EQUAL(static_cast<unsigned>(state.accepted), q.Size());
  for (int i = 0; i < state.accepted; ++i) VERIFY(q.PopFront().marker);
  VERIFY(q.Empty());
  // Every slot is free again.
  for (int i = 0; i < 4; ++i) q.PushFront(PushOnDestroyWork());
  VERIFY_IS_EQUAL(4u, q.Size());
  q.Flush();
}

EIGEN_DECLARE_TEST(threads_runqueue) {
  CALL_SUBTEST_1(test_basic_runqueue());
  CALL_SUBTEST_1(test_pop_back_half_grows_result_first());
  CALL_SUBTEST_1(test_pop_releases_element());
  CALL_SUBTEST_1(test_pop_front_with_pushing_destructor());
  CALL_SUBTEST_2(test_empty_runqueue());
  CALL_SUBTEST_3(test_stress_runqueue());
}
