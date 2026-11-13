// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_ACCUMULATOR_H
#define EIGEN_SPARSE_ACCUMULATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Maps [lo, hi] onto [0, m) monotonically:
//
//   h(i) = floor((i - lo) s / 2^k),   s = floor(m 2^k / span),   span = hi - lo + 1,
//
// k the width of StorageIndex. For m < span, s < 2^k and h(i) <= (i - lo) m / span < m. For m >= span, h(i) = i - lo
// (direct), and the range is [0, span).
template <typename StorageIndex>
class monotone_hash {
  using Unsigned = std::make_unsigned_t<StorageIndex>;

 public:
  monotone_hash() = default;
  monotone_hash(StorageIndex lo, StorageIndex hi, Index m) { compute(lo, hi, m); }

  void compute(StorageIndex lo, StorageIndex hi, Index m) {
    Index span = Index(hi) - Index(lo) + 1;
    m_lo = lo;
    m_direct = m >= span;
    m_range = m_direct ? span : m;
    if (!m_direct) m_scale = int_div::div_shifted(Unsigned(m), Unsigned(span));
  }

  Index operator()(StorageIndex i) const {
    Unsigned x = Unsigned(i - m_lo);
    return m_direct ? Index(x) : Index(int_div::umulhi(x, m_scale));
  }

  // h maps into [0, range()); direct() means h is one-to-one.
  Index range() const { return m_range; }
  bool direct() const { return m_direct; }

 private:
  StorageIndex m_lo = 0;
  Unsigned m_scale = 0;
  Index m_range = 1;
  bool m_direct = true;
};

// Sums (i, v) pairs with lo <= i <= hi in storage the caller owns. Slot i - lo holds index i or nothing, so one scan
// yields the sorted, duplicate-free pattern. The table has a slot for every index of [lo, hi]: it suits a vector
// that is dense in its range, see direct().
//
// Empty slots hold -1. Every other value of StorageIndex can be an index: a matrix may have highest() + 1 rows.
template <typename Scalar, typename StorageIndex>
class direct_accumulator {
  // Used by value only: in C++14 a static constexpr member that is bound to a reference needs a definition.
  static constexpr StorageIndex kEmpty = StorageIndex(-1);
  // compact() tests this many slots at once for being all empty.
  static constexpr Index kBlock = 16;
  // direct() accepts up to this many slots per key: emptying the table is one fill and compact() skips its empty
  // blocks, so unused slots are cheap.
  static constexpr Index kDirectRatio = 16;

 public:
  // Slots needed for keys in [lo, hi].
  static Index size(StorageIndex lo, StorageIndex hi) { return Index(hi) - Index(lo) + 1; }

  // Whether the table is worth its slots for at most maxKeys distinct keys in [lo, hi].
  static bool direct(StorageIndex lo, StorageIndex hi, Index maxKeys) { return size(lo, hi) <= kDirectRatio * maxKeys; }

  // Binds the table to keys[0, size(lo, hi)) and values[0, size(lo, hi)) and empties it.
  void init(StorageIndex* keys, Scalar* values, StorageIndex lo, StorageIndex hi) {
    m_keys = keys;
    m_values = values;
    m_lo = lo;
    m_top = 0;
    std::fill_n(keys, size(lo, hi), StorageIndex(kEmpty));
  }

  // The value of index i. If i is absent, inserts it with an unspecified value and sets inserted.
  Scalar& findOrInsert(StorageIndex i, bool& inserted) {
    Index s = Index(i) - Index(m_lo);
    inserted = m_keys[s] == kEmpty;
    if (inserted) {
      m_keys[s] = i;
      m_top = numext::maxi(m_top, s);
    }
    return m_values[s];
  }

  // Calls func(value, inserted, it) for each entry of the inner iterator it, with the value of index it.index() as
  // findOrInsert returns it. The table state stays in locals over the whole vector; findOrInsert reloads it from the
  // members at each call, since the stores to the table may alias them.
  template <typename InnerIterator, typename Func>
  void accumulate(InnerIterator it, const Func& func) {
    StorageIndex* keys = m_keys;
    Scalar* values = m_values;
    Index top = m_top;
    Index lo = m_lo;
    for (; it; ++it) {
      StorageIndex i = StorageIndex(it.index());
      Index s = Index(i) - lo;
      bool inserted = keys[s] == kEmpty;
      if (inserted) {
        keys[s] = i;
        top = numext::maxi(top, s);
      }
      func(values[s], inserted, it);
    }
    m_top = top;
  }

  // Moves the n entries to keys[0, n) and values[0, n) in increasing index order and returns n. The write slot never
  // passes the read slot, so the table may sit where its output belongs. init() must run before the next use.
  Index compact() {
    // Entries before the first empty slot are already in place; after it, n < s, so no value moves onto itself.
    Index s = 0;
    while (s <= m_top && m_keys[s] != kEmpty) ++s;
    Index n = s;
    // A block of kBlock empty slots is skipped after one test of all of them; its fixed length lets the compiler
    // vectorize the test.
    for (; s + kBlock <= m_top + 1; s += kBlock) {
      StorageIndex occupied = 0;
      for (Index k = 0; k < kBlock; ++k) occupied |= m_keys[s + k] ^ kEmpty;
      if (occupied == 0) continue;
      for (Index k = s; k < s + kBlock; ++k)
        if (m_keys[k] != kEmpty) {
          m_keys[n] = m_keys[k];
          m_values[n] = move_if_assignable(m_values[k]);
          ++n;
        }
    }
    for (; s <= m_top; ++s)
      if (m_keys[s] != kEmpty) {
        m_keys[n] = m_keys[s];
        m_values[n] = move_if_assignable(m_values[s]);
        ++n;
      }
    return n;
  }

 private:
  StorageIndex* m_keys = nullptr;
  Scalar* m_values = nullptr;
  StorageIndex m_lo = 0;
  Index m_top = 0;
};

// The scratch storage of sort_inner_vector and merge_buckets, kept between calls.
template <typename Scalar, typename StorageIndex>
struct inner_sort_workspace {
  // The average number of entries per bucket: with one entry per bucket the counters cost as much as the entries,
  // and a long bucket is slow to sort by insertion.
  static constexpr Index kBucketLoad = 4;
  // A bucket with more entries than this holds a cluster and is not sorted by insertion. Splitting it must make at
  // least two buckets, or the split would not end.
  static constexpr Index kBucketMax = 8;
  // Makes room for n entries in storage.
  static void reserve(CompressedStorage<Scalar, StorageIndex>& storage, Index n) {
    if (storage.allocatedSize() < n) storage.reserve(n);
  }
  // The entries in their buckets, and a second array of the same extent that an overfull bucket is split into.
  CompressedStorage<Scalar, StorageIndex> buckets, split;
  // A direct table.
  CompressedStorage<Scalar, StorageIndex> table;
  // cursor[base + b] is the end of bucket b, for the buckets of one split; the next split follows them.
  Matrix<Index, Dynamic, 1> cursor;
  direct_accumulator<Scalar, StorageIndex> acc;
};

// The hash that sends the indices of n entries in [lo, hi] to their buckets.
template <typename Scalar, typename StorageIndex>
monotone_hash<StorageIndex> bucket_hash(StorageIndex lo, StorageIndex hi, Index n) {
  Index buckets = n / inner_sort_workspace<Scalar, StorageIndex>::kBucketLoad;
  return monotone_hash<StorageIndex>(lo, hi, numext::maxi<Index>(1, buckets));
}

// Moves the n entries (srcIndices[k], srcValues[k]), with indices in [lo, hi], to their buckets in (dstIndices,
// dstValues), in the order of arrival within a bucket. Returns the number of buckets and sets cursor[base + b] to the
// end of bucket b; bucket 0 begins at 0.
template <typename Scalar, typename StorageIndex>
Index split_into_buckets(const StorageIndex* srcIndices, Scalar* srcValues, Index n, StorageIndex lo, StorageIndex hi,
                         StorageIndex* dstIndices, Scalar* dstValues, Matrix<Index, Dynamic, 1>& cursor, Index base) {
  monotone_hash<StorageIndex> hash = bucket_hash<Scalar>(lo, hi, n);
  Index buckets = hash.range();
  if (cursor.size() < base + buckets + 1) cursor.conservativeResize(base + buckets + 1);
  // ends[b + 1] counts bucket b, then ends[b] is where its next entry goes.
  Index* ends = cursor.data() + base;
  std::fill_n(ends, buckets + 1, Index(0));
  for (Index k = 0; k < n; ++k) ++ends[hash(srcIndices[k]) + 1];
  for (Index b = 1; b < buckets; ++b) ends[b + 1] += ends[b];
  for (Index k = 0; k < n; ++k) {
    Index position = ends[hash(srcIndices[k])]++;
    dstIndices[position] = srcIndices[k];
    dstValues[position] = move_if_assignable(srcValues[k]);
  }
  return buckets;
}

// Appends the entries of the buckets to (indices, values) from position count, sorted by index and with the values
// of equal indices summed, and returns the new count. Bucket b is [cursor[base + b - 1], cursor[base + b]) of
// (bucketIndices, bucketValues), bucket 0 beginning at 0, and the indices of a bucket are below those of the next.
// (splitIndices, splitValues) has the extent of the buckets; both are overwritten. cursor is workspace.cursor.
//
// A bucket is sorted by insertion. One with more than kBucketMax entries holds a cluster, which a hash over all the
// buckets could not spread. It goes through a direct table if it is dense in its own range, and is otherwise split
// again over that range. A split has at least two buckets, each with about half of the range or less, so n entries
// cost O(n d) for d nested splits, d <= log2(hi - lo); d is 1 unless clusters lie within clusters.
template <typename Scalar, typename StorageIndex>
Index merge_buckets(StorageIndex* bucketIndices, Scalar* bucketValues, StorageIndex* splitIndices, Scalar* splitValues,
                    inner_sort_workspace<Scalar, StorageIndex>& workspace, Index base, Index buckets,
                    StorageIndex* indices, Scalar* values, Index count) {
  using Workspace = inner_sort_workspace<Scalar, StorageIndex>;
  constexpr Index kBucketMax = Workspace::kBucketMax;
  EIGEN_STATIC_ASSERT(Workspace::kBucketMax >= 2 * Workspace::kBucketLoad, AN_OVERFULL_BUCKET_MUST_SPLIT_IN_TWO)
  Index begin = 0;
  for (Index b = 0; b < buckets; ++b) {
    Index end = workspace.cursor[base + b];
    Index n = end - begin;
    if (n > kBucketMax) {
      StorageIndex lo = bucketIndices[begin], hi = bucketIndices[begin];
      for (Index p = begin + 1; p < end; ++p) {
        lo = numext::mini(lo, bucketIndices[p]);
        hi = numext::maxi(hi, bucketIndices[p]);
      }
      direct_accumulator<Scalar, StorageIndex>& acc = workspace.acc;
      if (acc.direct(lo, hi, n)) {
        Workspace::reserve(workspace.table, acc.size(lo, hi));
        acc.init(workspace.table.indexPtr(), workspace.table.valuePtr(), lo, hi);
        for (Index p = begin; p < end; ++p) {
          bool inserted;
          Scalar& value = acc.findOrInsert(bucketIndices[p], inserted);
          if (inserted)
            value = move_if_assignable(bucketValues[p]);
          else
            value += bucketValues[p];
        }
        Index distinct = acc.compact();
        std::copy(workspace.table.indexPtr(), workspace.table.indexPtr() + distinct, indices + count);
        move_or_copy(workspace.table.valuePtr(), workspace.table.valuePtr() + distinct, values + count);
        count += distinct;
      } else {
        // The split has at least two buckets, the first with lo and the last with hi, so each is smaller than this
        // one.
        Index next = base + buckets + 1;
        Index parts = split_into_buckets(bucketIndices + begin, bucketValues + begin, n, lo, hi, splitIndices + begin,
                                         splitValues + begin, workspace.cursor, next);
        count = merge_buckets(splitIndices + begin, splitValues + begin, bucketIndices + begin, bucketValues + begin,
                              workspace, next, parts, indices, values, count);
      }
      begin = end;
      continue;
    }
    Index first = count;
    for (Index p = begin; p < end; ++p) {
      StorageIndex i = bucketIndices[p];
      Index q = count;
      while (q > first && indices[q - 1] > i) --q;
      if (q > first && indices[q - 1] == i) {
        values[q - 1] += bucketValues[p];
        continue;
      }
      for (Index t = count; t > q; --t) {
        indices[t] = indices[t - 1];
        values[t] = move_if_assignable(values[t - 1]);
      }
      indices[q] = i;
      values[q] = move_if_assignable(bucketValues[p]);
      ++count;
    }
    begin = end;
  }
  return count;
}

// Sorts the n distinct entries (indices[k], values[k]) by index, given the lowest and highest index and the number of
// descents, the positions k with indices[k] < indices[k - 1]: the entries form descents + 1 increasing runs.
//
// - No descent: sorted.
// - Up to kSmall entries: insertion sort, unless they fill half of their span.
// - Few runs: insertion sort, which moves an entry only past the larger ones before it, within a budget of moves.
//   Long interleaved runs exhaust it.
// - Otherwise, and for what the budget leaves unsorted: a direct table if the entries are dense in their span, which
//   sorts them without a comparison, and buckets if they are not.
template <typename Scalar, typename StorageIndex>
void sort_inner_vector(StorageIndex* indices, Scalar* values, Index n, StorageIndex lo, StorageIndex hi, Index descents,
                       inner_sort_workspace<Scalar, StorageIndex>& workspace) {
  using Workspace = inner_sort_workspace<Scalar, StorageIndex>;
  constexpr Index kSmall = 64;
  if (descents == 0) return;
  bool small = n <= kSmall && Index(hi) - Index(lo) + 1 > 2 * n;
  Index sortedEnd = 1;
  if (small || (n > kSmall && descents * 4 <= n)) {
    Index budget = small ? n * n : 4 * n;
    for (; sortedEnd < n && budget >= 0; ++sortedEnd) {
      Index q = sortedEnd;
      if (indices[q - 1] < indices[q]) continue;
      StorageIndex index = indices[q];
      Scalar value = std::move(values[q]);
      do {
        indices[q] = indices[q - 1];
        values[q] = move_if_assignable(values[q - 1]);
        --q;
      } while (q > 0 && indices[q - 1] > index);
      indices[q] = index;
      values[q] = move_if_assignable(value);
      budget -= sortedEnd - q;
    }
  }
  if (sortedEnd == n) return;
  direct_accumulator<Scalar, StorageIndex>& acc = workspace.acc;
  if (acc.direct(lo, hi, n)) {
    Workspace::reserve(workspace.table, acc.size(lo, hi));
    acc.init(workspace.table.indexPtr(), workspace.table.valuePtr(), lo, hi);
    for (Index k = 0; k < n; ++k) {
      bool inserted;
      acc.findOrInsert(indices[k], inserted) = move_if_assignable(values[k]);
    }
    acc.compact();
    std::copy(workspace.table.indexPtr(), workspace.table.indexPtr() + n, indices);
    move_or_copy(workspace.table.valuePtr(), workspace.table.valuePtr() + n, values);
    return;
  }
  Workspace::reserve(workspace.buckets, n);
  Workspace::reserve(workspace.split, n);
  Index buckets = split_into_buckets(indices, values, n, lo, hi, workspace.buckets.indexPtr(),
                                     workspace.buckets.valuePtr(), workspace.cursor, 0);
  merge_buckets(workspace.buckets.indexPtr(), workspace.buckets.valuePtr(), workspace.split.indexPtr(),
                workspace.split.valuePtr(), workspace, 0, buckets, indices, values, 0);
}

// The entries a sparse product stores: all of them, or those with !(|value| <= tolerance), which keeps a NaN.
struct sparse_product_keep_all {
  template <typename Scalar>
  constexpr bool operator()(const Scalar&) const {
    return true;
  }
};
template <typename RealScalar>
struct sparse_product_keep_above {
  RealScalar tolerance;
  template <typename Scalar>
  bool operator()(const Scalar& value) const {
    return !(numext::abs(value) <= tolerance);
  }
};

// Removes the entries (indices[k], values[k]), k < n, whose value keep rejects, keeps the order of the others and
// returns their number. With a marker, sets marker[i] = -1 for every index i at or after the first removed entry.
template <typename Scalar, typename StorageIndex, typename Keep>
Index prune_inner_vector(StorageIndex* indices, Scalar* values, Index n, const Keep& keep,
                         StorageIndex* marker = nullptr) {
  Index kept = 0;
  while (kept < n && keep(values[kept])) ++kept;
  for (Index k = kept; k < n; ++k) {
    if (marker != nullptr) marker[indices[k]] = StorageIndex(-1);
    if (!keep(values[k])) continue;
    indices[kept] = indices[k];
    values[kept] = move_if_assignable(values[k]);
    ++kept;
  }
  return kept;
}

// Builds the inner vectors of a compressed sparse matrix one after the other in its storage, each as a sum of terms
// with sorted indices and without the entries a predicate rejects. A term is a sparse vector. The storage grows by
// doubling and is not shrunk.
//
// Terms describes one sum, Term iterates over the entries of one term:
//   Terms: Index count(), Index maxEntries(), void range(StorageIndex& lo, StorageIndex& hi), Term term(Index t) for
//          t < 2, void forEachTerm(const F& f) calling f(Term) in order
//   Term:  explicit operator bool(), operator++(), StorageIndex index(), void assignTo(Scalar&), void addTo(Scalar&)
//
// A sum of one or two terms with strictly increasing indices is merged: one term needs no accumulator and two merge
// in order. After the direct referencing of Parger et al., "spECK", PPoPP 2020, doi:10.1145/3332466.3374521, and the
// row merging of Gremse et al., SIAM J. Sci. Comput. 37(1), 2015, doi:10.1137/130948811. Any other sum is
// - marked: a marker of the inner dimension locates each index in the vector, so an entry costs one lookup however
//   many terms it sums, and the distinct entries are sorted afterwards. For short vectors and repeated indices.
// - direct: a direct_accumulator sums the entries. For long vectors that are dense in their range.
// - appended: each entry goes to the bucket of its index and the buckets are sorted and summed (merge_buckets). For
//   long vectors that are sparse in their range.
// A vector is marked unless the previous one had at least kUnmarkedMin distinct indices with at most two entries
// each; the first is not marked. A vector that would be marked is direct if hi - lo + 1 <= kDirectRatio * (distinct
// indices of the previous vector): the table then replaces the sort, as the dense accumulation of spECK does.
//
// The predicate must see the finished sum of an index, so each vector is pruned once it is complete.
template <typename Scalar, typename StorageIndex>
class sparse_accumulator {
  using Storage = CompressedStorage<Scalar, StorageIndex>;
  using Workspace = inner_sort_workspace<Scalar, StorageIndex>;
  // A vector with fewer distinct indices than this is marked; setting up the other two ways costs more.
  static constexpr Index kUnmarkedMin = 48;

 public:
  // Vectors have indices in [0, innerSize) and are written to data from position 0.
  sparse_accumulator(Index innerSize, Storage& data) : m_data(data), m_innerSize(innerSize) {}

  // The number of entries written, which is where the next vector begins.
  Index size() const { return m_size; }

  // Appends the vector that is the sum of terms, without the entries that keep rejects.
  template <typename Terms, typename Keep>
  void append(const Terms& terms, const Keep& keep) {
    Index maxKeys = terms.maxEntries();
    if (maxKeys == 0) return;
    Index keys = terms.count() <= 2 ? merge(terms, maxKeys, keep) : Index(-1);
    if (keys < 0) {
      StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
      bool direct = false;
      if (!m_marked || m_prevKeys >= kUnmarkedMin) {
        terms.range(lo, hi);
        direct = m_workspace.acc.direct(lo, hi, m_marked ? m_prevKeys : maxKeys);
      }
      if (direct)
        keys = sumDirect(terms, lo, hi, keep);
      else if (!m_marked)
        keys = sumAppended(terms, lo, hi, maxKeys, keep);
      else
        keys = sumMarked(terms, maxKeys, keep);
    }
    m_marked = keys < kUnmarkedMin || 2 * keys < maxKeys;
    m_prevKeys = keys;
  }

  // Makes the entries written the content of the storage.
  void finish() { m_data.resize(m_size); }

 private:
  // Makes room for extra entries after those written.
  EIGEN_ALWAYS_INLINE void reserve(Index extra) {
    if (m_data.allocatedSize() >= m_size + extra) return;
    m_data.resize(m_size);
    m_data.reserve(numext::maxi(extra, m_size));
  }

  // Prunes the keys entries written after the vectors and makes them the next vector.
  template <typename Keep>
  EIGEN_ALWAYS_INLINE void close(Index keys, const Keep& keep) {
    m_size += prune_inner_vector(m_data.indexPtr() + m_size, m_data.valuePtr() + m_size, keys, keep);
  }

  // Each of the following writes one vector and returns its number of distinct indices before pruning. merge()
  // returns -1 and writes none if the output is not strictly increasing: a term is unsorted or repeats an index.
  template <typename Terms, typename Keep>
  Index merge(const Terms& terms, Index maxKeys, const Keep& keep) {
    reserve(maxKeys);
    StorageIndex* indices = m_data.indexPtr() + m_size;
    Scalar* values = m_data.valuePtr() + m_size;
    Index n = 0;
    bool increasing = true;
    StorageIndex last = 0;
    auto push = [&](StorageIndex i) {
      increasing &= n == 0 || last < i;
      last = i;
      indices[n++] = i;
    };
    auto a = terms.term(0);
    if (terms.count() == 2) {
      auto b = terms.term(1);
      while (a && b) {
        StorageIndex ia = a.index(), ib = b.index();
        if (ia <= ib) {
          a.assignTo(values[n]);
          if (ia == ib) {
            b.addTo(values[n]);
            ++b;
          }
          ++a;
          push(ia);
        } else {
          b.assignTo(values[n]);
          ++b;
          push(ib);
        }
      }
      for (; b; ++b) {
        b.assignTo(values[n]);
        push(b.index());
      }
    }
    for (; a; ++a) {
      a.assignTo(values[n]);
      push(a.index());
    }
    if (!increasing) return -1;
    close(n, keep);
    return n;
  }

  template <typename Terms, typename Keep>
  Index sumDirect(const Terms& terms, StorageIndex lo, StorageIndex hi, const Keep& keep) {
    direct_accumulator<Scalar, StorageIndex>& acc = m_workspace.acc;
    reserve(acc.size(lo, hi));
    acc.init(m_data.indexPtr() + m_size, m_data.valuePtr() + m_size, lo, hi);
    terms.forEachTerm([&acc](const typename Terms::Term& term) {
      acc.accumulate(term, [](Scalar& value, bool inserted, const typename Terms::Term& entry) {
        if (inserted)
          entry.assignTo(value);
        else
          entry.addTo(value);
      });
    });
    Index keys = acc.compact();
    close(keys, keep);
    return keys;
  }

  template <typename Terms, typename Keep>
  Index sumAppended(const Terms& terms, StorageIndex lo, StorageIndex hi, Index maxKeys, const Keep& keep) {
    monotone_hash<StorageIndex> hash = bucket_hash<Scalar>(lo, hi, maxKeys);
    Index buckets = hash.range();
    // cursor[b + 1] counts the entries of bucket b; cursor[b] then becomes where its next entry goes.
    Matrix<Index, Dynamic, 1>& cursor = m_workspace.cursor;
    if (cursor.size() < buckets + 1) cursor.resize(buckets + 1);
    cursor.head(buckets + 1).setZero();
    Index* ends = cursor.data();
    terms.forEachTerm([&](typename Terms::Term term) {
      for (; term; ++term) ++ends[hash(term.index()) + 1];
    });
    for (Index b = 1; b < buckets; ++b) ends[b + 1] += ends[b];
    Workspace::reserve(m_workspace.buckets, maxKeys);
    Workspace::reserve(m_workspace.split, maxKeys);
    StorageIndex* bucketIndices = m_workspace.buckets.indexPtr();
    Scalar* bucketValues = m_workspace.buckets.valuePtr();
    terms.forEachTerm([&](typename Terms::Term term) {
      for (; term; ++term) {
        StorageIndex i = term.index();
        Index position = ends[hash(i)]++;
        bucketIndices[position] = i;
        term.assignTo(bucketValues[position]);
      }
    });
    reserve(maxKeys);
    Index end = merge_buckets(bucketIndices, bucketValues, m_workspace.split.indexPtr(), m_workspace.split.valuePtr(),
                              m_workspace, 0, buckets, m_data.indexPtr(), m_data.valuePtr(), m_size);
    Index keys = end - m_size;
    close(keys, keep);
    return keys;
  }

  template <typename Terms, typename Keep>
  Index sumMarked(const Terms& terms, Index maxKeys, const Keep& keep) {
    if (m_marker.size() == 0) m_marker.setConstant(m_innerSize, StorageIndex(-1));
    reserve(numext::mini(maxKeys, m_innerSize));
    // marker[i] >= start locates inner index i in the vector being built, which begins at start.
    StorageIndex* marker = m_marker.data();
    StorageIndex* indices = m_data.indexPtr();
    Scalar* values = m_data.valuePtr();
    Index start = m_size, nnz = m_size;
    Index descents = 0;
    StorageIndex prev = 0, lo = NumTraits<StorageIndex>::highest(), hi = 0;
    terms.forEachTerm([&](typename Terms::Term term) {
      for (; term; ++term) {
        StorageIndex i = term.index();
        Index position = marker[i];
        if (position >= start) {
          term.addTo(values[position]);
        } else {
          descents += i < prev;
          prev = i;
          lo = numext::mini(lo, i);
          hi = numext::maxi(hi, i);
          marker[i] = convert_index<StorageIndex>(nnz);
          indices[nnz] = i;
          term.assignTo(values[nnz]);
          ++nnz;
        }
      }
    });
    Index keys = nnz - start;
    // A removed or moved entry leaves a marker at or past the new end of the vector, which the next vector would
    // take for its own. Removing entries adds no descent, and [lo, hi] still bounds the rest.
    Index kept = prune_inner_vector(indices + start, values + start, keys, keep, marker);
    sort_inner_vector(indices + start, values + start, kept, lo, hi, descents, m_workspace);
    m_size = start + kept;
    return keys;
  }

  Storage& m_data;
  Index m_innerSize;
  Index m_size = 0;
  bool m_marked = false;
  // The number of distinct indices of the previous non-empty vector, before pruning.
  Index m_prevKeys = 0;
  Matrix<StorageIndex, Dynamic, 1> m_marker;
  Workspace m_workspace;
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_SPARSE_ACCUMULATOR_H
