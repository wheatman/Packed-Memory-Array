#pragma once
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "StructOfArrays/multipointer.hpp"
#include "StructOfArrays/soa.hpp"
#include "helpers.hpp"
#include "timers.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ranges>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <bit>
#include <limits>
#include <malloc.h>
#include <tuple>
#include <type_traits>

#include "parlay/sequence.h"

/*
  Only works to store unique elements
  top bit is 1 for continue, or 0 for the end of an element
*/

template <class T> class delta_compressed_leaf {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T can only be 4 or 8 bytes");

public:
  static constexpr size_t max_element_size = sizeof(T) + (sizeof(T) / 4);
  static constexpr bool compressed = true;
  static constexpr bool binary = true;
  using key_type = T;

  using element_type = std::tuple<key_type>;

  using element_ref_type = std::tuple<key_type &>;

  using element_ptr_type = MultiPointer<key_type>;

  using SOA_type = SOA<key_type>;
  using value_type = std::tuple<>;

  T &head;

  T head_key() const { return head; }
  uint8_t *array;
  const int64_t length_in_bytes;

  T *T_array() const { return (T *)array; }

  // just for an optimized compare to end
  class iterator_end {};

  // to scan over the data when it is in valid state
  // does not deal with out of place data
  class iterator {
    // index considers the head so into the array its offset by 1
    const uint8_t *ptr;
    T curr_elem = 0;

  public:
    iterator(T first_elem, const uint8_t *pointer_to_next_elem)
        : ptr(pointer_to_next_elem), curr_elem(first_elem) {}

    iterator(element_ref_type first_elem, element_ptr_type pointer_to_next_elem)
        : ptr((uint8_t *)pointer_to_next_elem.get_pointer()),
          curr_elem(std::get<0>(first_elem)) {}

    // only for use comparing to end
    bool operator!=([[maybe_unused]] const iterator_end &end) const {
      return curr_elem != 0;
    }
    bool operator==([[maybe_unused]] const iterator_end &end) const {
      return curr_elem == 0;
    }

    iterator &operator++() {
      DecodeResult dr(ptr);
      curr_elem += dr.difference;
      ptr += dr.old_size;
      if (dr.difference == 0) {
        curr_elem = 0;
      }
      return *this;
    }
    bool inc_and_check_end() {
      DecodeResult dr(ptr);
      curr_elem += dr.difference;
      ptr += dr.old_size;
      if (dr.difference == 0) {
        curr_elem = 0;
        return true;
      }
      return false;
    }
    T operator*() const { return curr_elem; }
    [[nodiscard]] void *get_pointer() const { return (void *)ptr; }
  };
  iterator begin() const { return iterator(head, array); }
  iterator_end end() const { return {}; }

private:
  class FindResult {
  public:
    T difference;
    int64_t loc;
    int64_t size;
    void print() {
      std::cout << "FindResult { difference=" << difference << ", loc=" << loc
                << ", size=" << size << " }" << std::endl;
    }
    bool operator==(const FindResult &other) const {
      return (difference == other.difference) && (loc == other.loc) &&
             (size == other.size);
    }
  };
  class EncodeResult {
  public:
    static constexpr int storage_size = std::max(max_element_size + 1, 8UL);
    uint8_t data[storage_size] = {0};
    int64_t size;
    void print() {
      std::cout << "EncodeResult { data={";
      for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<uint32_t>(data[i]) << ", ";
      }
      std::cout << "} , size=" << size << " }" << std::endl;
    }
    static int64_t write_encoded(T difference, uint8_t *loc) {
      loc[0] = difference & 0x7FU;
      int64_t num_bytes = difference > 0;
      difference >>= 7;
      while (difference) {
        loc[num_bytes - 1] |= 0x80U;
        loc[num_bytes] = difference & 0x7FU;
        num_bytes += 1;
        difference >>= 7;
      }
      return num_bytes;
    }
    EncodeResult(T difference) {
      assert(difference != 0);
      size = write_encoded(difference, data);
      assert((size_t)size <= max_element_size);
    }
  };
  class DecodeResult {
  public:
    T difference = 0;
    int64_t old_size = 0;
    void print() {
      std::cout << "DecodeResult { difference=" << difference
                << ", old_size=" << old_size << " }" << std::endl;
    }
    static constexpr std::array<uint64_t, 8> extract_masks = {
        0x000000000000007FUL, 0x0000000000007F7FUL, 0x00000000007F7F7FUL,
        0x000000007F7F7F7FUL, 0x0000007F7F7F7F7FUL, 0x00007F7F7F7F7F7FUL,
        0x007F7F7F7F7F7F7FUL, 0x7F7F7F7F7F7F7F7FUL};

    static constexpr std::array<uint64_t, 8> extract_masks2 = {
        0b1111111UL,
        0b11111111111111UL,
        0b111111111111111111111UL,
        0b1111111111111111111111111111UL,
        0b11111111111111111111111111111111111UL,
        0b111111111111111111111111111111111111111111UL,
        0b1111111111111111111111111111111111111111111111111UL,
        0b11111111111111111111111111111111111111111111111111111111UL};
    static constexpr std::array<uint64_t, 9> extract_masks2_ = {
        0,
        0b1111111UL,
        0b11111111111111UL,
        0b111111111111111111111UL,
        0b1111111111111111111111111111UL,
        0b11111111111111111111111111111111111UL,
        0b111111111111111111111111111111111111111111UL,
        0b1111111111111111111111111111111111111111111111111UL,
        0b11111111111111111111111111111111111111111111111111111111UL};

    static constexpr std::array<uint64_t, 255> get_extract_masks3() {
      std::array<uint64_t, 255> arr = {0};
      for (uint64_t i = 0; i < 255; i++) {
        if ((i & 1UL) == 0) {
          arr[i] = 0b1111111UL;
        } else if ((i & 2UL) == 0) {
          arr[i] = 0b11111111111111UL;
        } else if ((i & 4UL) == 0) {
          arr[i] = 0b111111111111111111111UL;
        } else if ((i & 8UL) == 0) {
          arr[i] = 0b1111111111111111111111111111UL;
        } else if ((i & 16UL) == 0) {
          arr[i] = 0b11111111111111111111111111111111111UL;
        } else if ((i & 32UL) == 0) {
          arr[i] = 0b111111111111111111111111111111111111111111UL;
        } else if ((i & 64UL) == 0) {
          arr[i] = 0b1111111111111111111111111111111111111111111111111UL;
        } else if ((i & 128UL) == 0) {
          arr[i] = 0b11111111111111111111111111111111111111111111111111111111UL;
        }
      }

      return arr;
    }

    static constexpr std::array<uint64_t, 255> extract_masks3 =
        get_extract_masks3();

    static constexpr std::array<uint8_t, 255> get_extract_masks4() {
      std::array<uint8_t, 255> arr = {0};
      for (uint64_t i = 0; i < 255; i++) {
        if ((i & 1UL) == 0) {
          arr[i] = 1;
        } else if ((i & 2UL) == 0) {
          arr[i] = 2;
        } else if ((i & 4UL) == 0) {
          arr[i] = 3;
        } else if ((i & 8UL) == 0) {
          arr[i] = 4;
        } else if ((i & 16UL) == 0) {
          arr[i] = 5;
        } else if ((i & 32UL) == 0) {
          arr[i] = 6;
        } else if ((i & 64UL) == 0) {
          arr[i] = 7;
        } else if ((i & 128UL) == 0) {
          arr[i] = 8;
        }
      }

      return arr;
    }

    static constexpr std::array<uint8_t, 255> extract_masks4 =
        get_extract_masks4();

    DecodeResult() = default;

    DecodeResult(T d, int64_t s) : difference(d), old_size(s) {}

    DecodeResult(const uint8_t *loc) {

      if ((*loc & 0x80UL) == 0) {
        difference = *loc;
        old_size = *loc > 0;
        return;
      }
#if __BMI2__ == 1
      uint64_t chunks = unaligned_load<uint64_t>(loc);

      // these come in and out depending on the relative cost of
      // parallel extract and count_trailing_zeroes (tzcount)
      // and branches on your specific hardware
      // if ((chunks & 0x8000UL) == 0) {
      //   // T difference_guess = _pext_u64(chunks, 0x0000000000007F7FUL);
      //   difference = (*loc & 0x7F) | (*(loc + 1) << 7);
      //   // difference = difference_guess;
      //   old_size = 2;
      //   return;
      // }
      // if ((chunks & 0x800000UL) == 0) {
      //   // T difference_guess = _pext_u64(chunks, 0x00000000007F7F7FUL);
      //   difference =
      //       (*loc & 0x7F) | ((*(loc + 1) & 0x7F) << 7) | (*(loc + 2) << 14);

      //   // difference = difference_guess;
      //   old_size = 3;
      //   return;
      // }
      // if ((chunks & 0x80000000UL) == 0) {
      //   T difference_guess = _pext_u64(chunks, 0x000000007F7F7F7FUL);
      //   difference = difference_guess;
      //   old_size = 4;
      //   return;
      // }
      uint64_t mask = _pext_u64(chunks, 0x8080808080808080UL);
      if (sizeof(T) == 4 ||
          (chunks & 0x8080808080808080UL) != 0x8080808080808080UL) [[likely]] {
        // int32_t index = std::countr_zero(~mask);
        // difference = _pext_u64(chunks, extract_masks[index]);
        // difference =
        //     _pext_u64(chunks, 0x7F7F7F7F7F7F7F7FUL) & extract_masks2[index];
        difference =
            _pext_u64(chunks, 0x7F7F7F7F7F7F7F7FUL) & extract_masks3[mask];
        // assert(difference == (_pext_u64(chunks, 0x7F7F7F7F7F7F7F7FUL) &
        //                       extract_masks2[std::countr_zero(~mask)]));
        old_size = extract_masks4[mask];
        // old_size = index + 1;
        // assert(old_size == std::countr_zero(~mask) + 1);
        // printf("chunks = %lx, mask = %lx, index = %u, difference =%lu\n",
        //        chunks, mask, old_size, difference);
        return;
      }

#endif
      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          ASSERT(shift_amount < 8 * sizeof(T), "shift_amount = %u\n",
                 shift_amount);
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }

#if __BMI2__ == 1 && __AVX512BW__ == 1
    DecodeResult(const uint8_t *loc, uint64_t mask) {

      if ((*loc & 0x80UL) == 0) {
        difference = *loc;
        old_size = *loc > 0;
        return;
      }
      uint64_t chunks = unaligned_load<uint64_t>(loc);
      if (sizeof(T) == 4 ||
          (chunks & 0x8080808080808080UL) != 0x8080808080808080UL) [[likely]] {
        int32_t index = std::countr_zero(mask);
        difference =
            _pext_u64(chunks, 0x7F7F7F7F7F7F7F7FUL) & extract_masks2[index];
        old_size = index + 1;
        return;
      }

      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          ASSERT(shift_amount < 8 * sizeof(T), "shift_amount = %u\n",
                 shift_amount);
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }
#endif
  };

  // returns the starting byte location, the length of the specified element,
  // and the difference from the previous element length is 0 if the element
  // is not found starting byte location of 0 means this is the head
  // if we have just changed the head we pass in the old head val so we can
  // correctly interpret the bytes after that
  FindResult debug_find(T x) const {
    T curr_elem = head;
    T prev_elem = 0;
    // heads are dealt with separately
    assert(x > head);
    int64_t curr_loc = 0;
    DecodeResult dr;
    while (curr_elem < x) {
      dr = DecodeResult(array + curr_loc);
      prev_elem = curr_elem;
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
    }
    assert(curr_loc < length_in_bytes);
    if (x == curr_elem) {
      return {0, curr_loc - dr.old_size, dr.old_size};
    }
    // std::cout << "x = " << x << ", curr_elem = " << curr_elem << std::endl;
    return {x - prev_elem, curr_loc - dr.old_size, 0};
  }

  template <bool head_in_place> FindResult find(T x) const {

#if __BMI2__ == 1 && __AVX512BW__ == 1
    T curr_elem = head;
    T prev_elem = 0;
    // heads are dealt with separately
    assert(x > head);
    int64_t curr_loc = 0;
    DecodeResult dr;
    uint8_t *true_array_start = (head_in_place) ? array - sizeof(T) : array;
    uint64_t mask = ~_mm512_movepi8_mask(_mm512_loadu_si512(true_array_start));
    uint64_t mask_remaining = 64;
    if constexpr (head_in_place) {
      mask_remaining -= sizeof(T);
      mask >>= sizeof(T);
    }
    int64_t next_mask_offset = 64;
    while (curr_elem < x) {
      dr = DecodeResult(array + curr_loc, mask);
      prev_elem = curr_elem;
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      mask_remaining -= dr.old_size;
      mask >>= dr.old_size;
      if (mask_remaining <= 32 && next_mask_offset < length_in_bytes) {
        uint64_t next_mask = ~(uint32_t)_mm256_movemask_epi8(_mm256_loadu_si256(
            (__m256i *)(true_array_start + next_mask_offset)));
        next_mask <<= mask_remaining;
        mask |= next_mask;
        mask_remaining += 32;
        next_mask_offset += 32;
      }
    }
    assert(curr_loc < length_in_bytes);
    if (x == curr_elem) {
      assert((debug_find(x) ==
              FindResult{0, curr_loc - dr.old_size, dr.old_size}));
      return {0, curr_loc - dr.old_size, dr.old_size};
    }
    // std::cout << "x = " << x << ", curr_elem = " << curr_elem << std::endl;
    assert((debug_find(x) ==
            FindResult{x - prev_elem, curr_loc - dr.old_size, 0}));
    return {x - prev_elem, curr_loc - dr.old_size, 0};
#else
    return debug_find(x);
#endif
  }

  static T get_out_of_place_used_bytes(T *data) { return data[0]; }
  static void set_out_of_place_used_bytes(T *data, T bytes) { data[0] = bytes; }

  static T *get_out_of_place_pointer(T *data) {
    T *pointer = nullptr;
    memcpy(&pointer, data + 1, sizeof(T *));
    return pointer;
  }
  static T get_first_element_out_of_place(T *data) {
    T element;
    memcpy(&element, get_out_of_place_pointer(data), sizeof(T));
    return element;
  }
  static void set_out_of_place_pointer(T *data, T *pointer) {
    memcpy(data + 1, &pointer, sizeof(T *));
  }
  static T get_out_of_place_last_written(T *data) { return data[3]; }
  static void set_out_of_place_last_written(T *data, T last) { data[3] = last; }
  static T get_out_of_place_temp_size(T *data) { return data[4]; }
  static void set_out_of_place_temp_size(T *data, T size) { data[4] = size; }

  T get_out_of_place_used_bytes() const {
    return get_out_of_place_used_bytes(T_array());
  }
  void set_out_of_place_used_bytes(T bytes) const {
    set_out_of_place_used_bytes(T_array(), bytes);
  }

  T *get_out_of_place_pointer() const {
    return get_out_of_place_pointer(T_array());
  }
  T get_first_element_out_of_place() const {
    T element;
    memcpy(&element, get_out_of_place_pointer(), sizeof(T));
    return element;
  }
  void set_out_of_place_pointer(T *pointer) const {
    set_out_of_place_pointer(T_array(), pointer);
  }
  T get_out_of_place_last_written() const {
    return get_out_of_place_last_written(T_array());
  }
  void set_out_of_place_last_written(T last) const {
    set_out_of_place_last_written(T_array(), last);
  }
  T get_out_of_place_temp_size() const {
    return get_out_of_place_temp_size(T_array());
  }
  void set_out_of_place_temp_size(T size) const {
    set_out_of_place_temp_size(T_array(), size);
  }

  void slide_right(uint8_t *loc, uint64_t amount) {
    assert(amount <= max_element_size);
    std::memmove((void *)(loc + amount), (void *)loc,
                 length_in_bytes - (loc + amount - array));
    // std::memset((void *)loc, 0, amount);
  }

  void slide_left(uint8_t *loc, uint64_t amount) {
    assert(amount <= max_element_size);
    std::memmove((void *)(loc - amount), (void *)loc,
                 length_in_bytes - (loc + amount - array));
    std::memset((void *)(array + (length_in_bytes - amount)), 0, amount);
    // TODO(wheatman) get early exit working
    // uint16_t *long_data_pointer = (uint16_t *)loc;
    // uint16_t *long_data_pointer_offset = (uint16_t *)(loc - amount);
    // while (*long_data_pointer) {
    //   *long_data_pointer_offset = *long_data_pointer;
    //   long_data_pointer += 1;
    //   long_data_pointer_offset += 1;
    // }
    // *long_data_pointer_offset = 0;
    // assert(((uint8_t *)long_data_pointer) - array < length_in_bytes);
  }

  static void small_memcpy(uint8_t *dest, uint8_t *source, size_t n) {
    assert(n < 16);
    if (n >= 8) {
      unaligned_store(dest, unaligned_load<uint64_t>(source));
      source += 8;
      dest += 8;
      n -= 8;
    }
    assert(n < 8);
    if (n >= 4) {
      unaligned_store(dest, unaligned_load<uint32_t>(source));
      source += 4;
      dest += 4;
      n -= 4;
    }
    assert(n < 4);
    if (n >= 2) {
      unaligned_store(dest, unaligned_load<uint16_t>(source));
      source += 2;
      dest += 2;
      n -= 2;
    }
    assert(n < 2);
    if (n >= 1) {
      *dest = *source;
    }
  }

public:
  delta_compressed_leaf(T &head_, void *data_, int64_t length)
      : head(head_), array(static_cast<uint8_t *>(data_)),
        length_in_bytes(length) {}

  delta_compressed_leaf(element_ref_type head_, element_ptr_type data_,
                        int64_t length)
      : head(std::get<0>(head_)),
        array(static_cast<uint8_t *>((void *)data_.get_pointer())),
        length_in_bytes(length) {}

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements merged in, and number of bytes used in this
  // leaf)
  template <bool head_in_place, bool parallel, typename ValueUpdate,
            bool maintain_offsets>
  std::tuple<element_ptr_type, uint64_t, uint64_t>
  merge_into_leaf(element_ptr_type batch_start_, T *batch_end, uint64_t end_val,
                  [[maybe_unused]] ValueUpdate value_update,
                  [[maybe_unused]] auto offsets_array) {
    static_assert(!maintain_offsets);
    T *batch_start = batch_start_.get_pointer();
    // TODO(wheatman) deal with the case where end_val == max_uint and you still
    // want to add the last element
    //  case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      auto [inserted, byte_count] =
          insert<head_in_place, ValueUpdate, maintain_offsets>(
              *batch_start, value_update, offsets_array);
      return {batch_start + 1, inserted, byte_count};
    }

    // case 2: more than 1 elt from batch
    // two-finger merge into extra space, might overflow leaf
    uint64_t temp_size = length_in_bytes / sizeof(T);

    // get enough size for the batch
    while (batch_start + temp_size < batch_end &&
           batch_start[temp_size] < end_val) {
      temp_size *= 2;
    }
    // get enough size for the leaf
    temp_size = (temp_size * max_element_size) + length_in_bytes;

    // add sizeof(T) to store the head since it is out of place in the pma, but
    // with the data in the temp storage
    temp_size += sizeof(T);
    // so we can safely write off the end
    temp_size += max_element_size;
    if (temp_size % 32 != 0) {
      temp_size = (temp_size / 32) * 32 + 32;
    }
    uint8_t *temp_arr = (uint8_t *)aligned_alloc(32, temp_size);

#ifndef NDEBUG
    // helps some of the debug print statements to make sense
    std::memset(temp_arr, 0, temp_size);
#endif

    T *batch_ptr = batch_start;
    uint8_t *leaf_ptr = array;
    uint8_t *temp_ptr = temp_arr;

    uint64_t distinct_batch_elts = 0;
    const uint8_t *leaf_end = array + length_in_bytes;
    T last_written = 0;
    // deal with the head
    const T old_head = head;
    T last_in_leaf = old_head;
    // new head from batch
    if (batch_ptr[0] < old_head) {
      *((T *)temp_ptr) = *batch_ptr;
      last_written = *batch_ptr;
      distinct_batch_elts++;
      temp_ptr += sizeof(T);
      batch_ptr++;
      while (batch_ptr < batch_end && *batch_ptr == last_written) {
        ++batch_ptr;
      }
    } else { // copy over the old head
      *((T *)temp_ptr) = old_head;
      last_written = old_head;
      temp_ptr += sizeof(T);
    }
    // anything that needs go from the batch before the old head
    bool did_before_head = false;
    if constexpr (parallel && binary) {
      if (temp_size > 100 && *batch_ptr < old_head) {
        auto before_head_end = std::lower_bound(batch_ptr, batch_end, old_head);
        size_t number_before_head = before_head_end - batch_ptr;
        if (number_before_head > 1000) {
          size_t num_blocks = std::min(number_before_head / 100,
                                       (size_t)ParallelTools::getWorkers());
          if (num_blocks > 1) {
            std::vector<uint8_t *> blocks(num_blocks);
            std::vector<uint64_t> block_sizes(num_blocks + 1);
            std::vector<T *> block_ends(num_blocks + 1);
            blocks[0] = temp_ptr;
            block_ends[0] = batch_ptr;
            block_ends[num_blocks] = before_head_end;
            size_t per_block_count = number_before_head / num_blocks;
            ParallelTools::Reducer_sum<size_t> elements_added;
            for (size_t i = 1; i < num_blocks; i++) {
              size_t block_size = temp_size / num_blocks;
              block_size += max_element_size;
              if (block_size % 32 != 0) {
                block_size = (block_size / 32) * 32 + 32;
              }
              blocks[i] = (uint8_t *)aligned_alloc(32, block_size);
              block_ends[i] = batch_ptr + per_block_count * i;
            }
            ParallelTools::parallel_for(0, num_blocks, [&](size_t j) {
              // we know we can always look back one since we already past the
              // head, and each other one goes into the block before it
              uint8_t *block_temp_ptr = blocks[j];
              for (T *block_ptr = block_ends[j]; block_ptr < block_ends[j + 1];
                   block_ptr++) {
                T new_difference = *block_ptr - *(block_ptr - 1);
                if (new_difference > 0) {
                  int64_t er_size = EncodeResult::write_encoded(new_difference,
                                                                block_temp_ptr);
                  block_temp_ptr += er_size;
                  ++elements_added;
                }
              }
              block_sizes[j + 1] = block_temp_ptr - blocks[j];
            });
            for (size_t i = 1; i < num_blocks; i++) {
              block_sizes[i + 1] += block_sizes[i];
            }
            ParallelTools::parallel_for(1, num_blocks, [&](size_t j) {
              memcpy(temp_ptr + block_sizes[j], blocks[j],
                     block_sizes[j + 1] - block_sizes[j]);
              free(blocks[j]);
            });
            batch_ptr += number_before_head - 1;
            last_written = *batch_ptr;
            ++batch_ptr;
            temp_ptr += block_sizes[num_blocks];
            distinct_batch_elts += elements_added;
            did_before_head = true;
          }
        }
      }
    }
    if (!did_before_head) {
      while (batch_ptr < batch_end && batch_ptr[0] < old_head) {
        T new_difference = *batch_ptr - last_written;
        if (new_difference > 0) {
          int64_t er_size =
              EncodeResult::write_encoded(new_difference, temp_ptr);
          last_written = *batch_ptr;
          distinct_batch_elts++;
          temp_ptr += er_size;
        }
        batch_ptr++;
      }
    }
    // if we still need to copy the old leaf head
    if (leaf_ptr == array) {
      T new_difference = old_head - last_written;
      if (new_difference > 0) {
        int64_t er_size = EncodeResult::write_encoded(new_difference, temp_ptr);
        last_written = old_head;
        temp_ptr += er_size;
      }
    }

    // deal with the rest of the elements
    if (batch_ptr != batch_end && *batch_ptr < end_val) {
      while (true) {
        assert(leaf_ptr < leaf_end);
        const DecodeResult dr(leaf_ptr);
        // if duplicates in batch, skip
        if (*batch_ptr == last_written) {
          batch_ptr++;
          // std::cout << "skipping duplicate\n";
          if (batch_ptr == batch_end || *batch_ptr >= end_val) {
            break;
          }
          continue;
        }
        const T leaf_element = last_in_leaf + dr.difference;

        assert(leaf_element > last_written);
        // otherwise, do a step of the merge

        if (leaf_element <= *batch_ptr) {
          int64_t er_size;
          if (last_written != last_in_leaf) {
            er_size = EncodeResult::write_encoded(leaf_element - last_written,
                                                  temp_ptr);
          } else {
            er_size = dr.old_size;
            // small_memcpy(temp_ptr, leaf_ptr, dr.old_size);
            // everything has a few extra bytes on the end, so copy a fixed size
            // for simplicity and performance
            memcpy(temp_ptr, leaf_ptr, max_element_size);
          }
          last_written = leaf_element;
          assert(last_written < end_val);
          last_in_leaf = leaf_element;
          temp_ptr += er_size;
          leaf_ptr += dr.old_size;
          if (leaf_element == *batch_ptr) {
            batch_ptr++;
            if (batch_ptr == batch_end || *batch_ptr >= end_val) {
              break;
            }
          }
          if (*leaf_ptr == 0) {
            break;
          }
          assert(temp_ptr < temp_arr + temp_size);
        } else { // if (leaf_element > *batch_ptr) {
          int64_t er_size =
              EncodeResult::write_encoded(*batch_ptr - last_written, temp_ptr);
          last_written = *batch_ptr;
          assert(last_written < end_val);
          batch_ptr++;
          distinct_batch_elts++;
          temp_ptr += er_size;
          if (batch_ptr == batch_end || *batch_ptr >= end_val) {
            break;
          }
        }
      }
    }

    // write rest of the batch if it exists
    while (batch_ptr < batch_end && batch_ptr[0] < end_val) {
      if (*batch_ptr == last_written) {
        batch_ptr++;
        continue;
      }
      T new_difference = *batch_ptr - last_written;
      int64_t er_size = EncodeResult::write_encoded(new_difference, temp_ptr);
      last_written = *batch_ptr;
      assert(last_written < end_val);
      batch_ptr++;
      distinct_batch_elts++;
      temp_ptr += er_size;
      assert(temp_ptr < temp_arr + temp_size);
    }

    // write the rest of the original leaf if it exist
    // first write the next element which needs to calculate a new difference
    if (leaf_ptr < leaf_end) {
      DecodeResult dr(leaf_ptr);
      if (dr.old_size != 0) {
        T leaf_element = last_in_leaf + dr.difference;
        int64_t er_size =
            EncodeResult::write_encoded(leaf_element - last_written, temp_ptr);
        last_written = leaf_element;
        assert(last_written < end_val);
        temp_ptr += er_size;
        leaf_ptr += dr.old_size;
        // then just copy over the rest
        while (true) {
          assert(leaf_ptr < leaf_end);
          DecodeResult dr2(leaf_ptr);
          if (dr2.old_size == 0) {
            break;
          }
          // small_memcpy(temp_ptr, leaf_ptr, dr2.old_size);
          // everything has a few extra bytes on the end, so copy a fixed size
          // for simplicity and performance
          memcpy(temp_ptr, leaf_ptr, max_element_size);
          last_written += dr2.difference;
          assert(last_written < end_val);
          temp_ptr += dr2.old_size;
          leaf_ptr += dr2.old_size;
          assert(temp_ptr < temp_arr + temp_size);
        }
      }
    }
    assert(temp_ptr < temp_arr + temp_size);
    int64_t used_bytes = temp_ptr - temp_arr;
    // write the byte after to zero so nothing else tries to read off the end
    if ((uint64_t)used_bytes < temp_size - 1) {
      temp_ptr[0] = 0;
      temp_ptr[1] = 0;
    }
    assert((uint64_t)used_bytes < temp_size);
    // check if you can fit in the leaf with some extra space at the end for
    // safety
    if (used_bytes <= length_in_bytes - (int64_t)max_element_size) {
      head = *((T *)temp_arr);
      memcpy(array, temp_arr + sizeof(T), used_bytes - sizeof(T));
      free(temp_arr);
    } else { // special write for when you don't fit
      head = 0;
      set_out_of_place_used_bytes(used_bytes);
      set_out_of_place_pointer((T *)temp_arr);
      set_out_of_place_last_written(last_written);
      set_out_of_place_temp_size(temp_size);
      assert(last_written < end_val);
    }
    return {batch_ptr, distinct_batch_elts,
            used_bytes - ((head_in_place) ? 0 : sizeof(T)) /* for the head*/};
  }
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t>
  strip_from_leaf(T *batch_start, T *batch_end, uint64_t end_val) {
    // TODO(wheatman) deal with the case where end_val == max_uint and you still
    // want to add the last element
    //  case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      auto [removed, byte_count] =
          remove<head_in_place, false>(*batch_start, nullptr);
      return {batch_start + 1, removed, byte_count};
    }

    // case 2: more than 1 elt from batch

    T *batch_ptr = batch_start;
    uint8_t *front_pointer = array;
    uint8_t *back_pointer = array;

    uint64_t distinct_batch_elts = 0;
    const uint8_t *leaf_end = array + length_in_bytes;
    T last_written = 0;
    // deal with the head
    const T old_head = head;
    T last_in_leaf = old_head;
    // anything from the batch before the old head is skipped
    while (batch_ptr < batch_end && batch_ptr[0] < old_head) {
      batch_ptr++;
    }
    bool head_written = false;

    // if we are not removing the old leaf head
    if ((batch_ptr < batch_end && *batch_ptr != old_head) ||
        batch_ptr == batch_end) {
      last_written = old_head;
      assert(last_written < end_val);
      head_written = true;
    } else {
      distinct_batch_elts++;
    }

    // deal with the rest of the elemnts
    while (batch_ptr < batch_end && batch_ptr[0] < end_val &&
           front_pointer < leaf_end) {
      DecodeResult dr(front_pointer);
      if (dr.old_size == 0) {
        break;
      }
      T leaf_element = last_in_leaf + dr.difference;
      assert(leaf_element > last_written);
      // otherwise, do a step of the merge
      if (leaf_element == *batch_ptr) {
        front_pointer += dr.old_size;
        last_in_leaf = leaf_element;
        batch_ptr++;
        distinct_batch_elts++;
      } else if (leaf_element > *batch_ptr) {
        batch_ptr++;
      } else {
        if (head_written) {
          EncodeResult er(leaf_element - last_written);
          front_pointer += dr.old_size;
          small_memcpy(back_pointer, er.data, er.size);
          last_written = leaf_element;
          assert(last_written < end_val);
          last_in_leaf = leaf_element;
          back_pointer += er.size;
        } else {
          assert(back_pointer == array);
          head = leaf_element;
          front_pointer += dr.old_size;
          last_written = leaf_element;
          assert(last_written < end_val);
          head_written = true;
          last_in_leaf = leaf_element;
        }
        assert(back_pointer <= front_pointer);
      }
    }

    // write the rest of the original leaf if it exist
    // first write the next element which needs to calculate a new difference
    if (front_pointer < leaf_end) {
      DecodeResult dr(front_pointer);
      if (dr.old_size != 0) {
        T leaf_element = last_in_leaf + dr.difference;
        if (head_written) {
          EncodeResult er(leaf_element - last_written);
          small_memcpy(back_pointer, er.data, er.size);
          back_pointer += er.size;
        } else {
          head = leaf_element;
          head_written = true;
        }
        last_written = leaf_element;
        assert(last_written < end_val);

        front_pointer += dr.old_size;
        assert(back_pointer <= front_pointer);
        // then just copy over the rest
        while (front_pointer < leaf_end) {
          DecodeResult dr2(front_pointer);
          if (dr2.old_size == 0) {
            break;
          }
          small_memcpy(back_pointer, front_pointer, dr2.old_size);
          last_written += dr2.difference;
          assert(last_written < end_val);
          back_pointer += dr2.old_size;
          front_pointer += dr2.old_size;
          assert(back_pointer <= front_pointer);
        }
      }
    }
    int64_t used_bytes = back_pointer - array;
    if (!head_written) {
      head = 0;
    }
    memset(back_pointer, 0, front_pointer - back_pointer);
    if (head != 0) {
      if constexpr (head_in_place) {
        used_bytes += sizeof(T);
      }
    }

    return {batch_ptr, distinct_batch_elts, used_bytes};
  }

  class merged_data {
  public:
    delta_compressed_leaf leaf;
    uint64_t size;

    void free() {
      ::free(reinterpret_cast<uint8_t *>(leaf.array) - sizeof(key_type));
    }
  };

  // Inputs: start of PMA node , number of leaves we want to merge, size of
  // leaf in bytes, number of nonempty bytes in range

  // returns: merged leaf, number of full bytes in leaf
  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static merged_data
  parallel_merge(T *start, uint64_t num_leaves, uint64_t leaf_size,
                 uint64_t leaf_start_index, F index_to_head,
                 [[maybe_unused]] density_array_type density_array) {
    std::vector<uint64_t> bytes_per_leaf(num_leaves);
    std::vector<T> last_per_leaf(num_leaves);
    std::vector<T *> leaf_start(num_leaves);

    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      delta_compressed_leaf l(index_to_head(leaf_start_index + i),
                              start + i * leaf_size / sizeof(T) +
                                  ((head_in_place) ? 1 : 0),
                              leaf_size - ((head_in_place) ? sizeof(T) : 0));

      auto last_size_start = l.last_and_size_in_bytes();
      last_per_leaf[i] = std::get<0>(last_size_start);
      bytes_per_leaf[i] = std::get<1>(last_size_start);
      if (l.head == 0) {
        if (l.get_out_of_place_used_bytes() == 0) {
          // its empty, but we may as well put a valid pointer here
          leaf_start[i] = nullptr;
        } else {
          std::get<0>(index_to_head(leaf_start_index + i)) =
              l.get_first_element_out_of_place();
          assert(l.get_first_element_out_of_place() != 0);
          assert(std::get<0>(index_to_head(leaf_start_index + i)) != 0);
          leaf_start[i] = l.get_out_of_place_pointer() + 1;
        }
      } else {
        leaf_start[i] = l.T_array();
      }
    });

    // check to make sure the leaf we bring the head from has data
    uint64_t start_leaf = std::numeric_limits<uint64_t>::max();
    for (uint64_t i = 0; i < num_leaves; i++) {
      if (leaf_start[i] != nullptr) {
        start_leaf = i;
        break;
      }
    }

    if (start_leaf < std::numeric_limits<uint64_t>::max()) {
      ParallelTools::parallel_for(start_leaf + 1, num_leaves, [&](uint64_t i) {
        T leaf_head = std::get<0>(index_to_head(leaf_start_index + i));
        if (leaf_head != 0) {
          T last = last_per_leaf[i - 1];
          if (last == 0) {
            if (i >= 2) {
              uint64_t j = i - 2;
              while (j < i) {
                last = last_per_leaf[j];
                if (last) {
                  break;
                }
                j -= 1;
              }
            }
          }
          T difference = leaf_head - last;
          assert(difference < leaf_head);
          index_to_head(leaf_start_index + i) = difference;
          bytes_per_leaf[i] += EncodeResult(difference).size;
        }
      });
    }

    uint64_t total_size;
#if PARALLEL == 0
    total_size = prefix_sum_inclusive(bytes_per_leaf);
#else
    if (num_leaves > 1 << 15) {
      total_size = parlay::scan_inclusive_inplace(bytes_per_leaf);
    } else {
      total_size = prefix_sum_inclusive(bytes_per_leaf);
    }
#endif
    uint64_t memory_size =
        ((total_size + (num_leaves * sizeof(T)) + 31) / 32) * 32 + 32;
    // printf("memory_size = %lu\n", memory_size);
    uint8_t *merged_arr = (uint8_t *)(aligned_alloc(32, memory_size));
    // going to place the head 1 before the data
    merged_arr += sizeof(T);

    // first loop not in parallel due to weird compiler behavior
    if (start_leaf < std::numeric_limits<uint64_t>::max()) {
      uint64_t i = start_leaf;
      *(reinterpret_cast<T *>(merged_arr) - 1) =
          std::get<0>(index_to_head(leaf_start_index + start_leaf));
      assert(std::get<0>(index_to_head(leaf_start_index + start_leaf)) != 0);
      memcpy(merged_arr, leaf_start[i], bytes_per_leaf[start_leaf]);
      if (leaf_start[i] !=
          start + i * leaf_size / sizeof(T) + ((head_in_place) ? 1 : 0)) {
        // -1 since in the external leaves we store the head one before the data
        free(leaf_start[i] - 1);
      }
    }

    ParallelTools::parallel_for(start_leaf + 1, num_leaves, [&](uint64_t i) {
      if (std::get<0>(index_to_head(leaf_start_index + i)) != 0) {
        EncodeResult leaf_head(
            std::get<0>(index_to_head(leaf_start_index + i)));
        uint8_t *dest = merged_arr + bytes_per_leaf[i - 1];
        small_memcpy(dest, leaf_head.data, leaf_head.size);
        dest += leaf_head.size;
        uint8_t *source = (uint8_t *)(leaf_start[i]);
        memcpy(dest, source,
               bytes_per_leaf[i] - bytes_per_leaf[i - 1] - leaf_head.size);
        if (leaf_start[i] !=
            start + i * leaf_size / sizeof(T) + ((head_in_place) ? 1 : 0)) {
          // -1 since in the external leaves we store the head one before the
          // data
          free(leaf_start[i] - 1);
        }
      }
    });
    // TODO(wheatman) parallel for
    for (uint64_t i = total_size; i < memory_size - sizeof(T); i++) {
      merged_arr[i] = 0;
    }

    delta_compressed_leaf result(*(reinterpret_cast<T *>(merged_arr) - 1),
                                 (void *)merged_arr, memory_size);

    assert(result.head != 0 || (result.head == 0 && total_size == 0));
    //+sizeof(T) to include the head
    return {result, total_size + sizeof(T)};
  }

  template <bool head_in_place, bool have_densities, bool parallel, typename F,
            typename density_array_type>
  static merged_data merge(element_ptr_type start_, uint64_t num_leaves,
                           uint64_t leaf_size, uint64_t leaf_start_index,
                           F index_to_head,
                           [[maybe_unused]] density_array_type density_array) {
    T *start = start_.get_pointer();

    if constexpr (parallel) {
      if (num_leaves > ParallelTools::getWorkers() * 100U) {
        return parallel_merge<head_in_place, have_densities>(
            start, num_leaves, leaf_size, leaf_start_index, index_to_head,
            density_array);
      }
    }
    uint64_t dest_size = (max_element_size - sizeof(T)) * num_leaves;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * (leaf_size / sizeof(T));
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0 &&
          get_out_of_place_used_bytes(start + src_idx +
                                      ((head_in_place) ? 1 : 0)) != 0) {
        // +(max_element_size - sizeof(T)) to account for any extra space the
        // head might need
        dest_size += get_out_of_place_used_bytes(start + src_idx +
                                                 ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size;
      }
    }

    if constexpr (head_in_place) {
      start += 1;
    }
    // +32 to we don't need to worry about reading off the end
    uint64_t memory_size = ((dest_size + 31) / 32) * 32 + 32;
    T *merged_arr = (T *)(aligned_alloc(32, memory_size));

    uint8_t *dest_byte_position = (uint8_t *)merged_arr;

    uint8_t *src = (uint8_t *)start;
    T prev_elt = 0;
    uint8_t *leaf_start = (uint8_t *)start;

    // deal with first leaf separately
    // copy head uncompressed
    bool done_head = false;
    if (std::get<0>(index_to_head(leaf_start_index)) != 0) {
      done_head = true;
      merged_arr[0] = std::get<0>(index_to_head(leaf_start_index));
      prev_elt = std::get<0>(index_to_head(leaf_start_index));
      dest_byte_position += sizeof(T);

      // copy rest of leaf
      // T current_elem = start[0];
      DecodeResult dr(src);
      while (dr.old_size != 0 && (uint64_t)(src - leaf_start) < leaf_size) {
        // small_memcpy(dest_byte_position, src, dr.old_size);
        // everything has a few extra bytes on the end, so copy a fixed size
        // for simplicity and performance
        memcpy(dest_byte_position, src, max_element_size);
        dest_byte_position += dr.old_size;
        src += dr.old_size;
        prev_elt += dr.difference;
        if ((uint64_t)(src - leaf_start) >= leaf_size) {
          break;
        }
        dr = DecodeResult(src);
      }
    } else if (get_out_of_place_used_bytes(start) != 0) {
      // leaf is in extra storage
      done_head = true;
      memcpy(merged_arr, get_out_of_place_pointer(start),
             get_out_of_place_used_bytes(start));
      dest_byte_position += get_out_of_place_used_bytes(start);
      prev_elt = get_out_of_place_last_written(start);
      free(get_out_of_place_pointer(start)); // release temp storage
    }

    // prev_elt should be the end of this node at this point
    uint64_t leaves_so_far = 1;
    while (leaves_so_far < num_leaves) {
      // deal with head
      leaf_start = (uint8_t *)start + leaves_so_far * leaf_size;
      T *leaf_start_array_pointer = (T *)leaf_start;
      T head = std::get<0>(index_to_head(leaf_start_index + leaves_so_far));
      if (head != 0) {
        ASSERT(head > prev_elt, "head = %lu, prev_elt = %lu\n", (uint64_t)head,
               (uint64_t)prev_elt);
        if (done_head) {
          T diff = head - prev_elt;
          // copy in encoded head with diff from end of previous block
          EncodeResult er(diff);
          // small_memcpy(dest_byte_position, er.data, er.size);
          // everything has a few extra bytes on the end, so copy a fixed size
          // for simplicity and performance
          memcpy(dest_byte_position, er.data, max_element_size);
          dest_byte_position += er.size;
        } else {
          done_head = true;
          dest_byte_position += sizeof(T);
          // we know since we haven't written the head yet that we are at the
          // start
          *merged_arr = head;
        }

        src = leaf_start;
        prev_elt = head;

        // copy body
        DecodeResult dr(src);
        while (dr.old_size != 0 && (uint64_t)(src - leaf_start) < leaf_size) {
          // small_memcpy(dest_byte_position, src, dr.old_size);
          // everything has a few extra bytes on the end, so copy a fixed size
          // for simplicity and performance
          memcpy(dest_byte_position, src, max_element_size);
          dest_byte_position += dr.old_size;
          src += dr.old_size;
          prev_elt += dr.difference;
          if ((uint64_t)(src - leaf_start) + 1 >= leaf_size) {
            break;
          }
          dr = DecodeResult(src);
        }
      } else if (get_out_of_place_used_bytes(leaf_start_array_pointer) != 0) {
        T *extrenal_leaf = get_out_of_place_pointer(leaf_start_array_pointer);
        T external_leaf_head =
            get_first_element_out_of_place(leaf_start_array_pointer);
        EncodeResult er(external_leaf_head - prev_elt);
        small_memcpy(dest_byte_position, er.data, er.size);
        dest_byte_position += er.size;
        memcpy(dest_byte_position, extrenal_leaf + 1,
               get_out_of_place_used_bytes(leaf_start_array_pointer) -
                   sizeof(T));
        dest_byte_position +=
            get_out_of_place_used_bytes(leaf_start_array_pointer) - sizeof(T);
        prev_elt = get_out_of_place_last_written(leaf_start_array_pointer);
        free(extrenal_leaf); // release temp storage
      }
      leaves_so_far++;
    }

    // how many bytes were filled in the dest?
    uint64_t num_full_bytes = dest_byte_position - (uint8_t *)merged_arr;
    assert(num_full_bytes < dest_size);
    std::memset(dest_byte_position, 0, memory_size - num_full_bytes);

    delta_compressed_leaf result(merged_arr[0], (void *)(merged_arr + 1),
                                 memory_size - sizeof(T));

    assert((result.head != 0 || (result.head == 0 && num_full_bytes == 0)));

    // comment back in to debug, also need to comment out the freeing of
    // extra space if (num_leaves >= 2) {
    //   auto parallel_check = parallel_merge<head_in_place,
    //   have_densities>(
    //       start - head_in_place, num_leaves, leaf_size, leaf_start_index,
    //       index_to_head, density_array);
    //   if (num_full_bytes != parallel_check.second) {
    //     printf("%lu != %lu\n", num_full_bytes, parallel_check.second);
    //   }
    //   ASSERT(num_full_bytes == parallel_check.second, "%lu != %lu\n",
    //          num_full_bytes, parallel_check.second);
    //   for (uint64_t i = 0; i < num_full_bytes; i++) {
    //     uint8_t *arr = result.array;
    //     if (arr[i] != parallel_check.first.array[i]) {
    //       printf("%u != %u for i = %lu, len was %lu\n", arr[i],
    //              parallel_check.first.array[i], i, num_full_bytes);
    //       result.print();
    //       parallel_check.first.print();
    //       abort();
    //     }

    //     ASSERT(arr[i] == parallel_check.first.array[i],
    //            "%u != %u for i = %lu, len was %lu\n", arr[i],
    //            parallel_check.first.array[i], i, num_full_bytes);
    //   }
    // }

    return {result, num_full_bytes};
  }

  // input: a merged leaf in delta-compressed format
  // input: number of leaves to split into
  // input: number of occupied bytes in the input leaf
  // input: number of bytes per output leaf
  // input: pointer to the start of the output area to write to (requires that
  // you have num_leaves * num_bytes bytes available here to write to)
  // output: split input leaf into num_leaves leaves, each with
  // num_output_bytes bytes
  template <bool head_in_place, bool store_densities, bool support_rank,
            typename F, typename density_array_type,
            typename rank_tree_array_type>
  void parallel_split(const uint64_t num_leaves,
                      const uint64_t num_occupied_bytes,
                      const uint64_t bytes_per_leaf, T *dest_region,
                      uint64_t leaf_start_index, F index_to_head,
                      density_array_type density_array,
                      [[maybe_unused]] rank_tree_array_type rank_tree_array,
                      [[maybe_unused]] uint64_t total_leaves) {
    std::vector<uint8_t *> start_points(num_leaves + 1);
    start_points[0] = array;
    // - sizeof(T) is becuase num_occupied_bytes counts the head, but its not in
    // the array
    start_points[num_leaves] = array + num_occupied_bytes - sizeof(T);
    uint64_t count_per_leaf = num_occupied_bytes / num_leaves;
    uint64_t extra = num_occupied_bytes % num_leaves;

    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      uint64_t start_guess_index =
          count_per_leaf * i + std::min(i, extra) - max_element_size / 2;
      // we are looking for the end of the previous element
      uint8_t *start_guess = array + start_guess_index - 1;
      if ((*start_guess & 0x80U) == 0) {
        start_points[i] = start_guess + 1;
      }
      for (uint8_t j = 1; j < max_element_size; j++) {
        if ((start_guess[j] & 0x80U) == 0) {
          start_points[i] = start_guess + j + 1;
          break;
        }
      }
      assert(start_points[i] > array);
      assert(start_points[i] <= array + (bytes_per_leaf * num_leaves));
    });
    std::vector<T> difference_accross_leaf(num_leaves);
    std::vector<uint64_t> running_element_count;
    if constexpr (support_rank) {
      running_element_count.resize(num_leaves);
    }
    // first loop not in parallel due to weird compiler behavior
    {
      uint64_t i = 0;
      uint8_t *start = start_points[i];
      uint8_t *end = start_points[i + 1];
      assert(end > start);
      assert(start != end);
      T *dest = dest_region + i * bytes_per_leaf / sizeof(T);
      if constexpr (head_in_place) {
        dest += 1;
      }
      index_to_head(leaf_start_index) = head;
      memcpy(dest, start, end - start);
      if constexpr (store_densities) {
        density_array[leaf_start_index] = end - start;
        if constexpr (head_in_place) {
          if (std::get<0>(index_to_head(leaf_start_index)) != 0) {
            density_array[leaf_start_index] += sizeof(T);
          }
        }
      }
      ASSERT(uint64_t(end - start) + max_element_size < bytes_per_leaf,
             "end - start = %lu, bytes_per_leaf = %lu\n", uint64_t(end - start),
             bytes_per_leaf);
      memset(((uint8_t *)dest) + (end - start), 0,
             bytes_per_leaf - (end - start) -
                 ((head_in_place) ? sizeof(T) : 0));
      delta_compressed_leaf<T> l(index_to_head(leaf_start_index + i), dest,
                                 bytes_per_leaf -
                                     ((head_in_place) ? sizeof(T) : 0));
      if constexpr (support_rank) {
        auto [leaf_last, leaf_count] = l.last_and_count();
        difference_accross_leaf[i] = leaf_last;
        running_element_count[i] = leaf_count;
      } else {
        difference_accross_leaf[i] = l.last();
      }
      assert(difference_accross_leaf[i] != 0);
    }

    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      uint8_t *start = start_points[i];
      uint8_t *end = start_points[i + 1];
      assert(end > start);
      assert(start != end);
      T *dest = dest_region + i * bytes_per_leaf / sizeof(T);
      if constexpr (head_in_place) {
        dest += 1;
      }
      DecodeResult leaf_head(start);
      assert(leaf_head.difference != 0);
      index_to_head(leaf_start_index + i) = leaf_head.difference;
      memcpy(dest, start + leaf_head.old_size,
             (end - start) - leaf_head.old_size);
      if constexpr (store_densities) {
        density_array[leaf_start_index + i] =
            (end - start) - leaf_head.old_size;
        if constexpr (head_in_place) {
          // for head
          if (std::get<0>(index_to_head(leaf_start_index + i)) != 0) {
            density_array[leaf_start_index + i] += sizeof(T);
          }
        }
      }
      assert((end - start) - leaf_head.old_size + max_element_size <
             bytes_per_leaf - ((head_in_place) ? sizeof(T) : 0));
      for (uint8_t *it = ((uint8_t *)dest) + (end - start) - leaf_head.old_size;
           it < ((uint8_t *)dest_region) + (i + 1) * bytes_per_leaf; ++it) {
        *it = 0;
      }

      delta_compressed_leaf<T> l(index_to_head(leaf_start_index + i), dest,
                                 bytes_per_leaf -
                                     ((head_in_place) ? sizeof(T) : 0));
      // l.print();
      if constexpr (support_rank) {
        auto [leaf_last, leaf_count] = l.last_and_count();
        difference_accross_leaf[i] = leaf_last;
        running_element_count[i] = leaf_count;
      } else {
        difference_accross_leaf[i] = l.last();
      }
      assert(difference_accross_leaf[i] != 0);
    });

#if PARALLEL == 0
    prefix_sum_inclusive(difference_accross_leaf);
#else
    if (num_leaves > 1 << 15) {
      parlay::scan_inclusive_inplace(difference_accross_leaf);
    } else {
      prefix_sum_inclusive(difference_accross_leaf);
    }
#endif
    if constexpr (support_rank) {
#if PARALLEL == 0
      prefix_sum_inclusive(running_element_count);
#else
      if (num_leaves > 1 << 15) {
        parlay::scan_inclusive_inplace(running_element_count);
      } else {
        prefix_sum_inclusive(running_element_count);
      }
#endif
      ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
        if (i < nextPowerOf2(total_leaves) - 1 &&
            i < nextPowerOf2(num_leaves) - 1) {
          uint64_t running_element_total = running_element_count[i];
          uint64_t my_e_index = e_index(leaf_start_index + i, total_leaves - 1);
          uint64_t parent_i_plus_1 = (i + 1) & i;
          uint64_t parent_running_element_total = 0;
          if (parent_i_plus_1 > 0) {
            parent_running_element_total =
                running_element_count[parent_i_plus_1 - 1];
          }
          rank_tree_array[my_e_index] =
              running_element_total - parent_running_element_total;
        }
      });
      ParallelTools::parallel_for(
          num_leaves,
          std::min(nextPowerOf2(num_leaves) - 1, nextPowerOf2(total_leaves)),
          [&](uint64_t i) {
            uint64_t running_element_total =
                running_element_count[num_leaves - 1];
            uint64_t my_e_index =
                e_index(leaf_start_index + i, total_leaves - 1);
            uint64_t parent_i_plus_1 = (i + 1) & i;
            uint64_t parent_running_element_total = 0;
            if (parent_i_plus_1 > 0) {
              if (parent_i_plus_1 <= running_element_count.size()) {
                parent_running_element_total =
                    running_element_count[parent_i_plus_1 - 1];
              } else {
                parent_running_element_total =
                    running_element_count[num_leaves - 1];
              }
            }
            if (parent_running_element_total > running_element_total) {
              rank_tree_array[my_e_index] = 0;
            } else {
              rank_tree_array[my_e_index] =
                  running_element_total - parent_running_element_total;
            }
          });
    }
    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      std::get<0>(index_to_head(leaf_start_index + i)) +=
          difference_accross_leaf[i - 1];
      assert(std::get<0>(index_to_head(leaf_start_index + i)) != 0);
    });
  }

  template <bool head_in_place, bool store_densities, bool support_rank,
            bool parallel, bool maintain_offsets, typename F,
            typename density_array_type, typename rank_tree_array_type,
            typename offsets_array_type>
  void split(const uint64_t num_leaves, const uint64_t num_occupied_bytes,
             const uint64_t bytes_per_leaf, element_ptr_type dest_region_,
             uint64_t leaf_start_index, F index_to_head,
             density_array_type density_array,
             [[maybe_unused]] rank_tree_array_type rank_tree_array,
             [[maybe_unused]] uint64_t total_leaves,
             [[maybe_unused]] offsets_array_type offsets_array) {
    static_assert(!maintain_offsets);
    ASSERT(used_size_simple<head_in_place>() ==
               num_occupied_bytes - ((head_in_place) ? 0 : sizeof(T)),
           "used_size_simple() == %lu, num_occupied_bytes - ((head_in_place) ? "
           "0 : sizeof(T)) = %lu\n",
           used_size_simple<head_in_place>(),
           num_occupied_bytes - ((head_in_place) ? 0 : sizeof(T)));
    T *dest_region = dest_region_.get_pointer();

    if (num_leaves == 1) {
      if constexpr (head_in_place) {
        dest_region += 1;
      }
      index_to_head(leaf_start_index) = head;
      memcpy((void *)dest_region, (void *)array, num_occupied_bytes);
      if constexpr (store_densities) {
        density_array[leaf_start_index] = num_occupied_bytes;
        if constexpr (head_in_place) {
          // for head
          if (std::get<0>(index_to_head(leaf_start_index)) != 0) {
            density_array[leaf_start_index] += sizeof(T);
          }
        }
      }
      memset(((uint8_t *)dest_region) + num_occupied_bytes, 0,
             bytes_per_leaf - num_occupied_bytes -
                 ((head_in_place) ? sizeof(T) : 0));
      return;
    }
    if constexpr (parallel) {
      if (num_leaves > ParallelTools::getWorkers() * 100U) {
        return parallel_split<head_in_place, store_densities, support_rank>(
            num_leaves, num_occupied_bytes, bytes_per_leaf, dest_region,
            leaf_start_index, index_to_head, density_array, rank_tree_array,
            total_leaves);
      }
    }
    if constexpr (head_in_place) {
      dest_region += 1;
    }
    assert(num_occupied_bytes / num_leaves <= bytes_per_leaf);

    uint8_t *dest = (uint8_t *)dest_region;
    // get first elt
    uint8_t *src = array;
    T cur_elt = head;
    index_to_head(leaf_start_index) = head;
    uint64_t bytes_read = sizeof(T);
    // do intermediate leaves with heads
    uint64_t running_element_total = 1;
    uint64_t min_e_index_found = std::numeric_limits<uint64_t>::max();
    for (uint64_t leaf_idx = 0; leaf_idx < num_leaves - 1; leaf_idx++) {
      uint64_t bytes_for_leaf =
          (num_occupied_bytes - bytes_read) / (num_leaves - leaf_idx);
      // std::cout << "trying to put about " << bytes_for_leaf << " in leaf"
      //           << std::endl;
      // copy leaf head
      if (leaf_idx > 0) {
        DecodeResult dr(src);
        cur_elt += dr.difference;
        src += dr.old_size;
        bytes_for_leaf -= dr.old_size;
        index_to_head(leaf_start_index + leaf_idx) = cur_elt;
        assert(cur_elt > 0);
        bytes_read += dr.old_size;
        if constexpr (support_rank) {
          running_element_total += 1;
        }
      }
      uint64_t bytes_so_far = 0;
      uint8_t *leaf_start = src; // start of differences in this leaf from src
      while (bytes_so_far < bytes_for_leaf) {
        DecodeResult dr(src);
        assert(dr.old_size > 0);
        // early exit if we would end up with too much data somewhere
        if (bytes_so_far + dr.old_size >= bytes_per_leaf - max_element_size) {
          break;
        }
        src += dr.old_size;
        bytes_so_far += dr.old_size;
        cur_elt += dr.difference;
        bytes_read += dr.old_size;
        if constexpr (support_rank) {
          running_element_total += 1;
        }
      }

      uint64_t num_bytes_filled = src - leaf_start;

      ASSERT(num_bytes_filled < bytes_per_leaf - max_element_size,
             "num_bytes_filled = %lu, bytes_per_leaf = %lu, max_element_size = "
             "%lu\n",
             num_bytes_filled, bytes_per_leaf, max_element_size);
      memcpy(dest, leaf_start, num_bytes_filled);
      if constexpr (store_densities) {
        density_array[leaf_start_index + leaf_idx] = num_bytes_filled;
        if constexpr (head_in_place) {
          // for head
          if (std::get<0>(index_to_head(leaf_start_index + leaf_idx)) != 0) {
            density_array[leaf_start_index + leaf_idx] += sizeof(T);
          }
        }
      }
      if constexpr (support_rank) {
        if (leaf_idx < nextPowerOf2(total_leaves) - 1 &&
            leaf_idx < nextPowerOf2(num_leaves) - 1) {
          uint64_t my_e_index =
              e_index(leaf_start_index + leaf_idx, total_leaves - 1);
          rank_tree_array[my_e_index] =
              running_element_total - rank_tree_array_get_prior_in_range(
                                          leaf_start_index + leaf_idx,
                                          total_leaves - 1, min_e_index_found,
                                          rank_tree_array);
          min_e_index_found = std::min(min_e_index_found, my_e_index);
        }
      }
      memset(dest + num_bytes_filled, 0,
             bytes_per_leaf - num_bytes_filled -
                 ((head_in_place) ? sizeof(T) : 0));

      // jump to start of next leaf
      dest += bytes_per_leaf;
    }

    // handle last leaf
    // do the head
    DecodeResult dr(src);
    assert(dr.difference > 0);
    cur_elt += dr.difference;
    src += dr.old_size;
    bytes_read += dr.old_size;
    index_to_head(leaf_start_index + num_leaves - 1) = cur_elt;
    assert(cur_elt > 0);

    // copy the rest
    uint64_t leftover_bytes = num_occupied_bytes - bytes_read;

    ASSERT(leftover_bytes <= bytes_per_leaf,
           "leftover_bytes = %lu, bytes_per_leaf = %lu\n", leftover_bytes,
           bytes_per_leaf);
    memcpy(dest, src, leftover_bytes);
    if constexpr (store_densities) {
      density_array[leaf_start_index + num_leaves - 1] = leftover_bytes;
      if constexpr (head_in_place) {
        // for head
        if (std::get<0>(index_to_head(leaf_start_index + num_leaves - 1)) !=
            0) {
          density_array[leaf_start_index + num_leaves - 1] += sizeof(T);
        }
      }
    }

    memset(dest + leftover_bytes, 0,
           bytes_per_leaf - leftover_bytes - ((head_in_place) ? sizeof(T) : 0));

    if constexpr (support_rank) {
      if (num_leaves - 1 < nextPowerOf2(total_leaves) - 1 &&
          num_leaves - 1 < nextPowerOf2(num_leaves) - 1) {
        delta_compressed_leaf<T> l(
            std::get<0>(index_to_head(leaf_start_index + num_leaves - 1)), dest,
            bytes_per_leaf - ((head_in_place) ? sizeof(T) : 0));
        running_element_total += l.element_count();
        uint64_t my_e_index =
            e_index(leaf_start_index + num_leaves - 1, total_leaves - 1);
        rank_tree_array[my_e_index] =
            running_element_total - rank_tree_array_get_prior_in_range(
                                        leaf_start_index + num_leaves - 1,
                                        total_leaves - 1, min_e_index_found,
                                        rank_tree_array);
        min_e_index_found = std::min(min_e_index_found, my_e_index);
      }

      for (uint64_t i = num_leaves; i < nextPowerOf2(num_leaves) - 1 &&
                                    i < nextPowerOf2(total_leaves) - 1;
           i++) {
        uint64_t my_e_index = e_index(leaf_start_index + i, total_leaves - 1);
        rank_tree_array[my_e_index] =
            running_element_total - rank_tree_array_get_prior_in_range(
                                        leaf_start_index + i, total_leaves - 1,
                                        min_e_index_found, rank_tree_array);
        min_e_index_found = std::min(min_e_index_found, my_e_index);
      }
    }
  }

  // inserts an element
  // first return value indicates if something was inserted
  // if something was inserted the second value tells you the current size
  template <bool head_in_place, typename ValueUpdate, bool maintain_offsets>
  std::pair<bool, size_t> insert(element_type x_,
                                 [[maybe_unused]] ValueUpdate value_update,
                                 [[maybe_unused]] auto offsets_array) {
    static_assert(!maintain_offsets);
    key_type x = std::get<0>(x_);
    if constexpr (head_in_place) {
      // used_size counts the head, length in bytes does not
      assert(used_size<head_in_place>() <
             length_in_bytes + sizeof(T) - max_element_size);
    } else {
      assert(used_size<head_in_place>() < length_in_bytes - max_element_size);
    }
    if (x == head) {
      return {false, 0};
    }
    if (head == 0) {
      head = x;
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    if (x < head) {
      T temp = head;
      head = x;
      // since we just swapped the head we need to tell find to use the old head
      // when interpreting the bytes
      EncodeResult er(temp - head);
      slide_right(array, er.size);
      small_memcpy(array, er.data, er.size);
      return {true, used_size_with_start<head_in_place>(0)};
    }
    FindResult fr = find<head_in_place>(x);

    if (fr.size != 0) {
      return {false, 0};
    }
    EncodeResult er(fr.difference);
    DecodeResult next_difference(array + fr.loc);

    // we are inserting a new last element and don't need to slide
    if (next_difference.old_size == 0) {
      small_memcpy(array + fr.loc, er.data, er.size);
      return {true, fr.loc + er.size + ((head_in_place) ? sizeof(T) : 0)};
    }

    T old_difference = next_difference.difference;
    T new_difference = old_difference - fr.difference;
    EncodeResult new_er(new_difference);

    size_t slide_size = er.size - (next_difference.old_size - new_er.size);

    // its possible that after adding the new element we don't need to shift
    // anything over since the new and old difference together have the same
    // size as just the old difference
    if (slide_size > 0) {
      slide_right(array + fr.loc + next_difference.old_size, slide_size);
    }
    small_memcpy(array + fr.loc, er.data, er.size);
    small_memcpy(array + fr.loc + er.size, new_er.data, new_er.size);
    return {true, used_size_with_start<head_in_place>(fr.loc + er.size +
                                                      new_er.size)};
  }

  template <bool head_in_place>
  std::pair<bool, size_t> insert_by_rank([[maybe_unused]] T x,
                                         [[maybe_unused]] uint64_t rank) {
    std::cerr << "TODO implement insert_by_rank in compressed leaf\n";
    return {false, 0};
  }

  // removes an element
  // first return value indicates if something was removed
  // if something was removed the second value tells you the current size
  template <bool head_in_place, bool maintain_offsets>
  std::pair<bool, size_t> remove(T x, [[maybe_unused]] auto offsets_array) {
    static_assert(!maintain_offsets);
    if (head == 0 || x < head) {
      return {false, 0};
    }
    if (x == head) {
      DecodeResult dr(array);
      T old_head = head;
      // before there was only a head
      if (dr.old_size == 0) {
        head = 0;
        set_out_of_place_used_bytes(0);
        return {true, 0};
      }
      head = old_head + dr.difference;
      slide_left(array + dr.old_size, dr.old_size);
      return {true, used_size<head_in_place>()};
    }
    FindResult fr = find<head_in_place>(x);

    if (fr.size == 0) {
      return {false, 0};
    }

    DecodeResult dr(array + fr.loc);

    DecodeResult next_difference(array + fr.loc + dr.old_size);
    // we removed the last element
    if (next_difference.old_size == 0) {
      for (int64_t i = 0; i < dr.old_size; i++) {
        array[fr.loc + i] = 0;
      }
      size_t ret = fr.loc;
      if constexpr (head_in_place) {
        ret += sizeof(T);
      }
      return {true, ret};
    }

    T old_difference = next_difference.difference;
    T new_difference = old_difference + dr.difference;
    EncodeResult new_er(new_difference);

    size_t slide_size = dr.old_size - (new_er.size - next_difference.old_size);

    if (slide_size > 0) {
      slide_left(array + fr.loc + new_er.size + slide_size, slide_size);
    }
    small_memcpy(array + fr.loc, new_er.data, new_er.size);

    return {true, used_size_with_start<head_in_place>(fr.loc)};
  }
  template <bool head_in_place> bool contains(T x) const {
    if (x < head) {
      return false;
    }
    if (x == head) {
      return true;
    }
    FindResult fr = find<head_in_place>(x);
    return fr.size != 0;
  }
  template <bool head_in_place> bool debug_contains(T x) {
    if (head == 0) {
      T size_in_bytes = get_out_of_place_temp_size();
      T *ptr = get_out_of_place_pointer();
      auto leaf = delta_compressed_leaf(*ptr, ptr + 1, size_in_bytes);
      return leaf.template contains<true>(x);
    }
    return contains<head_in_place>(x);
  }
#if __BMI2__ == 1 && __AVX512BW__ == 1
  // template <bool head_in_place> uint64_t sum() {
  //   T curr_elem = head;
  //   uint64_t curr_sum = head;
  //   int64_t curr_loc = 0;
  //   int num_masks = length_in_bytes / 8 + head_in_place;
  //   uint64_t masks[num_masks + 1];
  //   masks[num_masks] = 0;
  //   uint8_t *true_array_start = (head_in_place) ? array - sizeof(T) : array;
  //   for (int i = 0; i < num_masks; i++) {
  //     masks[i] =
  //         ~_mm512_movepi8_mask(_mm512_loadu_si512(true_array_start + i *
  //         64));
  //   }
  //   uint64_t mask_remaining = 64;
  //   if constexpr (head_in_place) {
  //     mask_remaining -= sizeof(T);
  //     masks[0] >>= sizeof(T);
  //   }
  //   uint64_t mask = masks[0];
  //   int64_t next_mask_idx = 8;
  //   int num_masks_bytes = num_masks * 8;
  //   uint8_t *mask_bytes = (uint8_t *)masks;
  //   DecodeResult dr(array, mask);
  //   while (dr.difference != 0) {
  //     curr_elem += dr.difference;
  //     assert(contains(curr_elem));
  //     curr_loc += dr.old_size;
  //     curr_sum += curr_elem;
  //     mask_remaining -= dr.old_size;
  //     mask >>= dr.old_size;
  //     if (mask_remaining <= 16 && next_mask_idx < num_masks_bytes) {
  //       uint64_t next_mask =
  //           unaligned_load<uint64_t>(mask_bytes + next_mask_idx);
  //       next_mask <<= mask_remaining;
  //       mask |= next_mask;
  //       mask_remaining += 48;
  //       next_mask_idx += 6;
  //     }
  //     dr = DecodeResult(array + curr_loc, mask);
  //   }
  //   return curr_sum;
  // }
  template <bool head_in_place> uint64_t sum() {
    T curr_elem = head;
    uint64_t curr_sum = head;
    int64_t curr_loc = 0;
    uint8_t *true_array_start = (head_in_place) ? array - sizeof(T) : array;
    uint64_t mask = ~_mm512_movepi8_mask(_mm512_loadu_si512(true_array_start));
    uint64_t mask_remaining = 64;
    if constexpr (head_in_place) {
      mask_remaining -= sizeof(T);
      mask >>= sizeof(T);
    }
    int64_t next_mask_offset = 64;
    DecodeResult dr(array, mask);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      assert(contains<head_in_place>(curr_elem));
      curr_loc += dr.old_size;
      curr_sum += curr_elem;
      mask_remaining -= dr.old_size;
      mask >>= dr.old_size;
      if (mask_remaining <= 32 && next_mask_offset < length_in_bytes) {
        uint64_t next_mask = ~(uint32_t)_mm256_movemask_epi8(_mm256_loadu_si256(
            (__m256i *)(true_array_start + next_mask_offset)));
        next_mask <<= mask_remaining;
        mask |= next_mask;
        mask_remaining += 32;
        next_mask_offset += 32;
      }
      dr = DecodeResult(array + curr_loc, mask);
    }
    return curr_sum;
  }
#else
  template <bool head_in_place = false> uint64_t sum() {
    T curr_elem = head;
    uint64_t curr_sum = head;
    int64_t curr_loc = 0;
    DecodeResult dr(array);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      curr_sum += curr_elem;
      // while ((*(array + curr_loc) & 0x80U) == 0) {
      //   uint8_t a = *(array + curr_loc);
      //   if (!a) {
      //     return curr_sum;
      //   }
      //   curr_elem += a;
      //   curr_sum += curr_elem;
      //   curr_loc++;
      // }
      dr = DecodeResult(array + curr_loc);
    }
    return curr_sum;
  }
#endif

  template <bool no_early_exit, class F> bool map(F f) const {
    T curr_elem = head;
    if constexpr (no_early_exit) {
      f(curr_elem);
    } else {
      if (f(curr_elem)) {
        return true;
      }
    }
    int64_t curr_loc = 0;
    DecodeResult dr(array);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      if constexpr (no_early_exit) {
        f(curr_elem);
      } else {
        if (f(curr_elem)) {
          return true;
        }
      }
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
    }
    return false;
  }

  static std::pair<uint8_t *, T>
  find_loc_and_difference_with_hint(T element, uint8_t *position, T curr_elem) {
    DecodeResult dr(position);
    while (dr.difference != 0) {
      T new_el = curr_elem + dr.difference;
      if (new_el >= element) {
        return {position, curr_elem};
      }
      position += dr.old_size;
      dr = DecodeResult(position);
      curr_elem = new_el;
    }
    return {position, curr_elem};
  }

  std::pair<uint8_t *, T> find_loc_and_difference(T element) {
    T curr_elem = head;
    int64_t curr_loc = 0;
    DecodeResult dr(array);
    while (dr.difference != 0) {
      T new_el = curr_elem + dr.difference;
      if (new_el >= element) {
        // this is wrong if its the head
        return {array + curr_loc, curr_elem};
      }
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
      curr_elem = new_el;
    }
    return {array + curr_loc, curr_elem};
  }

  T last() const {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
    }
    return curr_elem;
  }

  // T last() const {
  //   std::array<uint64_t, max_element_size> parts = {0};
  //   uint64_t part_index = 0;
  //   int64_t array_index = 0;
  //   while (array_index < length_in_bytes) {
  //     // printf("element = %d, part_index = %lu, array_index %ld\n",
  //     //        array[array_index], part_index, array_index);
  //     if (part_index == 0 && array[array_index] == 0) {
  //       // std::cout << "end\n";
  //       break;
  //     }
  //     parts[part_index] += array[array_index] & 0x7F;
  //     if ((array[array_index] & 0x80U) == 0) {
  //       part_index = 0;
  //     } else {
  //       part_index += 1;
  //     }
  //     array_index += 1;
  //   }
  //   T res = head;
  //   uint64_t shift = 0;
  //   for (uint64_t i = 0; i < max_element_size; i++) {
  //     res += parts[i] << shift;
  //     shift += 7;
  //   }
  //   ASSERT(res == debug_last(), "got %lu, expected %lu\n", (uint64_t)res,
  //          (uint64_t)debug_last());
  //   return res;
  // }

  [[nodiscard]] uint64_t element_count() const {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    uint64_t count = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
      count += 1;
    }
    return count;
  }

  std::pair<T, uint64_t> last_and_count() {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    uint64_t count = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
      count += 1;
    }
    return {curr_elem, count};
  }

  std::tuple<T, uint64_t> last_and_size_in_bytes() {
    if (head == 0) {
      if (get_out_of_place_used_bytes() == 0) {
        return {0, 0};
      }
      return {get_out_of_place_last_written(),
              get_out_of_place_used_bytes() - sizeof(T)};
    }
    T curr_elem = 0;
    uint64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
    }
    return {curr_elem, curr_loc};
  }
  template <bool head_in_place> size_t used_size_simple() {
    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_bytes();
        } else {
          return get_out_of_place_used_bytes() - sizeof(T);
        }
      }
      return 0;
    }
    int64_t curr_loc = 0;
    while (unaligned_load<uint16_t>(&array[curr_loc]) != 0) {
      curr_loc += 1;
    }
    if (curr_loc == length_in_bytes) {
      // only possibility is that just the very last byte is empty
      assert(array[curr_loc - 1] == 0);
      if constexpr (head_in_place) {
        curr_loc += sizeof(T);
      }
      return curr_loc - 1;
    }
    if constexpr (head_in_place) {
      curr_loc += sizeof(T);
    }
    return curr_loc;
  }
  // assumes that the head has data, and it is full up until start
  template <bool head_in_place>
  size_t used_size_simple_with_start(int64_t start) {
    int64_t curr_loc = start;
    while (unaligned_load<uint16_t>(&array[curr_loc]) != 0) {
      curr_loc += 1;
    }
    if (curr_loc == length_in_bytes) {
      // only possibility is that just the very last byte is empty
      assert(array[curr_loc - 1] == 0);
      curr_loc -= 1;
    }
    if constexpr (head_in_place) {
      curr_loc += sizeof(T);
    }
    ASSERT((uint64_t)curr_loc == used_size_simple<head_in_place>(),
           "got %lu, expected %lu\n", curr_loc, used_size<head_in_place>());
    return curr_loc;
  }
  template <bool head_in_place> size_t used_size_no_overflow() {
#ifdef __AVX2__
    {
      T curr_elem = head;
      if (curr_elem == 0) {
        return 0;
      }
      uint8_t *true_array = array;
      if constexpr (head_in_place) {
        true_array -= sizeof(T);
      }

      uint32_t data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
          _mm256_load_si256((__m256i *)true_array), _mm256_setzero_si256()));
      size_t curr_loc = 0;
      if constexpr (head_in_place) {
        data >>= sizeof(T);
        curr_loc = sizeof(T);
      }
      // bottom sizeof(T) bytes are the head

      if (uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
          set_bit_mask) {
        size_t ret = curr_loc + std::countr_zero(set_bit_mask);
        ASSERT(ret == used_size_simple<head_in_place>(),
               "ret = %lu, correct = %lu\n", ret,
               used_size_simple<head_in_place>());
        return ret;
      }
      curr_loc = sizeof(__m256i);
      uint32_t last_bit;
      if constexpr (head_in_place) {
        last_bit = data >> (31U - sizeof(T));
      } else {
        last_bit = data >> 31U;
      }
      while (curr_loc < (size_t)length_in_bytes) {
        data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_load_si256((__m256i *)(true_array + curr_loc)),
            _mm256_setzero_si256()));
        if (uint32_t set_bit_mask =
                (data ^ (data >> 1U)) ^ (data | (data >> 1U));
            set_bit_mask) {
          size_t ret;
          if (data == 0xFFFFFFFFUL) {
            ret = curr_loc - last_bit;
          } else {
            ret = curr_loc + std::countr_zero(set_bit_mask);
          }
          ASSERT(ret == used_size_simple<head_in_place>(),
                 "ret = %lu, correct = %lu\n", ret,
                 used_size_simple<head_in_place>());
          return ret;
        }
        last_bit = data >> 31U;
        curr_loc += sizeof(__m256i);
      }
      // only possibility is that just the very last byte is empty
      ASSERT(true_array[curr_loc - 1] == 0, "true_array[curr_loc - 1] = %u\n",
             true_array[curr_loc - 1]);

      return length_in_bytes - 1;
    }
#endif
    return used_size_simple<head_in_place>();
  }

  template <bool head_in_place> size_t used_size() {
    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_bytes();
        } else {
          return get_out_of_place_used_bytes() - sizeof(T);
        }
      }
      return 0;
    }
    return used_size_no_overflow<head_in_place>();
  }

  template <bool head_in_place> size_t used_size_with_start(uint64_t start) {
#ifdef __AVX2__
    {
      if (start <= 32 - sizeof(T)) {
        return used_size<head_in_place>();
      }
      uint8_t *true_array = array;
      if constexpr (head_in_place) {
        true_array -= sizeof(T);
        start += sizeof(T);
      }
      uint64_t aligned_start = start & (~(0x1FU));
      uint32_t data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
          _mm256_load_si256((__m256i *)(true_array + aligned_start)),
          _mm256_setzero_si256()));

      if (uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
          set_bit_mask) {
        size_t ret = aligned_start + std::countr_zero(set_bit_mask);
        ASSERT(ret == used_size_simple<head_in_place>(),
               "ret = %lu, correct = %lu\n", ret,
               used_size_simple<head_in_place>());
        return ret;
      }
      size_t curr_loc = aligned_start;
      uint32_t last_bit = data >> 31U;
      while (curr_loc < (size_t)length_in_bytes) {
        data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_load_si256((__m256i *)(true_array + curr_loc)),
            _mm256_setzero_si256()));

        if (uint32_t set_bit_mask =
                (data ^ (data >> 1U)) ^ (data | (data >> 1U));
            set_bit_mask) {
          size_t ret;
          if (data == 0xFFFFFFFFUL) {
            ret = curr_loc - last_bit;
          } else {
            ret = curr_loc + std::countr_zero(set_bit_mask);
          }
          ASSERT(ret == used_size_simple<head_in_place>(),
                 "ret = %lu, correct = %lu\n", ret,
                 used_size_simple<head_in_place>());
          return ret;
        }
        last_bit = data >> 31U;
        curr_loc += sizeof(__m256i);
      }
      // only possibility is that just the very last byte is empty
      ASSERT(true_array[curr_loc - 1] == 0, "true_array[curr_loc - 1] = %u\n",
             true_array[curr_loc - 1]);
      size_t ret = length_in_bytes - 1;
      if constexpr (head_in_place) {
        ret += sizeof(T);
      }
      ASSERT(ret == used_size_simple<head_in_place>(),
             "got = %lu, correct = %lu\n", ret,
             used_size_simple<head_in_place>());
      return ret;
    }
#endif
    return used_size_simple_with_start<head_in_place>(start);
  }

  uint64_t rank(T e) const {
    if (e <= head) {
      return 0;
    } else {
      uint64_t count = 0;

      T curr_elem = 0;
      int64_t curr_loc = 0;
      DecodeResult dr(head, 0);
      while (dr.difference != 0) {
        curr_elem += dr.difference;
        if (curr_elem >= e) {
          break;
        }
        count += 1;
        curr_loc += dr.old_size;
        dr = DecodeResult(array + curr_loc);
      }
      return count;
    }
  }

  T select(uint64_t rank) const {
    if (rank == 0) {
      return head;
    }
    // something happened so we don't know the exact rank, but we should go to
    // the end
    if (rank >= static_cast<uint64_t>(length_in_bytes)) {
      return last();
    }
    uint64_t count = 0;

    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      if (count == rank) {
        break;
      }
      count += 1;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
    }
    return curr_elem;
  }

  iterator lower_bound(key_type key) {
    auto it = begin();
    for (; it != end(); ++it) {
      if (*it >= key) {
        return it;
      }
    }
    return iterator(0, array);
  }

  iterator lower_bound(key_type key, iterator it) {
    for (; it != end(); ++it) {
      if (*it >= key) {
        return it;
      }
    }
    return iterator(0, array);
  }

  void print(bool external = false) {
    std::cout << "##############LEAF##############" << std::endl;

    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() != 0) {
        if (external) {
          printf("*** EXTERNAL SHOULDNT BE HERE **\n");
          return;
        }
        T size_in_bytes = get_out_of_place_temp_size();
        T *ptr = get_out_of_place_pointer();
        auto leaf = delta_compressed_leaf(*ptr, ptr + 1, size_in_bytes);
        std::cout << "LEAF IN EXTERNAL STORAGE" << std::endl;
        std::cout << "used bytes = " << get_out_of_place_used_bytes()
                  << " last written = " << get_out_of_place_last_written()
                  << " temp size = " << get_out_of_place_temp_size()
                  << std::endl;

        leaf.print(true);
        return;
      }
    }
    uint8_t *p = array;
    std::cout << "head=" << curr_elem << std::endl;
    std::cout << "{ ";
    while (p - array < length_in_bytes && (*p != 0 || *(p + 1) != 0)) {
      std::cout << static_cast<uint32_t>(*p) << ", ";
      p += 1;
    }
    std::cout << " }" << std::endl;
    std::cout << "remaining_bytes {";
    while (p - array < length_in_bytes) {
      std::cout << static_cast<uint32_t>(*p) << ", ";
      p += 1;
    }
    std::cout << " }" << std::endl;
    std::cout << "leaf has: ";
    if (curr_elem != 0) {
      std::cout << curr_elem << ", ";
    }
    int64_t curr_loc = 0;
    while (curr_loc < length_in_bytes) {
      DecodeResult dr(array + curr_loc);
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      std::cout << curr_elem << ", ";
      curr_loc += dr.old_size;
    }
    std::cout << std::endl;
    std::cout << "used bytes = " << curr_loc << " out of " << length_in_bytes
              << std::endl;
  }
};
static_counter split_cnt("leaf split");
static_counter size_cnt("counting the bytes");
template <class T, typename... VTs> class uncompressed_leaf {

public:
  static constexpr size_t max_element_size = sizeof(T);
  static constexpr bool compressed = false;
  using key_type = T;
  static constexpr bool binary = sizeof...(VTs) == 0;

  using element_type =
      typename std::conditional<binary, std::tuple<key_type>,
                                std::tuple<key_type, VTs...>>::type;

  using element_ref_type =
      typename std::conditional<binary, std::tuple<key_type &>,
                                std::tuple<key_type &, VTs &...>>::type;

  using element_const_ref_type = typename std::conditional<
      binary, std::tuple<const key_type &>,
      std::tuple<const key_type &, const VTs &...>>::type;

  using element_ptr_type =
      typename std::conditional<binary, MultiPointer<key_type>,
                                MultiPointer<key_type, VTs...>>::type;

  using value_type = std::tuple<VTs...>;

  using SOA_type = typename std::conditional<binary, SOA<key_type>,
                                             SOA<key_type, VTs...>>::type;
  element_ref_type head;

private:
  template <typename... Ts>
  static void set_element_ref_type_directly(std::tuple<Ts &...> loc,
                                     std::tuple<const Ts &...> data) {
    std::memcpy(&std::get<0>(loc), &std::get<0>(data),
                sizeof(std::get<0>(data)));
    if constexpr (sizeof...(Ts) > 1) {
      set_element_ref_type_directly(leftshift_tuple(loc),
                                    leftshift_tuple(data));
    }
  }

public:
  T head_key() const { return std::get<0>(head); }
  element_ptr_type array;

  static constexpr std::array<uint8_t, SOA_type::get_size_static(1)>
      zero_element_spot = {};

  element_ptr_type zero_element_ptr() const {
    return SOA_type::get_static_ptr((void *)zero_element_spot.data(), 1, 0);
  }

  // just for an optimized compare to end
  class iterator_end {};

  // to scan over the data when it is in valid state
  // does not deal with out of place data
  class iterator {
    element_ptr_type curr_elem_ptr;
    element_ptr_type ptr;

  public:
    iterator(element_ref_type head_, element_ptr_type p)
        : curr_elem_ptr(head_), ptr(p) {}
    iterator(element_ptr_type head_, element_ptr_type p)
        : curr_elem_ptr(head_), ptr(p) {}
    // only for use comparing to end
    bool operator!=([[maybe_unused]] const iterator_end &end) const {
      return std::get<0>(*curr_elem_ptr) != 0;
    }
    bool operator==([[maybe_unused]] const iterator_end &end) const {
      return std::get<0>(*curr_elem_ptr) == 0;
    }

    iterator &operator++() {
      curr_elem_ptr = ptr;
      ptr = ptr + 1;
      return *this;
    }
    bool inc_and_check_end() {
      curr_elem_ptr = ptr;
      ptr = ptr + 1;
      return curr_elem_ptr.template get<0>() == 0;
    }
    auto operator*() const {
      if constexpr (binary) {
        return (const key_type &)curr_elem_ptr.template get<0>();
      } else {
        return (element_const_ref_type)*curr_elem_ptr;
      }
    }

    [[nodiscard]] void *get_pointer() const {
      return (void *)ptr.get_pointer();
    }
  };
  iterator begin() const { return iterator(head, array); }
  iterator_end end() const { return {}; }

private:
  const uint64_t length_in_elements;

  uint64_t find(T x) const {
    // heads are dealt with separately
    /*
        uint64_t count = 0;
        for (uint64_t i = 0; i < length_in_elements; i++) {
          count += (array[i] - 1 < x - 1);
        }
        return count;
    */

    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array.get(i) == 0 || array.get(i) >= x) {
        return i;
      }
    }
    return -1;
  }

  static uint64_t get_out_of_place_used_elements(T *data) {
    uint64_t *long_ptr = (uint64_t *)data;
    uint64_t used_bytes;
    memcpy(&used_bytes, long_ptr, sizeof(uint64_t));
    // not theoretically true, but practially means something is wrong
    ASSERT(used_bytes < 0x1000000000000UL, "bytes = %lu\n", used_bytes);
    return used_bytes;
  }
  static void set_out_of_place_used_elements(T *data, uint64_t bytes) {
    // not theoretically true, but practially means something is wrong
    ASSERT(bytes < 0x1000000000000UL, "bytes = %lu\n", bytes);
    memcpy(data, &bytes, sizeof(uint64_t));
  }

  static void *get_out_of_place_pointer(T *data) {
    uint64_t *long_ptr = (uint64_t *)data;
    void *pointer = nullptr;
    memcpy(&pointer, (void *)(long_ptr + 1), sizeof(void *));
    return pointer;
  }
  static void set_out_of_place_pointer(T *data, void *pointer) {
    uint64_t *long_ptr = (uint64_t *)data;
    memcpy((void *)(long_ptr + 1), &pointer, sizeof(void *));
  }

  static uint64_t get_out_of_place_soa_size(T *data) {
    uint64_t *long_ptr = (uint64_t *)data;
    uint64_t soa_size;
    memcpy(&soa_size, long_ptr + 2, sizeof(uint64_t));
    // not theoretically true, but practially means something is wrong
    ASSERT(soa_size < 0x1000000000000UL, "num elements = %lu\n", soa_size);
    return soa_size;
  }

  static void set_out_of_place_soa_size(T *data, uint64_t size) {
    // not theoretically true, but practially means something is wrong
    ASSERT(size < 0x1000000000000UL, "num elemets = %lu\n", size);
    uint64_t *long_ptr = (uint64_t *)data;
    memcpy(long_ptr + 2, &size, sizeof(uint64_t));
  }

  [[nodiscard]] uint64_t get_out_of_place_used_elements() const {
    return get_out_of_place_used_elements(array.get_pointer());
  }
  void set_out_of_place_used_elements(T bytes) const {
    set_out_of_place_used_elements(array.get_pointer(), bytes);
  }
  [[nodiscard]] void *get_out_of_place_pointer() const {
    return get_out_of_place_pointer(array.get_pointer());
  }
  void set_out_of_place_pointer(void *pointer) const {
    set_out_of_place_pointer(array.get_pointer(), pointer);
  }

  [[nodiscard]] uint64_t get_out_of_place_soa_size() const {
    return get_out_of_place_soa_size(array.get_pointer());
  }
  void set_out_of_place_soa_size(uint64_t size) const {
    set_out_of_place_soa_size(array.get_pointer(), size);
  }

public:
  uncompressed_leaf(T &head_, void *array_, uint64_t length)
      : head(head_), array(static_cast<T *>(array_)),
        length_in_elements(length / sizeof(T)) {
    static_assert(binary);
  }

  uncompressed_leaf(element_ref_type head_, element_ptr_type array_,
                    uint64_t length)
      : head(head_), array(array_), length_in_elements(length / sizeof(T)) {}

  uint64_t rank(T e) const {
    if (e <= head) {
      return 0;
    } else {
      for (uint64_t i = 0; i < length_in_elements; i++) {
        if (array[i] == 0 || array[i] >= e) {
          return i + 1;
        }
      }
      return length_in_elements + 1;
    }
  }

  // returns the element with rank
  // if rank is greater than the number of elements return the last element
  T select(uint64_t rank) const {
    if (rank == 0) {
      return head;
    }
    // since we are no longer considering the head
    rank -= 1;
    // something happened so we don't know the exact rank, but we should go to
    // the end
    if (rank >= length_in_elements || array[rank] == 0) {
      return last();
    }
    return array[rank];
  }

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements merged in, and number of bytes used in this
  // leaf)
  template <bool head_in_place, bool parallel, typename ValueUpdate,
            bool maintain_offsets>
  std::tuple<element_ptr_type, uint64_t, uint64_t>
  merge_into_leaf(element_ptr_type batch_start, T *batch_end, uint64_t end_val,
                  ValueUpdate value_update, auto offsets_array) {
    static_assert(!maintain_offsets);
    static_assert(
        binary ||
            std::is_invocable_v<ValueUpdate, element_ref_type, element_type>,
        "the value update function must take in a reference to the "
        "current value and the new value");
#if DEBUG == 1
    if (!check_increasing_or_zero()) {
      print();
      assert(false);
    }
#endif

    // TODO(wheatman) deal with the case where end_val == max_uint and you still
    // want to add the last element
    // case 1: only one element from the batch goes into the leaf
    if (batch_start.get_pointer() + 1 == batch_end ||
        batch_start.template get<0>(1) >= end_val) {
      auto [inserted, byte_count] =
          insert<head_in_place, ValueUpdate, maintain_offsets>(
              *batch_start, ValueUpdate(), offsets_array);

      return {batch_start + 1, inserted, byte_count};
    }
    // case 2: more than 1 elt from batch
    // two-finger merge into extra space, might overflow leaf
    uint64_t temp_size = length_in_elements;

    // get enough size for the batch
    while (batch_start.get_pointer() + temp_size < batch_end &&
           batch_start.template get<0>(temp_size) <= end_val) {
      temp_size *= 2;
    }
    // get enough size for the leaf
    temp_size += length_in_elements;
    temp_size += 1;

    void *temp_arr = malloc(SOA_type::get_size_static(temp_size));
    element_ptr_type batch_ptr = batch_start;
    element_ptr_type leaf_ptr = array;
    auto temp_ptr = SOA_type::get_static_ptr(temp_arr, temp_size, 0);
    const auto temp_arr_start = temp_ptr;

    uint64_t distinct_batch_elts = 0;
    T *leaf_end = array.get_pointer() + length_in_elements;
    key_type last_written = key_type();

    // merge into temp space
    // everything that needs to go before the head
    bool did_before_head = false;
    if constexpr (parallel && binary) {
      if (temp_size > 100) {
        auto before_head_end =
            std::lower_bound(batch_ptr.get_pointer(), batch_end, head_key());
        size_t number_before_head = before_head_end - batch_ptr.get_pointer();
        size_t num_blocks = std::min(number_before_head / 100,
                                     (size_t)ParallelTools::getWorkers());
        // TODO(wheatman) deal with duplicates in the batch
        if (num_blocks > 1) {
          std::vector<uint64_t> block_sizes(num_blocks + 1);
          size_t per_block_count = number_before_head / num_blocks;
          ParallelTools::parallel_for(0, num_blocks, [&](size_t j) {
            auto start = batch_ptr + per_block_count * j;
            auto end = batch_ptr + per_block_count * (j + 1);
            if (end > before_head_end || j == num_blocks - 1) {
              end = before_head_end;
            }
            if (j == 0) {
              ++start;
              block_sizes[j + 1]++;
            }
            while (start < end) {
              if (start.get(0) != (start - 1).get(0)) {
                block_sizes[j + 1]++;
              }
              ++start;
            }
          });
          for (size_t i = 1; i < num_blocks; i++) {
            block_sizes[i + 1] += block_sizes[i];
          }
          number_before_head = block_sizes[num_blocks];
          assert(number_before_head <= temp_size);
          ParallelTools::parallel_for(0, num_blocks, [&](size_t j) {
            auto start = batch_ptr + per_block_count * j;
            auto end = batch_ptr + per_block_count * (j + 1);
            if (end > before_head_end || j == num_blocks - 1) {
              end = before_head_end;
            }
            size_t i = 0;
            if (j == 0) {
              temp_ptr[0] = *start;
              ++start;
              i++;
            }
            while (start < end) {
              if (start.get(0) != (start - 1).get(0)) {
                if (j == 0) {
                  assert(i < (block_sizes[j + 1]));
                } else {
                  assert(i < (block_sizes[j + 1] - block_sizes[j]));
                }
                temp_ptr[block_sizes[j] + i] = *start;
                i++;
              }
              ++start;
            }
          });
          number_before_head = block_sizes[num_blocks];
          batch_ptr += number_before_head;
          temp_ptr += number_before_head - 1;
          last_written = temp_ptr.get();
          ++temp_ptr;
          distinct_batch_elts += number_before_head;
          did_before_head = true;
        }
      }
    }
    if (!did_before_head) {
      while (batch_ptr.get_pointer() < batch_end &&
             batch_ptr.get() < head_key()) {
        // if duplicates in batch, skip
        // zeros are handled someplace else
        assert(batch_ptr.get() != 0);
        if (batch_ptr.get() == last_written) {
          if constexpr (!binary) {
            value_update((temp_ptr - 1)[0], batch_ptr[0]);
          }
          batch_ptr = batch_ptr + 1;
          continue;
        }
        *temp_ptr = *batch_ptr;
        ++batch_ptr;
        distinct_batch_elts++;
        last_written = temp_ptr.get();
        ++temp_ptr;
      }
    }

    if constexpr (!binary) {
      // if we are not binary merge, then we could have gotten a new value for
      // the head
      if (batch_ptr.get_pointer() < batch_end &&
          batch_ptr.get() == head_key()) {
        value_update(head, batch_ptr[0]);
        ++batch_ptr;
      }
    }

    // deal with the head
    temp_ptr[0] = head;
    last_written = std::get<0>(head);
    ++temp_ptr;

    // the standard merge
    while (batch_ptr.get_pointer() < batch_end && batch_ptr.get() < end_val &&
           leaf_ptr.get_pointer() < leaf_end && leaf_ptr.get() > 0) {
      // if duplicates in batch, skip
      if (batch_ptr.get() == last_written) {
        if constexpr (!binary) {
          value_update((temp_ptr - 1)[0], batch_ptr[0]);
        }
        ++batch_ptr;
        continue;
      }

      // otherwise, do a step of the merge
      if (std::get<0>(leaf_ptr[0]) == std::get<0>(batch_ptr[0])) {
        // if the key was already there first take the element that was already
        // there, then merge in the new element from the batch
        *temp_ptr = *leaf_ptr;
        value_update(*temp_ptr, *batch_ptr);
        ++leaf_ptr;
        ++batch_ptr;
      } else if (std::get<0>(leaf_ptr[0]) > std::get<0>(batch_ptr[0])) {
        *temp_ptr = *batch_ptr;
        ++batch_ptr;
        distinct_batch_elts++;
      } else {
        *temp_ptr = *leaf_ptr;
        ++leaf_ptr;
      }

      last_written = temp_ptr.get();
      ++temp_ptr;
    }

    // write rest of the batch if it exists
    while (batch_ptr.get_pointer() < batch_end && batch_ptr.get() < end_val) {
      if (std::get<0>(batch_ptr[0]) == last_written) {
        if constexpr (!binary) {
          value_update((temp_ptr - 1)[0], batch_ptr[0]);
        }
        ++batch_ptr;
        continue;
      }
      *temp_ptr = *batch_ptr;
      ++batch_ptr;
      distinct_batch_elts++;

      last_written = temp_ptr.get();
      ++temp_ptr;
    }
    // write rest of the leaf it exists
    while (leaf_ptr.get_pointer() < leaf_end && leaf_ptr.get() > 0) {
      *temp_ptr = *leaf_ptr;
      ++leaf_ptr;
      ++temp_ptr;
    }

    uint64_t used_elts = temp_ptr.get_pointer() - temp_arr_start.get_pointer();
    // check if you can fit in the leaf
    if (used_elts <= length_in_elements) {
      head = *temp_arr_start;
      for (uint64_t i = 0; i < (used_elts - 1); i++) {
        array[i] = temp_arr_start[i + 1];
      }
      free(temp_arr);
    } else {                 // special write for when you don't fit
      head = element_type(); // special case head
      assert(used_elts < (uint64_t)std::numeric_limits<T>::max());
      set_out_of_place_used_elements((T)used_elts);
      set_out_of_place_pointer(temp_arr);
      set_out_of_place_soa_size(temp_size);
    }
#if DEBUG == 1
    if (!check_increasing_or_zero()) {
      print();
      assert(false);
    }
#endif

    return {batch_ptr, distinct_batch_elts,
            used_elts * sizeof(T) -
                ((head_in_place) ? 0 : sizeof(T)) /* for the head*/};
  }

  // this differs from above in that it knows how to deal with sentinals and
  // different nodes regions
  template <bool head_in_place, bool parallel, typename ValueUpdate,
            bool maintain_offsets, std::ranges::random_access_range R,
            class element_func>
  std::tuple<int64_t, uint64_t, uint64_t>
  merge_into_leaf_pcsr(R &es_raw, element_func f, key_type this_head_src,
                       key_type next_head_src, key_type next_head_dest,
                       ValueUpdate value_update, auto &offsets_array) {
    // this leaf can't be empty or external
    assert(head_key() != 0);
    assert(array.get(length_in_elements - 1) == 0);

    auto es = std::ranges::transform_view(
        es_raw, [&](const auto &element) { return f(element); });

#if DEBUG == 1
    if (!check_increasing_or_zero<true>()) {
      print();
      assert(false);
    }
#endif
    static constexpr T top_bit = (std::numeric_limits<T>::max() >> 1) + 1;
    static_assert(maintain_offsets);
    static_assert(
        binary ||
            std::is_invocable_v<ValueUpdate, element_ref_type, element_type>,
        "the value update function must take in a reference to the "
        "current value and the new value");
    // make a tuple of src dest from the elements in the batch
    auto b_tuple = [](const auto &element)
        -> std::tuple<key_type, std::make_signed_t<key_type>> {
      // we convert the dest to signed, this means that sentinals, who always
      // have the first bit set will be less than all other elements in the same
      // source
      return {std::get<0>(element), std::get<1>(element)};
    };

    auto mark_inserted = [](auto &element) { std::get<0>(element) |= top_bit; };

    auto b_value = [](const auto &element) {
      if constexpr (binary) {
        return std::get<1>(element);
      } else {
        static_assert(!binary);
        return leftshift_tuple(element);
      }
    };
    // make a tuple for the elements in the leaf
    auto l_tuple = [](key_type src, key_type dest)
        -> std::tuple<key_type, std::make_signed_t<key_type>> {
      return {src, dest};
    };

    if (es.size() == 1 ||
        b_tuple(es[1]) >= l_tuple(next_head_src, next_head_dest)) {
      // if the batch size is 1 we never need to use extra space
      // if we are inserting into the first and the head isn't a sentinal we can
      // just call the standard function, since it will properly shift things to
      // the right
      const auto &elem = es[0];
      if (this_head_src == std::get<0>(elem) &&
          (head_key() < top_bit ||
           head_key() >= (top_bit | (this_head_src + 1)))) {
        auto [inserted, byte_count] =
            insert<head_in_place, ValueUpdate, maintain_offsets>(
                leftshift_tuple(elem), ValueUpdate(), offsets_array);
        if (inserted) {
          mark_inserted(es_raw[0]);
        }
        return {1, inserted, byte_count};
      }
      // else we need to find the start of our fake leaf
      T my_sentinal = top_bit | std::get<0>(elem);
      // this means the head is our sentinal
      if (head_key() == my_sentinal) {
        uncompressed_leaf l(*array, array + 1,
                            (length_in_elements - 1) * sizeof(T));
        auto [inserted, byte_count] =
            l.template insert<true, ValueUpdate, maintain_offsets>(
                leftshift_tuple(elem), ValueUpdate(), offsets_array);
        // added since this function won't count the space used in the real
        // leaf, just the fake one
        if constexpr (head_in_place) {
          byte_count += sizeof(T);
        }
        // std::cout << "insert 2\n";
        if (inserted) {
          mark_inserted(es_raw[0]);
        }
        return {1, inserted, byte_count};
      }
      // we loop through to find the correct sentinal
      size_t offset = std::numeric_limits<size_t>::max();
      for (uint64_t i = 0; i < length_in_elements; i++) {
        if (array.get(i) == my_sentinal) {
          offset = i;
        }
      }
      assert(offset != std::numeric_limits<size_t>::max());
      // offset would be the sentinal
      //  offset+1 is the fake head
      //  offset+2 is the fake data
      uncompressed_leaf l(*(array + 1 + offset), array + 2 + offset,
                          (length_in_elements - (2 + offset)) * sizeof(T));
      auto [inserted, byte_count] =
          l.template insert<true, ValueUpdate, maintain_offsets>(
              leftshift_tuple(elem), ValueUpdate(), offsets_array);
      // added since this function won't count the space used in the real
      // leaf, just the fake one
      byte_count += sizeof(T) * (offset + 1);
      if constexpr (head_in_place) {
        byte_count += sizeof(T);
      }
      // std::cout << "insert 3\n";
      if (inserted) {
        mark_inserted(es_raw[0]);
      }
      return {1, inserted, byte_count};
    }

    // we have more than one element and want to do a proper merge
    // two-finger merge into extra space, might overflow leaf
    uint64_t temp_size = length_in_elements;
    // get enough size for the batch
    while (temp_size < es.size() &&
           b_tuple(es[temp_size]) <= l_tuple(next_head_src, next_head_dest)) {
      temp_size *= 2;
    }
    temp_size += length_in_elements;
    temp_size += 1;

    void *temp_arr = malloc(SOA_type::get_size_static(temp_size));
    element_ptr_type leaf_ptr = array;
    auto temp_ptr = SOA_type::get_static_ptr(temp_arr, temp_size, 0);
    const auto temp_arr_start = temp_ptr;

    uint64_t distinct_batch_elts = 0;
    uint64_t batch_index = 0;
    T *leaf_end = array.get_pointer() + length_in_elements;
    auto last_written = l_tuple(key_type(), key_type());

    // merge into temp space
    // everything that needs to go before the head

    bool did_before_head = false;

    // TODO(wheatman) do the parallel merge, though this case is only needed
    // when you insert a constant fraction of the total elements to a single
    // leaf
    T leaf_src = this_head_src;

    if (!did_before_head) {
      while (batch_index < es.size() &&
             b_tuple(es[batch_index]) < l_tuple(leaf_src, head_key())) {
        // if duplicates in batch, skip
        // zeros are handled someplace else
        assert(std::get<1>(es[batch_index]) != 0);
        if (b_tuple(es[batch_index]) == last_written) {
          if constexpr (!binary) {
            value_update((temp_ptr - 1)[0], b_value(es[batch_index]));
          }
          batch_index += 1;
          continue;
        }
        *temp_ptr = b_value(es[batch_index]);
        mark_inserted(es_raw[batch_index]);
        batch_index += 1;
        distinct_batch_elts++;
        last_written = {leaf_src, temp_ptr.get()};
        ++temp_ptr;
      }
    }

    if constexpr (!binary) {
      // if we are not binary merge, then we could have gotten a new value for
      // the head
      if (batch_index < es.size() &&
          b_tuple(es[batch_index]) == l_tuple(leaf_src, head_key())) {
        value_update(head, b_value(es[batch_index]));
        batch_index += 1;
      }
    }

    // deal with the head
    temp_ptr.set(head);
    last_written = {leaf_src, std::get<0>(head)};
    ++temp_ptr;

    // the standard merge
    while (batch_index < es.size() &&
           b_tuple(es[batch_index]) < l_tuple(next_head_src, next_head_dest) &&
           leaf_ptr.get_pointer() < leaf_end && leaf_ptr.get() > 0) {
      // if duplicates in batch, skip
      if (b_tuple(es[batch_index]) == last_written) {
        if constexpr (!binary) {
          value_update((temp_ptr - 1)[0], b_value(es[batch_index]));
        }
        batch_index += 1;
        continue;
      } else {
      }
      //  if the element the leaf is pointing me at is a sentinal, then update
      //  leaf src and move on, there will never be sentinals in the batch
      T temp_leaf_src = leaf_src;
      if (std::get<0>(leaf_ptr[0]) >= top_bit) {
        temp_leaf_src = std::get<0>(leaf_ptr[0]) ^ top_bit;
      }

      // otherwise, do a step of the merge
      if (l_tuple(temp_leaf_src, std::get<0>(leaf_ptr[0])) ==
          b_tuple(es[batch_index])) {
        assert(std::get<0>(leaf_ptr[0]) < top_bit);
        // if the key was already there first take the element that was already
        // there, then merge in the new element from the batch
        temp_ptr.set(leaf_ptr[0]);
        value_update(*temp_ptr, b_value(es[batch_index]));
        ++leaf_ptr;
        ++batch_index;
      } else if (l_tuple(temp_leaf_src, std::get<0>(leaf_ptr[0])) >
                 b_tuple(es[batch_index])) {
        temp_ptr.set(b_value(es[batch_index]));
        mark_inserted(es_raw[batch_index]);
        ++batch_index;
        distinct_batch_elts++;
      } else {
        temp_ptr.set(leaf_ptr[0]);
        ++leaf_ptr;
        leaf_src = temp_leaf_src;
      }
      last_written = {leaf_src, temp_ptr.get()};
      ++temp_ptr;
    }

    // write rest of the batch if it exists
    while (batch_index < es.size() &&
           b_tuple(es[batch_index]) < l_tuple(next_head_src, next_head_dest)) {
      if (b_tuple(es[batch_index]) == last_written) {
        if constexpr (!binary) {
          value_update((temp_ptr - 1)[0], b_value(es[batch_index]));
        }
        ++batch_index;
        continue;
      }
      temp_ptr.set(b_value(es[batch_index]));
      mark_inserted(es_raw[batch_index]);
      ++batch_index;
      distinct_batch_elts++;

      last_written = {leaf_src, temp_ptr.get()};
      ++temp_ptr;
    }

    // write rest of the leaf it exists
    while (leaf_ptr.get_pointer() < leaf_end && leaf_ptr.get() > 0) {
      temp_ptr.set(leaf_ptr[0]);
      ++leaf_ptr;
      ++temp_ptr;
    }

    uint64_t used_elts = temp_ptr.get_pointer() - temp_arr_start.get_pointer();
    // check if you can fit in the leaf
    if (used_elts <= length_in_elements) {
      set_element_ref_type_directly(head, *temp_arr_start);
      element_move_function(head_key(), &std::get<0>(head), offsets_array);
      for (uint64_t i = 0; i < (used_elts - 1); i++) {
        (array + i).set(temp_arr_start[i + 1]);
        element_move_function(std::get<0>(array[i]),
                              (array + i).template get_pointer<0>(),
                              offsets_array);
      }
      free(temp_arr);
    } else {                  // special write for when you don't fit
      std::get<0>(head) = {}; // special case head
      assert(used_elts < (uint64_t)std::numeric_limits<T>::max());
      set_out_of_place_used_elements((T)used_elts);
      set_out_of_place_pointer(temp_arr);
      set_out_of_place_soa_size(temp_size);
    }
    assert(batch_index <= es.size());

#if DEBUG == 1
    if (!check_increasing_or_zero<true>()) {
      print();
      assert(false);
    }
#endif
    return {batch_index, distinct_batch_elts,
            used_elts * sizeof(T) -
                ((head_in_place) ? 0 : sizeof(T)) /* for the head*/};
  }

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements striped_from, and number of bytes used in
  // this leaf)
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t>
  strip_from_leaf(T *batch_start, T *batch_end, uint64_t end_val) {
    // TODO(wheatman) deal with the case where end_val == max_uint and you still
    // want to add the last element
    // case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      auto [removed, byte_count] =
          remove<head_in_place, false>(*batch_start, nullptr);
      return {batch_start + 1, removed, byte_count};
    }
    // case 2: more than 1 elt from batch
    // two-finger merge into side array for ease
    T *batch_ptr = batch_start;
    element_ptr_type front_pointer = array;
    element_ptr_type back_pointer = array;

    uint64_t distinct_batch_elts = 0;
    T *leaf_end = array.get_pointer() + length_in_elements;

    // everything that needs to go before the head
    while (batch_ptr < batch_end && *batch_ptr < std::get<0>(head)) {
      batch_ptr++;
    }
    bool head_done = false;
    if ((batch_ptr < batch_end && *batch_ptr != head_key()) ||
        batch_ptr == batch_end) {
      head_done = true;
    } else {
      distinct_batch_elts++;
    }

    // merge into temp space
    while (batch_ptr < batch_end && *batch_ptr < end_val &&
           front_pointer.get_pointer() < leaf_end && front_pointer.get() > 0) {

      // otherwise, do a step of the merge
      if (front_pointer.get() == *batch_ptr) {
        ++front_pointer;
        batch_ptr++;
        distinct_batch_elts++;
      } else if (front_pointer.get() > *batch_ptr) {
        batch_ptr++;
      } else {
        if (!head_done) {
          head = *front_pointer;
          head_done = true;
        } else {
          *back_pointer = *front_pointer;
          ++back_pointer;
        }
        ++front_pointer;
      }
    }
    // write rest of the leaf it exists
    while (front_pointer.get_pointer() < leaf_end && front_pointer.get() > 0) {
      if (!head_done) {
        head = *front_pointer;
        head_done = true;
      } else {
        *back_pointer = *front_pointer;
        ++back_pointer;
      }
      ++front_pointer;
    }

    uint64_t used_elts = back_pointer.get_pointer() - array.get_pointer();
    if (!head_done) {
      head = element_type();
      set_out_of_place_used_elements(0);
    }

    // clear the rest of the space
    while (back_pointer.get_pointer() < front_pointer.get_pointer()) {
      if constexpr (!binary) {
        // the key needs to be a simple type that this is not needed for
        back_pointer.zero();
      }
      *back_pointer = element_type();
      ++back_pointer;
    }
    uint64_t used_bytes = used_elts * sizeof(T);
    if (head_key() != 0) {
      if constexpr (head_in_place) {
        used_bytes += sizeof(T);
      }
    }

    return {batch_ptr, distinct_batch_elts, used_bytes};
  }

  // this differs from above in that it knows how to deal with sentinals and
  // different nodes regions
  template <bool head_in_place, bool parallel, bool maintain_offsets,
            std::ranges::random_access_range R, class element_func>
  std::tuple<int64_t, uint64_t, uint64_t>
  strip_from_leaf_pcsr(R &es_raw, element_func f, key_type this_head_src,
                       key_type next_head_src, key_type next_head_dest,
                       auto &offsets_array) {
    // this leaf can't be empty or external
    assert(head_key() != 0);
    assert(array.get(length_in_elements - 1) == 0);

    auto es = std::ranges::transform_view(
        es_raw, [&](const auto &element) { return f(element); });

#if DEBUG == 1
    if (!check_increasing_or_zero<true>()) {
      print();
      assert(false);
    }
#endif
    static constexpr T top_bit = (std::numeric_limits<T>::max() >> 1) + 1;
    static_assert(maintain_offsets);
    // make a tuple of src dest from the elements in the batch
    auto b_tuple = [](const auto &element)
        -> std::tuple<key_type, std::make_signed_t<key_type>> {
      // we convert the dest to signed, this means that sentinals, who always
      // have the first bit set will be less than all other elements in the same
      // source
      return {std::get<0>(element), std::get<1>(element)};
    };

    auto mark_removed = [](auto &element) {
      assert(std::get<0>(element) < top_bit);
      std::get<0>(element) |= top_bit;
    };

    // make a tuple for the elements in the leaf
    auto l_tuple = [](key_type src, key_type dest)
        -> std::tuple<key_type, std::make_signed_t<key_type>> {
      return {src, dest};
    };
    assert(es.size() != 0);
    if (es.size() == 1 ||
        b_tuple(es[1]) >= l_tuple(next_head_src, next_head_dest)) {
      // if the batch size is 1 we never need to use extra space
      // if we are inserting into the first and the head isn't a sentinal we can
      // just call the standard function, since it will properly shift things to
      // the right
      const auto &elem = es[0];
      if (this_head_src == std::get<0>(elem) &&
          (head_key() < top_bit ||
           head_key() >= (top_bit | (this_head_src + 1)))) {
        auto [removed, byte_count] = remove<head_in_place, maintain_offsets>(
            std::get<1>(elem), offsets_array);
        if (removed) {
          mark_removed(es_raw[0]);
        }
        return {1, removed, byte_count};
      }
      // else we need to find the start of our fake leaf
      T my_sentinal = top_bit | std::get<0>(elem);
      // this means the head is our sentinal
      if (head_key() == my_sentinal) {
        uncompressed_leaf l(*array, array + 1,
                            (length_in_elements - 1) * sizeof(T));
        auto [removed, byte_count] = l.template remove<true, maintain_offsets>(
            std::get<1>(elem), offsets_array);
        // added since this function won't count the space used in the real
        // leaf, just the fake one
        if constexpr (head_in_place) {
          byte_count += sizeof(T);
        }
        // std::cout << "insert 2\n";
        if (removed) {
          mark_removed(es_raw[0]);
        }
        return {1, removed, byte_count};
      }
      // we loop through to find the correct sentinal
      size_t offset = std::numeric_limits<size_t>::max();
      for (uint64_t i = 0; i < length_in_elements; i++) {
        if (array.get(i) == my_sentinal) {
          offset = i;
        }
      }
      if (offset == std::numeric_limits<size_t>::max()) {
        return {1, false, 0};
      }
      // offset would be the sentinal
      //  offset+1 is the fake head
      //  offset+2 is the fake data
      uncompressed_leaf l(*(array + 1 + offset), array + 2 + offset,
                          (length_in_elements - (2 + offset)) * sizeof(T));
      auto [removed, byte_count] = l.template remove<true, maintain_offsets>(
          std::get<1>(elem), offsets_array);
      // added since this function won't count the space used in the real
      // leaf, just the fake one
      byte_count += sizeof(T) * (offset + 1);
      if constexpr (head_in_place) {
        byte_count += sizeof(T);
      }
      // std::cout << "insert 3\n";
      if (removed) {
        mark_removed(es_raw[0]);
      }
      return {1, removed, byte_count};
    }

    element_ptr_type front_pointer = array;
    element_ptr_type back_pointer = array;

    uint64_t distinct_batch_elts = 0;
    uint64_t batch_index = 0;
    T *leaf_end = array.get_pointer() + length_in_elements;

    // TODO(wheatman) do the parallel strip, though this case is only needed
    // when you insert a constant fraction of the total elements to a single
    // leaf
    T leaf_src = this_head_src;

    while (batch_index < es.size() &&
           b_tuple(es[batch_index]) < l_tuple(leaf_src, head_key())) {
      batch_index += 1;
    }

    // deal with the head
    bool head_done = false;
    if ((batch_index < es.size() &&
         l_tuple(leaf_src, std::get<0>(head)) != b_tuple(es[batch_index])) ||
        batch_index == es.size()) {
      head_done = true;
    } else {
      mark_removed(es_raw[batch_index]);
      batch_index++;
      distinct_batch_elts++;
    }

    // the standard merge
    while (batch_index < es.size() &&
           b_tuple(es[batch_index]) < l_tuple(next_head_src, next_head_dest) &&
           front_pointer.get_pointer() < leaf_end && front_pointer.get() > 0) {
      //  if the element the leaf is pointing me at is a sentinal, then update
      //  leaf src and move on, there will never be sentinals in the batch
      T temp_leaf_src = leaf_src;
      if (std::get<0>(front_pointer[0]) >= top_bit) {
        temp_leaf_src = std::get<0>(front_pointer[0]) ^ top_bit;
      }

      // otherwise, do a step of the merge
      if (l_tuple(temp_leaf_src, std::get<0>(front_pointer[0])) ==
          b_tuple(es[batch_index])) {
        assert(std::get<0>(front_pointer[0]) < top_bit);
        mark_removed(es_raw[batch_index]);
        ++front_pointer;
        ++batch_index;
        distinct_batch_elts++;
      } else if (l_tuple(temp_leaf_src, std::get<0>(front_pointer[0])) >
                 b_tuple(es[batch_index])) {
        ++batch_index;
      } else {
        if (!head_done) {
          head = *front_pointer;
          element_move_function(head_key(), &std::get<0>(head), offsets_array);
          head_done = true;
        } else {
          *back_pointer = *front_pointer;
          element_move_function(back_pointer.get(),
                                (back_pointer).template get_pointer<0>(),
                                offsets_array);
          ++back_pointer;
        }
        ++front_pointer;
        leaf_src = temp_leaf_src;
      }
    }

    // write rest of the leaf it exists
    while (front_pointer.get_pointer() < leaf_end && front_pointer.get() > 0) {
      if (!head_done) {
        head = *front_pointer;
        element_move_function(head_key(), &std::get<0>(head), offsets_array);
        head_done = true;
      } else {
        *back_pointer = *front_pointer;
        element_move_function(back_pointer.get(),
                              (back_pointer).template get_pointer<0>(),
                              offsets_array);
        ++back_pointer;
      }
      ++front_pointer;
    }

    uint64_t used_elts = back_pointer.get_pointer() - array.get_pointer();
    if (!head_done) {
      head = element_type();
      set_out_of_place_used_elements(0);
    }

    // clear the rest of the space
    while (back_pointer.get_pointer() < front_pointer.get_pointer()) {
      if constexpr (!binary) {
        // the key needs to be a simple type that this is not needed for
        back_pointer.zero();
      }
      *back_pointer = element_type();
      ++back_pointer;
    }
    uint64_t used_bytes = used_elts * sizeof(T);
    if (head_key() != 0) {
      if constexpr (head_in_place) {
        used_bytes += sizeof(T);
      }
    }

    // eat up as much of the batch as we can, so we don't try and remove
    // something else from this leaf

    while (batch_index < es.size() &&
           b_tuple(es[batch_index]) < l_tuple(next_head_src, next_head_dest)) {
      batch_index++;
    }

    assert(batch_index <= es.size());

#if DEBUG == 1
    if (!check_increasing_or_zero<true>()) {
      print();
      assert(false);
    }
#endif
    return {batch_index, distinct_batch_elts, used_bytes};
  }

  class merged_data {
  public:
    uncompressed_leaf leaf;
    uint64_t size;

    void free() {
      ::free(reinterpret_cast<uint8_t *>(leaf.array.get_pointer()) -
             sizeof(key_type));
    }
  };

  // TODO(wheatman) combine with the serial one and just skip the frees
  template <bool head_in_place, typename F>
  static merged_data merge_debug(element_ptr_type array, uint64_t num_leaves,
                                 uint64_t leaf_size_in_bytes,
                                 uint64_t leaf_start_index, F index_to_head) {
    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);

    // giving it an extra leaf_size_in_bytes to ensure we don't write off the
    // end without needed to be exact about what we are writing for some extra
    // performance
    uint64_t dest_size = leaf_size_in_bytes / sizeof(T);
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0) {
        dest_size += get_out_of_place_used_elements(
            array.get_pointer() + src_idx + ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size_in_bytes / sizeof(T);
      }
    }

    // printf("mallocing size %u\n", dest_size);
    uint64_t memory_size = dest_size;
    void *merged_arr_base = malloc(SOA_type::get_size_static(memory_size));
    auto merged_mp = SOA_type::get_static_ptr(merged_arr_base, memory_size, 0);

    uint64_t start = 0;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      element_ptr_type leaf_data_start =
          array + src_idx + ((head_in_place) ? 1 : 0);

      // if reading overflowed leaf, copy in the temporary storage
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0 &&
          get_out_of_place_used_elements(leaf_data_start.get_pointer()) != 0) {
        auto ptr_to_temp = SOA_type::get_static_ptr(
            get_out_of_place_pointer(leaf_data_start.get_pointer()),
            get_out_of_place_soa_size(leaf_data_start.get_pointer()), 0);
        for (uint64_t j = 0;
             j < get_out_of_place_used_elements(leaf_data_start.get_pointer());
             j++) {
          if constexpr (!binary) {
            (merged_mp + start + j).zero();
          }
          (merged_mp + start + j).set(ptr_to_temp[j]);
        }
        start += get_out_of_place_used_elements(leaf_data_start.get_pointer());
      } else { // otherwise, reading regular leaf
        if constexpr (!binary) {
          (merged_mp + start).zero();
        }
        merged_mp[start] = index_to_head(leaf_start_index + i);
        start += (std::get<0>(index_to_head(leaf_start_index + i)) != 0);
        uint64_t local_start = start;
        if constexpr (head_in_place) {
          for (uint64_t j = 0; j < leaf_size - 1; j++) {
            if constexpr (!binary) {
              (merged_mp + local_start + j).zero();
            }
            merged_mp[local_start + j] = array[j + src_idx + 1];
            start += (array.get(j + src_idx + 1) != 0);
          }
        } else {
          for (uint64_t j = 0; j < leaf_size; j++) {
            if constexpr (!binary) {
              (merged_mp + local_start + j).zero();
            }
            merged_mp[local_start + j] = array[j + src_idx];
            start += (array.get(j + src_idx) != 0);
          }
        }
      }
    }

    // fill in the rest of the leaf with 0
    for (uint64_t i = start; i < memory_size; i++) {
      if constexpr (!binary) {
        // the key needs to be a simple type that this is not needed for
        (merged_mp + i).zero();
      }
      merged_mp[i] = element_type();
    }

    uncompressed_leaf result(*merged_mp, merged_mp + 1,
                             memory_size * sizeof(T) - sizeof(T));

    // +1 for head
    return {result, start};
  }

  // Inputs: start of PMA node , number of leaves we want to merge, size of
  // leaf in bytes, number of nonempty bytes in range
  // if we re after an insert that means head being zero means data is stored
  // in an extra storage, if we are after a delete then heads can just be
  // empty from having no data

  // returns: merged leaf, number of full elements in leaf
  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static merged_data parallel_merge(element_ptr_type array, uint64_t num_leaves,
                                    uint64_t leaf_size_in_bytes,
                                    uint64_t leaf_start_index, F index_to_head,
                                    density_array_type density_array) {
#if DEBUG == 1
    auto checker = merge_debug<head_in_place>(
        array, num_leaves, leaf_size_in_bytes, leaf_start_index, index_to_head);

#endif
    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);
    std::vector<uint64_t> elements_per_leaf(num_leaves);
    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      uncompressed_leaf l(index_to_head(leaf_start_index + i),
                          array + i * leaf_size + ((head_in_place) ? 1 : 0),
                          leaf_size_in_bytes -
                              ((head_in_place) ? sizeof(T) : 0));
      if constexpr (have_densities) {
        if (density_array[leaf_start_index + i] ==
            std::numeric_limits<uint16_t>::max()) {
          elements_per_leaf[i] =
              l.template used_size<head_in_place>() / sizeof(T);
        } else {
          elements_per_leaf[i] =
              density_array[leaf_start_index + i] / sizeof(T);
        }
      } else {
        elements_per_leaf[i] =
            l.template used_size<head_in_place>() / sizeof(T);
      }
      // for the head
      if constexpr (!head_in_place) {
        if (std::get<0>(l.head) != 0 ||
            l.get_out_of_place_used_elements() != 0) {
          elements_per_leaf[i] += 1;
        }
      }
    });
    uint64_t total_size;
#if PARALLEL == 0
    total_size = prefix_sum_inclusive(elements_per_leaf);
#else
    if (num_leaves > 1 << 15) {
      total_size = parlay::scan_inclusive_inplace(elements_per_leaf);
    } else {
      total_size = prefix_sum_inclusive(elements_per_leaf);
    }
#endif
    uint64_t memory_size = ((total_size + 31) / 32) * 32;
    void *merged_arr_base = malloc(SOA_type::get_size_static(memory_size));
    auto merged_mp = SOA_type::get_static_ptr(merged_arr_base, memory_size, 0);
    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      uint64_t start = 0;
      if (i > 0) {
        start = elements_per_leaf[i - 1];
      }
      uint64_t src_idx = i * leaf_size;
      uint64_t end = elements_per_leaf[i];
      element_ptr_type leaf_data_start =
          array + src_idx + ((head_in_place) ? 1 : 0);
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0 &&
          get_out_of_place_used_elements(leaf_data_start.get_pointer()) != 0) {
        // get the point from extra storage

        auto ptr_to_temp = SOA_type::get_static_ptr(
            get_out_of_place_pointer(leaf_data_start.get_pointer()),
            get_out_of_place_soa_size(leaf_data_start.get_pointer()), 0);
        for (uint64_t j = 0;
             j < get_out_of_place_used_elements(leaf_data_start.get_pointer());
             j++) {
          if constexpr (!binary) {
            (merged_mp + start + j).zero();
          }
          (merged_mp + start + j).set_and_zero(ptr_to_temp[j]);
        }

        ptr_to_temp.free_first(); // release temp storage
      } else if (std::get<0>(index_to_head(leaf_start_index + i)) != 0) {

        if constexpr (!head_in_place) {
          if constexpr (!binary) {
            (merged_mp + start).zero();
          }
          (merged_mp + start).set_and_zero(index_to_head(leaf_start_index + i));

          start += 1;
        }
        for (uint64_t j = 0; j < end - start; j++) {
          if constexpr (!binary) {
            (merged_mp + start + j).zero();
          }
          (merged_mp + start + j)
              .set_and_zero((leaf_data_start - ((head_in_place) ? 1 : 0))[j]);
        }
      }
    });
    // ParallelTools::parallel_for((total_size / sizeof(T)),
    //                             memory_size / sizeof(T),
    //                             [&](uint64_t i) { merged_arr[i] = 0; });
    uncompressed_leaf result(*merged_mp, merged_mp + 1,
                             memory_size * sizeof(T) - sizeof(T));
#if DEBUG == 1
    bool good_shape = true;
    if (checker.size != total_size) {
      printf("bad total size, got %lu, expected %lu\n", total_size,
             checker.size);
      good_shape = false;
    }
    if (checker.leaf.head != result.head) {
      std::cout << "bad head, got " << result.head << " , expected "
                << checker.leaf.head << "\n";
      good_shape = false;
    }
    for (uint64_t i = 0; i < total_size / sizeof(T) - 1; i++) {
      if (checker.leaf.array[i] != result.array[i]) {
        std::cout << "got bad value in array in position " << i << ", got "
                  << result.array[i] << ", expected " << checker.leaf.array[i]
                  << " check length = " << checker.size << ", got length "
                  << total_size / sizeof(T) << "\n";
        good_shape = false;
      }
    }
    checker.free();
    assert(good_shape);

#endif
    return {result, total_size};
  }

  // TODO(wheatman) make leaf_size_in_bytes number of elements
  template <bool head_in_place, bool have_densities, bool parallel, typename F,
            typename density_array_type>
  static merged_data merge(element_ptr_type array, uint64_t num_leaves,
                           uint64_t leaf_size_in_bytes,
                           uint64_t leaf_start_index, F index_to_head,
                           density_array_type density_array) {
#if PARALLEL == 1
    if constexpr (parallel) {
      if (num_leaves > ParallelTools::getWorkers() * 100U) {
        return parallel_merge<head_in_place, have_densities>(
            array, num_leaves, leaf_size_in_bytes, leaf_start_index,
            index_to_head, density_array);
      }
    }
#endif

    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);

    // giving it an extra leaf_size_in_bytes to ensure we don't write off the
    // end without needed to be eact about what we are writing for some extra
    // performance
    uint64_t dest_size = leaf_size_in_bytes / sizeof(T);
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0) {
        dest_size += get_out_of_place_used_elements(
            array.get_pointer() + src_idx + ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size_in_bytes / sizeof(T);
      }
    }

    uint64_t memory_size = dest_size + num_leaves;
    void *merged_arr_base = malloc(SOA_type::get_size_static(memory_size));
    auto merged_mp = SOA_type::get_static_ptr(merged_arr_base, memory_size, 0);

    uint64_t start = 0;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      element_ptr_type leaf_data_start =
          array + src_idx + ((head_in_place) ? 1 : 0);

      // if reading overflowed leaf, copy in the temporary storage
      if (std::get<0>(index_to_head(leaf_start_index + i)) == 0 &&
          get_out_of_place_used_elements(leaf_data_start.get_pointer()) != 0) {
        // TODO(wheatman) this out of place used elements is not the correct n
        // for the SOA, it the number of elements, not the number of spaces
        auto ptr_to_temp = SOA_type::get_static_ptr(
            get_out_of_place_pointer(leaf_data_start.get_pointer()),
            get_out_of_place_soa_size(leaf_data_start.get_pointer()), 0);
        for (uint64_t j = 0;
             j < get_out_of_place_used_elements(leaf_data_start.get_pointer());
             j++) {
          if constexpr (!binary) {
            (merged_mp + start + j).zero();
          }
          (merged_mp + start + j).set_and_zero(ptr_to_temp[j]);
        }

        ptr_to_temp.free_first(); // release temp storage

        start += get_out_of_place_used_elements(leaf_data_start.get_pointer());
      } else { // otherwise, reading regular leaf
        (merged_mp + start).set_and_zero(index_to_head(leaf_start_index + i));
        start += (std::get<0>(merged_mp[start]) != 0);
        uint64_t local_start = start;
        if constexpr (head_in_place) {
          for (uint64_t j = 0; j < leaf_size - 1; j++) {
            if constexpr (!have_densities) {
              start += (array.get(j + src_idx + 1) != 0);
            }
            if constexpr (!binary) {
              (merged_mp + local_start + j).zero();
            }
            (merged_mp + local_start + j).set_and_zero(array[j + src_idx + 1]);
          }
        } else {
          for (uint64_t j = 0; j < leaf_size; j++) {
            if constexpr (!have_densities) {
              start += (array.get(j + src_idx) != 0);
            }
            (merged_mp + local_start + j).set_and_zero(array[j + src_idx]);
          }
        }
        if constexpr (have_densities) {
          start += density_array[leaf_start_index + i] / sizeof(T);
          if constexpr (head_in_place) {
            start -= (std::get<0>(index_to_head(leaf_start_index + i)) != 0);
          }
        }
      }
    }

    // fill in the rest of the leaf with 0
    for (uint64_t i = start; i < memory_size; i++) {
      if constexpr (!binary) {
        // the key needs to be a simple type that this is not needed for
        (merged_mp + i).zero();
      }
      merged_mp[i] = element_type();
    }

    uncompressed_leaf result(*merged_mp, merged_mp + 1,
                             memory_size * sizeof(T) - sizeof(T));
    ASSERT(start == result.element_count(), "got %lu, expected %lu\n", start,
           result.element_count());
    // assert(result.check_increasing_or_zero());
    return {result, start};
  }

  // input: a merged leaf in delta-compressed format
  // input: number of leaves to split into
  // input: number of elements in the input leaf
  // input: number of elements per output leaf
  // input: pointer to the start of the output area to write to (requires that
  // you have num_leaves * num_bytes bytes available here to write to)
  // output: split input leaf into num_leaves leaves, each with
  // num_output_bytes bytes
  template <bool head_in_place, bool store_densities, bool support_rank,
            bool parallel, bool maintain_offsets, typename F,
            typename density_array_type, typename rank_tree_array_type,
            typename offsets_array_type>
  void split(uint64_t num_leaves, const uint64_t num_elements,
             uint64_t bytes_per_leaf, element_ptr_type dest_region,
             uint64_t leaf_start_index, F index_to_head,
             density_array_type density_array,
             rank_tree_array_type rank_tree_array, uint64_t total_leaves,
             offsets_array_type offsets_array) const {

    split_cnt.add(num_leaves);
    uint64_t elements_per_leaf = bytes_per_leaf / sizeof(T);

    // approx occupied bytes per leaf
    uint64_t count_per_leaf = num_elements / num_leaves;
    uint64_t extra = num_elements % num_leaves;

    assert(count_per_leaf + (extra > 0) <= elements_per_leaf);

    if constexpr (parallel) {
      if (num_leaves > 1UL << 12UL) {
        {
          // first loop not in parallel due to weird compiler behavior
          uint64_t i = 0;
          const uint64_t j3 = 0;
          index_to_head(leaf_start_index) = head;
          if constexpr (maintain_offsets) {
            element_move_function(std::get<0>(index_to_head(leaf_start_index)),
                                  &std::get<0>(index_to_head(leaf_start_index)),
                                  offsets_array);
          }
          uint64_t out = 0;
          // -1 for head
          uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
          if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
            count_for_leaf = 0;
          }
          ASSERT(j3 + count_for_leaf <= length_in_elements,
                 "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n",
                 j3, count_for_leaf, length_in_elements);
          if constexpr (head_in_place) {
            out += 1;
          }
          for (uint64_t k = 0; k < count_for_leaf; k++) {
            if constexpr (!binary) {
              (dest_region + out + k).zero();
            }
            (dest_region + out + k).set(array[j3 + k]);
            if constexpr (maintain_offsets) {
              element_move_function(std::get<0>(dest_region[out + k]),
                                    (dest_region + out + k).get_pointer(),
                                    offsets_array);
            }
          }
          if constexpr (store_densities) {
            density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
            if constexpr (head_in_place) {
              // for head
              if (std::get<0>(index_to_head(leaf_start_index)) != 0) {
                density_array[leaf_start_index + i] += sizeof(T);
              }
            }
          }
          if constexpr (support_rank) {
            if (i < nextPowerOf2(total_leaves) - 1 &&
                i < nextPowerOf2(num_leaves) - 1) {
              uint64_t running_element_total = 1 + count_for_leaf;
              uint64_t my_e_index =
                  e_index(leaf_start_index + i, total_leaves - 1);
              rank_tree_array[my_e_index] = running_element_total;
            }
          }
          if constexpr (head_in_place) {
            for (uint64_t k = 0; k < elements_per_leaf - count_for_leaf - 1;
                 k++) {
              if constexpr (!binary) {
                // the key needs to be a simple type that this is not needed for
                (dest_region + out + count_for_leaf + k).zero();
              }
              dest_region[out + count_for_leaf + k] = element_type();
            }
          } else {
            for (uint64_t k = 0; k < elements_per_leaf - count_for_leaf; k++) {
              if constexpr (!binary) {
                // the key needs to be a simple type that this is not needed for
                (dest_region + out + count_for_leaf + k).zero();
              }
              dest_region[out + count_for_leaf + k] = element_type();
            }
          }
        }
        // seperate loops due to more weird compiler behavior
        ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
          uint64_t j3 = count_per_leaf * i + std::min(i, extra);
          element_ptr_type(index_to_head(leaf_start_index + i))
              .set(array[j3 - 1]);
          if constexpr (maintain_offsets) {
            element_move_function(
                std::get<0>(index_to_head(leaf_start_index + i)),
                &std::get<0>(index_to_head(leaf_start_index + i)),
                offsets_array);
          }
        });
        ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
          const uint64_t j3 = count_per_leaf * i + std::min(i, extra);
          uint64_t out = i * elements_per_leaf;
          // -1 for head
          uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
          if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
            count_for_leaf = 0;
          }
          ASSERT(j3 + count_for_leaf <= length_in_elements,
                 "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n",
                 j3, count_for_leaf, length_in_elements);
          if constexpr (head_in_place) {
            out += 1;
          }
          for (uint64_t k = 0; k < count_for_leaf; k++) {
            if constexpr (!binary) {
              (dest_region + out + k).zero();
            }
            (dest_region + out + k).set(array[j3 + k]);
            if constexpr (maintain_offsets) {
              element_move_function(std::get<0>(dest_region[out + k]),
                                    (dest_region + out + k).get_pointer(),
                                    offsets_array);
            }
          }
          if constexpr (store_densities) {
            density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
            if constexpr (head_in_place) {
              // for head
              if (std::get<0>(index_to_head(leaf_start_index + i)) != 0) {
                density_array[leaf_start_index + i] += sizeof(T);
              }
            }
          }
          if constexpr (support_rank) {
            if (i < nextPowerOf2(total_leaves) - 1 &&
                i < nextPowerOf2(num_leaves) - 1) {
              // j3 is the number of elements I wrote in past leaves
              // second +1 is the head of this leaf, not counted for in
              // count_for_leaf
              uint64_t running_element_total = j3 + 1 + count_for_leaf;
              uint64_t my_e_index =
                  e_index(leaf_start_index + i, total_leaves - 1);
              uint64_t parent_i_plus_1 = (i + 1) & i;
              uint64_t parent_j3 = count_per_leaf * parent_i_plus_1 +
                                   std::min(parent_i_plus_1, extra);
              uint64_t parent_running_element_total = parent_j3;
              if (parent_i_plus_1 == 0) {
                parent_running_element_total = 0;
              }
              rank_tree_array[my_e_index] =
                  running_element_total - parent_running_element_total;
            }
          }
          if constexpr (head_in_place) {
            for (uint64_t k = 0; k < elements_per_leaf - count_for_leaf - 1;
                 k++) {
              if constexpr (!binary) {
                // the key needs to be a simple type that this is not needed for
                (dest_region + out + count_for_leaf + k).zero();
              }
              dest_region[out + count_for_leaf + k] = element_type();
            }
          } else {
            for (uint64_t k = 0; k < elements_per_leaf - count_for_leaf; k++) {
              if constexpr (!binary) {
                // the key needs to be a simple type that this is not needed for
                (dest_region + out + count_for_leaf + k).zero();
              }
              dest_region[out + count_for_leaf + k] = element_type();
            }
          }
        });
        if constexpr (support_rank) {
          ParallelTools::parallel_for(
              num_leaves,
              std::min(nextPowerOf2(num_leaves) - 1,
                       nextPowerOf2(total_leaves)),
              [&](uint64_t i) {
                // j3 is the number of elements I wrote in past leaves
                // first +1 is for the original head, not counted for in j3
                // second +1 is the head of this leaf, not counted for in
                // count_for_leaf
                uint64_t running_element_total = num_elements;
                uint64_t my_e_index =
                    e_index(leaf_start_index + i, total_leaves - 1);
                uint64_t parent_i_plus_1 = (i + 1) & i;
                uint64_t parent_j3 = count_per_leaf * parent_i_plus_1 +
                                     std::min(parent_i_plus_1, extra);
                uint64_t parent_running_element_total = parent_j3;
                if (parent_i_plus_1 == 0) {
                  parent_running_element_total = 0;
                }
                if (parent_running_element_total > running_element_total) {
                  rank_tree_array[my_e_index] = 0;
                } else {
                  rank_tree_array[my_e_index] =
                      running_element_total - parent_running_element_total;
                }
              });
        }
        return;
      }
    }

    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t j3 = count_per_leaf * i + std::min(i, extra) - 1;
      if (i == 0) {
        element_ptr_type(index_to_head(leaf_start_index)).set(head);
        if constexpr (maintain_offsets) {
          element_move_function(std::get<0>(index_to_head(leaf_start_index)),
                                &std::get<0>(index_to_head(leaf_start_index)),
                                offsets_array);
        }
        j3 = 0;
      } else {
        element_ptr_type(index_to_head(leaf_start_index + i)).set(array[j3]);
        if constexpr (maintain_offsets) {
          element_move_function(
              std::get<0>(index_to_head(leaf_start_index + i)),
              &std::get<0>(index_to_head(leaf_start_index + i)), offsets_array);
        }
        j3 += 1;
      }
      uint64_t out = i * elements_per_leaf;
      // -1 for head
      uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
      if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
        count_for_leaf = 0;
      }
      ASSERT(j3 + count_for_leaf <= length_in_elements,
             "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n", j3,
             count_for_leaf, length_in_elements);
      if constexpr (head_in_place) {
        out += 1;
      }
      for (uint64_t k = 0; k < count_for_leaf; k++) {
        (dest_region + out + k).set(array[j3 + k]);
        if constexpr (maintain_offsets) {
          element_move_function(std::get<0>(dest_region[out + k]),
                                (dest_region + out + k).get_pointer(),
                                offsets_array);
        }
      }
      if constexpr (store_densities) {
        density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
        if constexpr (head_in_place) {
          // for head
          if (std::get<0>(index_to_head(leaf_start_index + i)) != 0) {
            density_array[leaf_start_index + i] += sizeof(T);
          }
        }
      }
      if constexpr (support_rank) {
        if (i < nextPowerOf2(total_leaves) - 1 &&
            i < nextPowerOf2(num_leaves) - 1) {
          uint64_t running_element_total = j3 + 1 + count_for_leaf;
          uint64_t my_e_index = e_index(leaf_start_index + i, total_leaves - 1);
          uint64_t parent_i_plus_1 = (i + 1) & i;
          uint64_t parent_j3 = count_per_leaf * parent_i_plus_1 +
                               std::min(parent_i_plus_1, extra);
          uint64_t parent_running_element_total = parent_j3;
          if (parent_i_plus_1 == 0) {
            parent_running_element_total = 0;
          }
          rank_tree_array[my_e_index] =
              running_element_total - parent_running_element_total;
        }
      }
      if constexpr (head_in_place) {
        for (uint64_t k = 0; k < elements_per_leaf - count_for_leaf - 1; k++) {
          if constexpr (!binary) {
            // the key needs to be a simple type that this is not needed for
            (dest_region + out + count_for_leaf + k).zero();
          }
          dest_region[out + count_for_leaf + k] = element_type();
        }

      } else {
        for (uint64_t k = count_for_leaf; k < elements_per_leaf; k++) {
          if constexpr (!binary) {
            // the key needs to be a simple type that this is not needed for
            (dest_region + out + k).zero();
          }
          dest_region[out + k] = element_type();
        }
      }
    }

    if constexpr (support_rank) {
      for (uint64_t i = num_leaves; i < nextPowerOf2(num_leaves) - 1 &&
                                    i < nextPowerOf2(total_leaves) - 1;
           i++) {
        uint64_t running_element_total = num_elements;
        uint64_t my_e_index = e_index(leaf_start_index + i, total_leaves - 1);
        uint64_t parent_i_plus_1 = (i + 1) & i;
        uint64_t parent_j3 =
            count_per_leaf * parent_i_plus_1 + std::min(parent_i_plus_1, extra);
        uint64_t parent_running_element_total = parent_j3;
        if (parent_i_plus_1 == 0) {
          parent_running_element_total = 0;
        }
        if (parent_running_element_total > running_element_total) {
          rank_tree_array[my_e_index] = 0;
        } else {
          rank_tree_array[my_e_index] =
              running_element_total - parent_running_element_total;
        }
      }
    }
  }

  template <typename... Ts, std::size_t... I>
  void tuple_bit_swap_impl(std::tuple<Ts &...> a, std::tuple<Ts...> &b,
                           std::index_sequence<I...>) {
    (std::swap(std::get<I>(a), std::get<I>(b)), ...);
  }

  void tuple_bit_swap(element_ref_type a, element_type &b) {
    tuple_bit_swap_impl(a, b, std::make_index_sequence<1 + sizeof...(VTs)>{});
  }

  // inserts an element
  // first return value indicates if something was inserted
  // if something was inserted the second value tells you the current size
  template <bool head_in_place, typename ValueUpdate, bool maintain_offsets>
  std::pair<bool, size_t> insert(element_type x, ValueUpdate value_update,
                                 auto offsets_array) {
    static_assert(
        binary ||
            std::is_invocable_v<ValueUpdate, element_ref_type, element_type &>,
        "the value update function must take in a reference to the current "
        "value and the new value by reference");
#if DEBUG == 1
    if (!check_increasing_or_zero<maintain_offsets>()) {
      print();
      assert(false);
    }
#endif
    assert(array.get(length_in_elements - 1) == 0);
    assert(std::get<0>(x) != 0);
    if (std::get<0>(x) == head_key()) {
      if constexpr (!binary) {
        value_update(head, x);
      }
      return {false, 0};
    }
    bool new_head = false;
    if (std::get<0>(x) < head_key()) {
      tuple_bit_swap(head, x);
      new_head = true;
    }
    if (head_key() == 0) {
      head = std::move(x);
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    uint64_t fr = 0;
    if (!new_head) {
      fr = find(std::get<0>(x));
    }
    if (array.get(fr) == std::get<0>(x)) {
      if constexpr (!binary) {
        value_update(array[fr], x);
      }
      return {false, 0};
    }
    size_t num_elements = fr + 1;
    // for (uint64_t i = length_in_elements - 1; i >= fr + 1; i--) {
    //   num_elements += (array[i - 1] != 0);
    //   array[i] = array[i - 1];
    // }
    // for (uint64_t i = fr; i < length_in_elements - 1; i++)
    for (uint64_t i = length_in_elements - 1; i > fr; i--) {
      (array + i).set_and_zero(array[i - 1]);
      if constexpr (maintain_offsets) {
        element_move_function(std::get<0>(array[i]), &std::get<0>(array[i]),
                              offsets_array);
      }
      num_elements += (std::get<0>(array[i]) != 0);
    }
    if constexpr (!binary) {
      (array + fr).zero();
    }
    array[fr] = std::move(x);
    if constexpr (maintain_offsets) {
      element_move_function(std::get<0>(array[fr]), &std::get<0>(array[fr]),
                            offsets_array);
    }
#if DEBUG == 1
    if (!check_increasing_or_zero<maintain_offsets>()) {
      print();
      assert(false);
    }
#endif
    return {true, num_elements * sizeof(T) + ((head_in_place) ? sizeof(T) : 0)};
  }

  template <bool head_in_place>
  std::pair<bool, size_t> insert_by_rank(element_type x, uint64_t rank) {
    assert(array[length_in_elements - 1] == 0);
    if (head_key() == 0) {
      // if the leaf is empty
      head = x;
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    if (rank == 0) {
      element_type temp = head;
      head = x;
      x = temp;
      rank = 1;
    }
    // something happened so we don't know the exact rank, but we should go to
    // the end
    if (rank == std::numeric_limits<uint64_t>::max()) {
      size_t num_elements = 0;
      for (uint64_t i = 0; i < length_in_elements; i++) {
        num_elements += (array.get(i) != 0);
      }
      array[num_elements] = x;
      return {true, (num_elements + 1) * sizeof(T) +
                        ((head_in_place) ? sizeof(T) : 0)};
    }

    uint64_t fr = rank - 1;
    size_t num_elements = fr + 1;
    for (uint64_t i = length_in_elements - 1; i >= fr + 1; i--) {
      num_elements += (array.get(i - 1) != 0);
      array[i] = array[i - 1];
    }
    array[fr] = x;
    return {true, num_elements * sizeof(T) + ((head_in_place) ? sizeof(T) : 0)};
  }

  template <bool head_in_place>
  std::pair<bool, size_t> update_by_rank(T x, uint64_t rank) {
    assert(array[length_in_elements - 1] == 0);
    if (head == 0) {
      // if the leaf is empty
      head = x;
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    if (rank == 0) {
      head = x;
      return {false, 0};
    }
    // something happened so we don't know the exact rank, but we should go to
    // the end
    if (rank == std::numeric_limits<uint64_t>::max()) {
      size_t num_elements = 0;
      for (uint64_t i = 0; i < length_in_elements; i++) {
        num_elements += (array[i] != 0);
      }
      array[num_elements] = x;
      return {true, (num_elements + 1) * sizeof(T) +
                        ((head_in_place) ? sizeof(T) : 0)};
    }
    array[rank - 1] = x;
    return {false, 0};
  }

  // removes an element
  // first return value indicates if something was removed
  // if something was removed the second value tells you the current size
  template <bool head_in_place, bool maintain_offsets>
  std::pair<bool, size_t> remove(T x, auto offsets_array) {
    if (head_key() == 0 || x < head_key()) {
      return {false, 0};
    }
    if (x == head_key()) {
      head = array[0];
      if constexpr (maintain_offsets) {
        element_move_function(std::get<0>(head), &std::get<0>(head),
                              offsets_array);
      }
      x = array.get();
      if (x == 0) {
        return {true, 0};
      }
    }
    uint64_t fr = find(x);
    if (array.get(fr) != x) {
      return {false, 0};
    }
    for (uint64_t i = fr; i < length_in_elements - 1; i++) {
      array[i] = array[i + 1];
      if constexpr (maintain_offsets) {
        element_move_function(std::get<0>(array[i]), &std::get<0>(array[i]),
                              offsets_array);
      }
    }
    if constexpr (!binary) {
      // the key needs to be a simple type that this is not needed for
      (array + length_in_elements - 1).zero();
    }
    array[length_in_elements - 1] = element_type();
    size_t num_elements = fr;
    for (uint64_t i = fr; i < length_in_elements; i++) {
      if (array.get(i) == 0) {
        break;
      }
      num_elements += 1;
    }
    return {true, num_elements * sizeof(T) + ((head_in_place) ? sizeof(T) : 0)};
  }

  template <bool unused = false> bool contains(key_type x) const {
    if (x < head_key()) {
      return false;
    }
    if (x == head_key()) {
      return true;
    }
    uint64_t fr = find(x);
    return array.get(fr) == x;
  }

  template <bool unused = false> value_type value(key_type x) const {
    if (x < head_key()) {
      return {};
    }
    if (x == head_key()) {
      return leftshift_tuple(head);
    }
    uint64_t fr = find(x);
    return leftshift_tuple(array[fr]);
  }

  template <bool unused = false> bool debug_contains(T x) const {
    if (head_key() == 0) {
      T size_in_bytes = get_out_of_place_used_elements() * sizeof(T);
      element_ptr_type ptr = SOA_type::get_static_ptr(
          get_out_of_place_pointer(), get_out_of_place_soa_size(), 0);
      auto leaf = uncompressed_leaf(*ptr, ptr + 1, size_in_bytes);
      return leaf.contains(x);
    }
    return contains(x);
  }

  template <bool head_in_place = false> [[nodiscard]] uint64_t sum() const {
    if constexpr (head_in_place) {
      uint64_t curr_sum = 0;
      curr_sum += array.get(-1);
      for (uint64_t i = 0; i < length_in_elements; i++) {
        curr_sum += array.get(i);
      }
      return curr_sum;
    } else {
      uint64_t curr_sum = 0;
      for (uint64_t i = 0; i < length_in_elements; i++) {
        curr_sum += array.get(i);
      }
      return head_key() + curr_sum;
    }
  }

  template <bool no_early_exit, class F> bool map(F f) const {
    auto process = [](const auto &elem, F f) {
      if constexpr (binary) {
        return f(std::get<0>(elem));
      } else {
        return f(elem);
      }
    };
    if (std::get<0>(head) == 0) {
      return false;
    }
    if constexpr (no_early_exit) {
      process(head, f);
    } else {
      if (process(head, f)) {
        return true;
      }
    }
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array.get(i) == 0) {
        return false;
      }
      if constexpr (no_early_exit) {
        process(array[i], f);
      } else {
        if (process(array[i], f)) {
          return true;
        }
      }
    }
    return false;
  }
#ifdef __AVX512F__

  template <bool head_in_place = false> [[nodiscard]] __m512i sum512() const {
    static_assert(
        std::is_same_v<T, uint64_t>,
        "This vectorized code has only been implemented for the 64 bit case");
    assert(head_in_place ? (length_in_elements + 1) % 8 == 0
                         : length_in_elements % 8 == 0);
    key_type *arr = array.get_pointer();
    __m512i a;
    if constexpr (head_in_place) {
      a = _mm512_loadu_epi64(arr - 1);
    } else {
      a = _mm512_maskz_loadu_epi64(0x7F, arr);
      a = _mm512_mask_set1_epi64(a, 0x80, head_key());
    }
    int iters = ((length_in_elements + 1) / 8);
    for (int i = 1; i < iters; i++) {
      a = _mm512_add_epi64(a, _mm512_loadu_epi64(arr + i * 8 - 1));
    }
    return a;
  }
#endif
#ifdef __AVX2__
  template <bool head_in_place = false>
  [[nodiscard]] __m256i sum32_256() const {
    static_assert(
        std::is_same_v<T, uint64_t>,
        "This vectorized code has only been implemented for the 64 bit case");
    assert(head_in_place ? length_in_elements == 31 : length_in_elements == 32);
    __m256i a;
    key_type *arr = array.get_pointer();
    if constexpr (head_in_place) {
      a = _mm256_loadu_epi64(arr - 1);
    } else {
      a = _mm256_maskz_loadu_epi64(0x7, arr);
      a = _mm256_insert_epi64(a, head, 3);
    }
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 3));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 7));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 11));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 15));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 19));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 23));
    a = _mm256_add_epi64(a, _mm256_loadu_epi64(arr + 27));
    return a;
  }
#endif
  key_type last() const {
    element_type last_elem = head;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array.get(i) == 0) {
        break;
      }
      last_elem = array[i];
    }
    return std::get<0>(last_elem);
  }

  template <bool head_in_place>
  [[nodiscard]] size_t used_size_no_overflow() const {
    size_cnt.add(1);
    size_t num_elements = 0;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      num_elements += (array.get(i) != 0);
    }
    if constexpr (head_in_place) {
      // for the head
      num_elements += 1;
    }
    return num_elements * sizeof(T);
  }

  // for use in rank counts, doesn't consider overflow
  [[nodiscard]] size_t element_count() const {
    size_t num_elements = 0;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      num_elements += (array.get(i) != 0);
    }
    // for the head
    num_elements += 1;
    return num_elements;
  }

  template <bool head_in_place> [[nodiscard]] size_t used_size() const {
    // if using temp elsewhere
    if (head_key() == 0) {
      if (get_out_of_place_used_elements() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_elements() * sizeof(T);
        } else {
          return get_out_of_place_used_elements() * sizeof(T) - sizeof(T);
        }
      }
      return 0;
    }
    // otherwise not
    return used_size_no_overflow<head_in_place>();
  }

  void print(bool external = false) const {
    std::cout << "##############LEAF##############" << std::endl;
    std::cout << "length_in_elements = " << length_in_elements << std::endl;

    if (head_key() == 0 && get_out_of_place_used_elements() > 0) {
      if (external) {
        printf("*** EXTERNAL SHOULDNT BE HERE **\n");
        return;
      }
      uint64_t size_in_bytes = get_out_of_place_used_elements() * sizeof(T);
      void *ptr = get_out_of_place_pointer();
      auto m_ptr =
          SOA_type::get_static_ptr(ptr, get_out_of_place_soa_size(), 0);
      auto leaf =
          uncompressed_leaf(*m_ptr, m_ptr + 1, size_in_bytes - sizeof(T));
      std::cout << "LEAF IN EXTERNAL STORAGE" << std::endl;
      leaf.print(true);
      return;
    }
    std::cout << "head=" << head << std::endl;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array.get(i) == 0) {
        std::cout << " - ";
      } else {
        std::cout << array[i] << ", ";
      }
    }
    std::cout << std::endl;
  }

  iterator lower_bound(key_type key) {
    auto it = begin();
    for (; it != end(); ++it) {
      if (*it >= key) {
        return it;
      }
    }
    return iterator(zero_element_ptr(), array);
  }

  iterator lower_bound(key_type key, iterator it) {
    for (; it != end(); ++it) {
      if (*it >= key) {
        return it;
      }
    }
    return iterator(zero_element_ptr(), array);
  }

  template <bool pcsr = false>
  [[nodiscard]] bool check_increasing_or_zero(bool external = false) const {
    static constexpr T top_bit = (std::numeric_limits<T>::max() >> 1) + 1;
    // if using temp elsewhere
    if (head_key() == 0) {
      if (get_out_of_place_used_elements() > 0) {
        if (external) {
          printf("*** EXTERNAL SHOULDNT BE HERE **\n");
          return false;
        }
        uint64_t size_in_bytes = get_out_of_place_used_elements() * sizeof(T);
        void *ptr = get_out_of_place_pointer();
        assert(ptr != nullptr);
        element_ptr_type mptr =
            SOA_type::get_static_ptr(ptr, get_out_of_place_soa_size(), 0);
        auto leaf =
            uncompressed_leaf(*mptr, mptr + 1, size_in_bytes - sizeof(T));
        return leaf.template check_increasing_or_zero<pcsr>(true);
      }
      return true;
    }

    key_type check = std::get<0>(head);
    key_type last_sentinal = 0;
    if constexpr (pcsr) {
      if (std::get<0>(head) >= top_bit) {
        last_sentinal = std::get<0>(head);
        check = 0;
      }
    }
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if constexpr (pcsr) {
        if (array.get(i) >= top_bit) {
          if (array.get(i) <= last_sentinal) {
            std::cout << "bad sentinal in position " << i << std::endl;
            std::cout << array[i] << " <= " << last_sentinal << std::endl;
            return false;
          }
          last_sentinal = array.get(i);
          check = 0;
          continue;
        }
      }
      if (array.get(i) == 0) {
        continue;
      }
      if (std::get<0>(array[i]) <= check) {
        std::cout << "bad in position " << i << std::endl;
        std::cout << array[i] << " <= " << check << std::endl;
        return false;
      }
      check = std::get<0>(array[i]);
    }
    // otherwise not
    return true;
  }
};
