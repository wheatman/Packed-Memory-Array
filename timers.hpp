#pragma once

#include "helpers.h"
#include <cstdint>
#include <iostream> // cout
#include <string>
#include <sys/time.h>

// static objects have to exist, so if we want static timers they can't be
// statics when they are supposed to be optimized away
#if ENABLE_TRACE_TIMER == 1
#define static_timer static timer
#else
#define static_timer timer
#endif

class timer {
#if ENABLE_TRACE_TIMER == 1
  uint64_t starting_time = 0;
  uint64_t elapsed_time = 0;
  std::string name;
  static constexpr bool csv_printing = true;
#endif

#if CYCLE_TIMER == 1
  static inline uint64_t get_time() { return __rdtsc(); }
#else
  static inline uint64_t get_time() { return get_usecs(); }
#endif

public:
#if ENABLE_TRACE_TIMER == 1
  timer(std::string n) : name(std::move(n)) {}
#else
  template <typename T> timer([[maybe_unused]] T n) {}
#endif

  inline void start() {
#if ENABLE_TRACE_TIMER == 1
    starting_time = get_time();
#endif
  }

  inline void stop() {
#if ENABLE_TRACE_TIMER == 1
    uint64_t end_time = get_time();
    elapsed_time += (end_time - starting_time);
#endif
  }
  inline void report() {
#if ENABLE_TRACE_TIMER == 1
    if (csv_printing) {
#if CYCLE_TIMER == 1
      std::cout << name << ", " << elapsed_time << ", cycles" << std::endl;
#else
      std::cout << name << ", " << elapsed_time << ", micro secs" << std::endl;
#endif
    } else {
#if CYCLE_TIMER == 1
      std::cout << name << ": " << elapsed_time << " [cycles]" << std::endl;
#else
      std::cout << name << ": " << elapsed_time << " [micro secs]" << std::endl;
#endif
    }
#endif
  }

  ~timer() {
#if ENABLE_TRACE_TIMER == 1
    report();
#endif
  }
};

#if ENABLE_TRACE_COUNTER == 1
#define static_counter static counter
#else
#define static_counter counter
#endif

class counter {
#if ENABLE_TRACE_COUNTER == 1
  uint64_t count = 0;
  std::string name;
#endif

public:
#if ENABLE_TRACE_COUNTER == 1
  counter(std::string n) : name(std::move(n)) {}
#else
  template <typename T> counter([[maybe_unused]] T n) {}
#endif

  inline void add([[maybe_unused]] uint64_t x) {
#if ENABLE_TRACE_COUNTER == 1
    count += x;
#endif
  }
  inline void reset() {
#if ENABLE_TRACE_COUNTER == 1
    count = 0;
#endif
  }
  inline void report() {
#if ENABLE_TRACE_COUNTER == 1
    std::cout << name << ", " << count << "\n";
#endif
  }
  ~counter() { report(); }
};
