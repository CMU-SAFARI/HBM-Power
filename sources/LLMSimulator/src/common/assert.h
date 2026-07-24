#pragma once
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <string>

// always fail
inline void fail(const std::string &msg) {
  std::cerr << "ERROR: " << msg << std::endl << std::flush;
  std::cout << std::filesystem::current_path() << std::endl;
  std::exit(EXIT_FAILURE);
}

inline void assertTrue(bool expr, const std::string &msg) {
  if (!expr) {
    fail(msg);
  }
}

inline void assertFalse(bool expr, const std::string &msg) {
  assertTrue(!expr, msg);
}

// Only produce warnings

inline void warn(const std::string &msg) {
  std::clog << "WARN: " << msg << std::endl;
}

inline void checkTrue(bool expr, const std::string &msg) {
  if (!expr) {
    warn(msg);
  }
}

inline void checkFalse(bool expr, const std::string &msg) {
  checkTrue(!expr, msg);
}

inline void notYetImplemented(const std::string &name) {
  warn(name + " not yet implemented");
}
