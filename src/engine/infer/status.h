#pragma once

#include <string>
#include <utility>

namespace engine {

class Status {
 public:
  Status() = default;

  static Status Ok() { return Status(); }

  static Status Error(std::string message) {
    return Status(false, std::move(message));
  }

  bool ok() const { return ok_; }
  const std::string& message() const { return message_; }

 private:
  Status(bool ok, std::string message)
      : ok_(ok), message_(std::move(message)) {}

  bool ok_ = true;
  std::string message_;
};

}  // namespace engine
