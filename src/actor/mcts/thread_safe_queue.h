#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace engine {

template <typename T>
class ThreadSafeQueue {
 public:
  bool push(T item) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) return false;
      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return std::nullopt;

    T item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  template <class Clock, class Duration>
  std::optional<T> pop_until(
      std::chrono::time_point<Clock, Duration> deadline) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_until(lock, deadline,
                   [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return std::nullopt;

    T item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      closed_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  bool closed_ = false;
};

}  // namespace engine
