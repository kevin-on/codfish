#pragma once

#include <cassert>
#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

namespace engine {

template <typename Request, typename Reply, typename Result>
class RequestReplyCoroutine {
 public:
  struct promise_type {
    std::optional<Request> yielded;
    std::optional<Reply> received;
    std::optional<Result> result;

    RequestReplyCoroutine get_return_object() {
      return RequestReplyCoroutine{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_value(Result value) { result = std::move(value); }
    void unhandled_exception() { std::terminate(); }

    auto yield_value(Request value) {
      yielded = std::move(value);
      struct Awaiter {
        promise_type* promise;
        bool await_ready() { return false; }
        void await_suspend(std::coroutine_handle<>) {}
        Reply await_resume() {
          assert(promise->received.has_value());
          Reply value = std::move(*promise->received);
          promise->received.reset();
          return value;
        }
      };
      return Awaiter{this};
    }
  };

  std::optional<Request> next() {
    if (!handle_ || handle_.done()) return std::nullopt;
    assert(!started_);
    assert(!waiting_for_reply_);

    started_ = true;
    handle_.resume();
    if (handle_.done()) {
      waiting_for_reply_ = false;
      return std::nullopt;
    }

    waiting_for_reply_ = true;
    return TakeYielded();
  }

  std::optional<Request> send(Reply input) {
    if (!handle_ || handle_.done()) return std::nullopt;
    assert(started_);
    assert(waiting_for_reply_);

    handle_.promise().received = std::move(input);
    waiting_for_reply_ = false;
    handle_.resume();
    if (handle_.done()) return std::nullopt;

    waiting_for_reply_ = true;
    return TakeYielded();
  }

  bool done() const { return !handle_ || handle_.done(); }
  const Result& result() const {
    assert(handle_ && handle_.done());
    assert(handle_.promise().result.has_value());
    return *handle_.promise().result;
  }
  Result take_result() {
    assert(handle_ && handle_.done());
    assert(handle_.promise().result.has_value());
    Result value = std::move(*handle_.promise().result);
    handle_.promise().result.reset();
    return value;
  }

  ~RequestReplyCoroutine() {
    if (handle_) handle_.destroy();
  }
  RequestReplyCoroutine(RequestReplyCoroutine&& o)
      : handle_(o.handle_),
        started_(o.started_),
        waiting_for_reply_(o.waiting_for_reply_) {
    o.handle_ = nullptr;
    o.started_ = false;
    o.waiting_for_reply_ = false;
  }
  RequestReplyCoroutine& operator=(RequestReplyCoroutine&& o) {
    if (handle_) handle_.destroy();
    handle_ = o.handle_;
    started_ = o.started_;
    waiting_for_reply_ = o.waiting_for_reply_;
    o.handle_ = nullptr;
    o.started_ = false;
    o.waiting_for_reply_ = false;
    return *this;
  }
  RequestReplyCoroutine(const RequestReplyCoroutine&) = delete;
  RequestReplyCoroutine& operator=(const RequestReplyCoroutine&) = delete;

 private:
  std::optional<Request> TakeYielded() {
    assert(handle_.promise().yielded.has_value());
    std::optional<Request> value = std::move(handle_.promise().yielded);
    handle_.promise().yielded.reset();
    return value;
  }

  RequestReplyCoroutine(std::coroutine_handle<promise_type> h) : handle_(h) {}
  std::coroutine_handle<promise_type> handle_;
  bool started_ = false;
  bool waiting_for_reply_ = false;
};

}  // namespace engine
