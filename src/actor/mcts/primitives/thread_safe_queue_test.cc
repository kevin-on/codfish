#include "actor/mcts/primitives/thread_safe_queue.h"

#include <chrono>
#include <future>
#include <thread>

#include <gtest/gtest.h>

namespace engine {
namespace {

using namespace std::chrono_literals;

TEST(ThreadSafeQueue, PopReturnsItemsInFifoOrder) {
  ThreadSafeQueue<int> queue;

  ASSERT_TRUE(queue.push(1));
  ASSERT_TRUE(queue.push(2));

  std::optional<int> first = queue.pop();
  std::optional<int> second = queue.pop();

  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, 1);
  EXPECT_EQ(*second, 2);
}

TEST(ThreadSafeQueue, PushFailsAfterClose) {
  ThreadSafeQueue<int> queue;
  queue.close();

  EXPECT_FALSE(queue.push(7));
}

TEST(ThreadSafeQueue, PopReturnsNulloptWhenClosedAndEmpty) {
  ThreadSafeQueue<int> queue;
  queue.close();

  std::optional<int> value = queue.pop();
  EXPECT_FALSE(value.has_value());
}

TEST(ThreadSafeQueue, PopUnblocksWhenProducerPushes) {
  ThreadSafeQueue<int> queue;
  std::future<std::optional<int>> consumer =
      std::async(std::launch::async, [&queue] { return queue.pop(); });

  std::this_thread::sleep_for(20ms);
  ASSERT_TRUE(queue.push(7));

  EXPECT_EQ(consumer.wait_for(1s), std::future_status::ready);
  std::optional<int> value = consumer.get();
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, 7);
}

TEST(ThreadSafeQueue, PopReturnsNulloptWhenCloseWakesConsumer) {
  ThreadSafeQueue<int> queue;
  std::future<std::optional<int>> consumer =
      std::async(std::launch::async, [&queue] { return queue.pop(); });

  std::this_thread::sleep_for(20ms);
  queue.close();

  EXPECT_EQ(consumer.wait_for(1s), std::future_status::ready);
  std::optional<int> value = consumer.get();
  EXPECT_FALSE(value.has_value());
}

TEST(ThreadSafeQueue, PopUntilTimesOutWhenEmpty) {
  ThreadSafeQueue<int> queue;

  std::optional<int> value = queue.pop_until(std::chrono::steady_clock::now() +
                                             20ms);
  EXPECT_FALSE(value.has_value());
}

TEST(ThreadSafeQueue, PopUntilReturnsItemBeforeDeadline) {
  ThreadSafeQueue<int> queue;
  std::future<void> producer = std::async(std::launch::async, [&queue] {
    std::this_thread::sleep_for(20ms);
    EXPECT_TRUE(queue.push(9));
  });

  std::optional<int> value =
      queue.pop_until(std::chrono::steady_clock::now() + 200ms);
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, 9);
  producer.get();
}

TEST(ThreadSafeQueue, PopUntilReturnsNulloptWhenCloseBeatsDeadline) {
  ThreadSafeQueue<int> queue;
  std::future<void> closer = std::async(std::launch::async, [&queue] {
    std::this_thread::sleep_for(20ms);
    queue.close();
  });

  std::optional<int> value =
      queue.pop_until(std::chrono::steady_clock::now() + 200ms);
  EXPECT_FALSE(value.has_value());
  closer.get();
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
