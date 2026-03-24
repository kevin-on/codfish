#include "mock_backend.h"

#include <gtest/gtest.h>

namespace engine {
namespace {

TEST(MockBackend, RunBeforeLoadReturnsError) {
  MockBackend backend;
  InferenceBatch batch;
  batch.planes = nullptr;
  batch.batch_size = 0;

  InferenceOutputs out;
  out.policy_logits = {1.0f};
  out.wdl_probs = {2.0f};
  const Status status = backend.Run(batch, &out);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "mock backend not loaded");
  EXPECT_TRUE(out.policy_logits.empty());
  EXPECT_TRUE(out.wdl_probs.empty());
}

TEST(MockBackend, LoadRejectsInvalidPolicySize) {
  MockBackend backend;
  ModelManifest manifest;
  manifest.policy_size = 0;

  const Status status = backend.Load(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "invalid policy size");
}

TEST(MockBackend, LoadRejectsInvalidInputChannels) {
  MockBackend backend;
  ModelManifest manifest;
  manifest.input_channels = 0;

  const Status status = backend.Load(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "invalid input channels");
}

TEST(MockBackend, RunWithNullOutReturnsError) {
  MockBackend backend;
  ModelManifest manifest;
  ASSERT_TRUE(backend.Load(manifest).ok());

  InferenceBatch batch;
  uint8_t dummy_planes[1] = {0};
  batch.planes = dummy_planes;
  batch.batch_size = 1;

  const Status status = backend.Run(batch, nullptr);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "output buffer is null");
}

TEST(MockBackend, RunWithNegativeBatchReturnsError) {
  MockBackend backend;
  ModelManifest manifest;
  ASSERT_TRUE(backend.Load(manifest).ok());

  InferenceBatch batch;
  batch.planes = nullptr;
  batch.batch_size = -1;

  InferenceOutputs out;
  out.policy_logits = {1.0f};
  out.wdl_probs = {2.0f};
  const Status status = backend.Run(batch, &out);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "negative batch size");
  EXPECT_TRUE(out.policy_logits.empty());
  EXPECT_TRUE(out.wdl_probs.empty());
}

TEST(MockBackend, RunWithNullPlanesAndPositiveBatchReturnsError) {
  MockBackend backend;
  ModelManifest manifest;
  ASSERT_TRUE(backend.Load(manifest).ok());

  InferenceBatch batch;
  batch.planes = nullptr;
  batch.batch_size = 1;

  InferenceOutputs out;
  const Status status = backend.Run(batch, &out);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "planes buffer is null");
}

TEST(MockBackend, LoadAndRunProducesExpectedShapes) {
  MockBackend backend;
  ModelManifest manifest;
  manifest.policy_size = 7;
  ASSERT_TRUE(backend.Load(manifest).ok());

  uint8_t dummy_planes[1] = {0};
  InferenceBatch batch;
  batch.planes = dummy_planes;
  batch.batch_size = 4;

  InferenceOutputs out;
  const Status status = backend.Run(batch, &out);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(out.policy_logits.size(), 28u);
  EXPECT_EQ(out.wdl_probs.size(), 12u);
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
