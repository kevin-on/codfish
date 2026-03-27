#include <gtest/gtest.h>

#include <filesystem>
#include <type_traits>
#include <utility>

#include "learner/raw_types.h"
#include "learner/sample_facade.h"
#include "learner/storage_facade.h"

namespace engine::learner {
namespace {

using ReadRawChunkFileReturn =
    decltype(ReadRawChunkFile(std::declval<const std::filesystem::path&>()));
using EncodeRawGameReturn =
    decltype(EncodeRawGame(std::declval<const RawGame&>()));

static_assert(std::is_same_v<ReadRawChunkFileReturn, RawChunkFile>);
static_assert(std::is_same_v<EncodeRawGameReturn, EncodedGameSamples>);

TEST(LearnerPublicApi, RawTypesExposeLearnerOwnedStorageShape) {
  RawChunkFile chunk{
      .version = 1,
      .games =
          {
              RawGame{
                  .initial_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                  .game_result = lczero::GameResult::WHITE_WON,
                  .plies =
                      {
                          RawPly{
                              .selected_move_uci = "e2e4",
                              .policy =
                                  {
                                      RawPolicyEntry{
                                          .move_uci = "e2e4",
                                          .prob = 0.75f,
                                      },
                                      RawPolicyEntry{
                                          .move_uci = "d2d4",
                                          .prob = 0.25f,
                                      },
                                  },
                          },
                      },
              },
          },
  };

  ASSERT_EQ(chunk.version, 1u);
  ASSERT_EQ(chunk.games.size(), 1u);
  ASSERT_TRUE(chunk.games.front().initial_fen.has_value());
  EXPECT_EQ(*chunk.games.front().initial_fen,
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
  EXPECT_EQ(chunk.games.front().game_result, lczero::GameResult::WHITE_WON);
  ASSERT_EQ(chunk.games.front().plies.size(), 1u);
  EXPECT_EQ(chunk.games.front().plies.front().selected_move_uci, "e2e4");
  ASSERT_EQ(chunk.games.front().plies.front().policy.size(), 2u);
  EXPECT_EQ(chunk.games.front().plies.front().policy.front().move_uci, "e2e4");
  EXPECT_FLOAT_EQ(chunk.games.front().plies.front().policy.front().prob, 0.75f);
}

TEST(LearnerPublicApi, EncodedGameSamplesExposeStableBufferMetadata) {
  const EncodedGameSamples samples;

  EXPECT_EQ(samples.sample_count, 0);
  EXPECT_EQ(samples.input_channels, kInputPlanes);
  EXPECT_EQ(samples.policy_size, lczero::kPolicySize);
  EXPECT_TRUE(samples.inputs.empty());
  EXPECT_TRUE(samples.policy_targets.empty());
  EXPECT_TRUE(samples.wdl_targets.empty());
}

}  // namespace
}  // namespace engine::learner

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
