#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "actor/mcts/aoti_match.h"
#include "actor/mcts/aoti_selfplay.h"
#include "actor/mcts/mock_selfplay.h"
#include "learner/raw_types.h"
#include "learner/sample_facade.h"
#include "learner/storage_facade.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace engine::learner {
namespace {

struct PythonEncodedGameSamples {
  int sample_count = 0;
  int input_channels = 0;
  int policy_size = 0;
  nb::object inputs;
  nb::object policy_targets;
  nb::object wdl_targets;
};

struct PythonModelIOShape {
  int input_channels = 0;
  int policy_size = 0;
};

template <typename T>
nb::object MakeOwnedNumpyArray(std::span<const T> values, nb::handle dtype,
                               nb::tuple shape) {
  nb::module_ numpy = nb::module_::import_("numpy");
  const char* raw =
      values.empty() ? "" : reinterpret_cast<const char*>(values.data());
  const std::size_t raw_size = values.size() * sizeof(T);
  nb::object array = numpy.attr("frombuffer")(nb::bytes(raw, raw_size), dtype);
  return array.attr("reshape")(shape).attr("copy")();
}

PythonEncodedGameSamples ToPythonEncodedGameSamples(
    const EncodedGameSamples& samples) {
  nb::module_ numpy = nb::module_::import_("numpy");

  return PythonEncodedGameSamples{
      .sample_count = samples.sample_count,
      .input_channels = samples.input_channels,
      .policy_size = samples.policy_size,
      .inputs = MakeOwnedNumpyArray<uint8_t>(
          samples.inputs, numpy.attr("uint8"),
          nb::make_tuple(samples.sample_count, samples.input_channels, 8, 8)),
      .policy_targets = MakeOwnedNumpyArray<float>(
          samples.policy_targets, numpy.attr("float32"),
          nb::make_tuple(samples.sample_count, samples.policy_size)),
      .wdl_targets =
          MakeOwnedNumpyArray<float>(samples.wdl_targets, numpy.attr("float32"),
                                     nb::make_tuple(samples.sample_count, 3)),
  };
}

PythonModelIOShape ToPythonModelIOShape() {
  return PythonModelIOShape{
      .input_channels = engine::kInputPlanes,
      .policy_size = lczero::kPolicySize,
  };
}

}  // namespace
}  // namespace engine::learner

NB_MODULE(_native, m) {
  using engine::learner::EncodedGameSamples;
  using engine::learner::PythonEncodedGameSamples;
  using engine::learner::PythonModelIOShape;
  using engine::learner::RawChunkFile;
  using engine::learner::RawGame;
  using engine::learner::RawPly;
  using engine::learner::RawPolicyEntry;
  using engine::learner::ToPythonEncodedGameSamples;
  using engine::learner::ToPythonModelIOShape;

  m.doc() = "Private learner bindings for codfish.";

  nb::enum_<lczero::GameResult>(m, "GameResult")
      .value("UNDECIDED", lczero::GameResult::UNDECIDED)
      .value("WHITE_WON", lczero::GameResult::WHITE_WON)
      .value("DRAW", lczero::GameResult::DRAW)
      .value("BLACK_WON", lczero::GameResult::BLACK_WON)
      .export_values();

  nb::class_<RawPolicyEntry>(m, "RawPolicyEntry")
      .def(nb::init<>())
      .def_rw("move_uci", &RawPolicyEntry::move_uci)
      .def_rw("prob", &RawPolicyEntry::prob);

  nb::class_<RawPly>(m, "RawPly")
      .def(nb::init<>())
      .def_rw("selected_move_uci", &RawPly::selected_move_uci)
      .def_rw("policy", &RawPly::policy);

  nb::class_<RawGame>(m, "RawGame")
      .def(nb::init<>())
      .def_prop_rw(
          "initial_fen",
          [](const RawGame& self) -> nb::object {
            if (!self.initial_fen.has_value()) {
              return nb::none();
            }
            return nb::str(self.initial_fen->c_str());
          },
          [](RawGame& self, nb::object value) {
            if (value.is_none()) {
              self.initial_fen.reset();
              return;
            }
            self.initial_fen = nb::cast<std::string>(value);
          })
      .def_rw("game_result", &RawGame::game_result)
      .def_rw("plies", &RawGame::plies);

  nb::class_<RawChunkFile>(m, "RawChunkFile")
      .def(nb::init<>())
      .def_rw("version", &RawChunkFile::version)
      .def_rw("games", &RawChunkFile::games);

  nb::class_<PythonEncodedGameSamples>(m, "EncodedGameSamples")
      .def(nb::init<>())
      .def_rw("sample_count", &PythonEncodedGameSamples::sample_count)
      .def_rw("input_channels", &PythonEncodedGameSamples::input_channels)
      .def_rw("policy_size", &PythonEncodedGameSamples::policy_size)
      .def_rw("inputs", &PythonEncodedGameSamples::inputs)
      .def_rw("policy_targets", &PythonEncodedGameSamples::policy_targets)
      .def_rw("wdl_targets", &PythonEncodedGameSamples::wdl_targets);

  nb::class_<PythonModelIOShape>(m, "ModelIOShape")
      .def(nb::init<>())
      .def_rw("input_channels", &PythonModelIOShape::input_channels)
      .def_rw("policy_size", &PythonModelIOShape::policy_size);

  m.def("read_raw_chunk_file", [](const std::string& path) {
    return engine::learner::ReadRawChunkFile(std::filesystem::path(path));
  });

  m.def("encode_raw_game", [](const RawGame& raw_game) {
    return ToPythonEncodedGameSamples(engine::learner::EncodeRawGame(raw_game));
  });

  m.def("get_model_io_shape", []() { return ToPythonModelIOShape(); });

  m.def(
      "run_aoti_selfplay",
      [](const std::string& model_package_path, int input_channels,
         int policy_size, const std::string& raw_output_dir, int num_workers,
         int num_games, uint64_t raw_chunk_max_bytes, int num_action,
         int num_simulation, float c_puct, float c_visit, float c_scale) {
        engine::RunAotiSelfPlay(engine::AotiSelfPlayOptions{
            .model_package_path = std::filesystem::path(model_package_path),
            .input_channels = input_channels,
            .policy_size = policy_size,
            .raw_output_dir = std::filesystem::path(raw_output_dir),
            .num_workers = num_workers,
            .num_games = num_games,
            .raw_chunk_max_bytes = raw_chunk_max_bytes,
            .num_action = num_action,
            .num_simulation = num_simulation,
            .c_puct = c_puct,
            .c_visit = c_visit,
            .c_scale = c_scale,
        });
      },
      nb::arg("model_package_path"), nb::arg("input_channels"),
      nb::arg("policy_size"), nb::arg("raw_output_dir"), nb::arg("num_workers"),
      nb::arg("num_games"), nb::arg("raw_chunk_max_bytes"),
      nb::arg("num_action"), nb::arg("num_simulation"), nb::arg("c_puct"),
      nb::arg("c_visit"), nb::arg("c_scale"),
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "run_aoti_match",
      [](const std::string& model_package_path_a,
         const std::string& model_package_path_b,
         const std::string& player_name_a, const std::string& player_name_b,
         int input_channels, int policy_size,
         const std::string& output_pgn_path, int num_workers, int num_games,
         int num_action, int num_simulation, float c_puct, float c_visit,
         float c_scale) {
        engine::RunAotiMatch(engine::AotiMatchOptions{
            .model_package_paths =
                {
                    std::filesystem::path(model_package_path_a),
                    std::filesystem::path(model_package_path_b),
                },
            .player_names = {player_name_a, player_name_b},
            .input_channels = input_channels,
            .policy_size = policy_size,
            .output_pgn_path = std::filesystem::path(output_pgn_path),
            .num_workers = num_workers,
            .num_games = num_games,
            .num_action = num_action,
            .num_simulation = num_simulation,
            .c_puct = c_puct,
            .c_visit = c_visit,
            .c_scale = c_scale,
        });
      },
      nb::arg("model_package_path_a"), nb::arg("model_package_path_b"),
      nb::arg("player_name_a"), nb::arg("player_name_b"),
      nb::arg("input_channels"), nb::arg("policy_size"),
      nb::arg("output_pgn_path"), nb::arg("num_workers"), nb::arg("num_games"),
      nb::arg("num_action"), nb::arg("num_simulation"), nb::arg("c_puct"),
      nb::arg("c_visit"), nb::arg("c_scale"),
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "run_mock_selfplay",
      [](const std::string& raw_output_dir, int num_workers, int num_games,
         uint64_t raw_chunk_max_bytes, int num_action, int num_simulation,
         float c_puct, float c_visit, float c_scale) {
        engine::RunMockSelfPlay(engine::MockSelfPlayOptions{
            .raw_output_dir = std::filesystem::path(raw_output_dir),
            .num_workers = num_workers,
            .num_games = num_games,
            .raw_chunk_max_bytes = raw_chunk_max_bytes,
            .num_action = num_action,
            .num_simulation = num_simulation,
            .c_puct = c_puct,
            .c_visit = c_visit,
            .c_scale = c_scale,
        });
      },
      nb::arg("raw_output_dir"), nb::arg("num_workers"), nb::arg("num_games"),
      nb::arg("raw_chunk_max_bytes"), nb::arg("num_action"),
      nb::arg("num_simulation"), nb::arg("c_puct"), nb::arg("c_visit"),
      nb::arg("c_scale"), nb::call_guard<nb::gil_scoped_release>());
}
