#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

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

}  // namespace
}  // namespace engine::learner

NB_MODULE(_native, m) {
  using engine::learner::EncodedGameSamples;
  using engine::learner::PythonEncodedGameSamples;
  using engine::learner::RawChunkFile;
  using engine::learner::RawGame;
  using engine::learner::RawPly;
  using engine::learner::RawPolicyEntry;
  using engine::learner::ToPythonEncodedGameSamples;

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

  m.def("read_raw_chunk_file", [](const std::string& path) {
    return engine::learner::ReadRawChunkFile(std::filesystem::path(path));
  });

  m.def("encode_raw_game", [](const RawGame& raw_game) {
    return ToPythonEncodedGameSamples(engine::learner::EncodeRawGame(raw_game));
  });
}
