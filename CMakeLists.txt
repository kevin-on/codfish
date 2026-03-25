cmake_minimum_required(VERSION 3.23)

project(codfish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CODFISH_ENABLE_PEXT "Enable BMI2/PEXT-based move generation" ON)

set(CODFISH_PRODUCTION_SOURCES
    src/actor/backends/mock_backend.cc
    src/actor/mcts/output/chunk_writer_runtime.cc
    src/actor/mcts/output/raw_chunk_format.cc
    src/actor/mcts/runtime/game_runner.cc
    src/actor/mcts/runtime/inference_runtime.cc
    src/actor/mcts/runtime/worker_runtime.cc
    src/actor/mcts/search_coordinator.cc
    src/actor/mcts/searchers/gumbel/gumbel_mcts.cc
    src/engine/encoder.cc
    src/learner/sample_facade.cc
    src/learner/storage_facade.cc
    src/lc0/chess/board.cc
    src/lc0/chess/gamestate.cc
    src/lc0/chess/position.cc
    src/lc0/move_index.cc
)

set(CODFISH_TEST_SOURCES
    src/actor/backends/mock_backend_test.cc
    src/actor/mcts/output/chunk_writer_runtime_test.cc
    src/actor/mcts/output/raw_chunk_format_test.cc
    src/actor/mcts/primitives/thread_safe_queue_test.cc
    src/actor/mcts/runtime/game_runner_test.cc
    src/actor/mcts/runtime/inference_runtime_test.cc
    src/actor/mcts/runtime/worker_runtime_test.cc
    src/actor/mcts/search_coordinator_smoke_test.cc
    src/actor/mcts/search_coordinator_test.cc
    src/actor/mcts/searchers/gumbel/gumbel_mcts_test.cc
    src/engine/encoder_test.cc
    src/learner/public_api_test.cc
    src/learner/sample_facade_test.cc
    src/learner/storage_facade_test.cc
    src/lc0/chess/board_test.cc
    src/lc0/chess/position_test.cc
    src/lc0/move_index_test.cc
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(codfish_core STATIC ${CODFISH_PRODUCTION_SOURCES})
add_library(codfish::core ALIAS codfish_core)

target_include_directories(
    codfish_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/actor/backends
        ${PROJECT_SOURCE_DIR}/src/engine
        ${PROJECT_SOURCE_DIR}/src/lc0
)

target_link_libraries(codfish_core PUBLIC Threads::Threads)

set(CODFISH_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES codfish_arch_count)
    if(codfish_arch_count GREATER 1)
        message(FATAL_ERROR
            "CODFISH_ENABLE_PEXT=ON does not support universal builds. "
            "Use -DCMAKE_OSX_ARCHITECTURES=x86_64 or disable PEXT.")
    endif()

    list(GET CMAKE_OSX_ARCHITECTURES 0 CODFISH_TARGET_ARCH)
endif()

if(CODFISH_ENABLE_PEXT)
    if(NOT CODFISH_TARGET_ARCH MATCHES "^(x86_64|X86_64|amd64|AMD64)$")
        message(FATAL_ERROR
            "CODFISH_ENABLE_PEXT=ON requires an x86_64 BMI2 target. "
            "For local Apple Silicon development, configure with "
            "-DCODFISH_ENABLE_PEXT=OFF.")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(codfish_core PRIVATE -mbmi2)
    endif()
else()
    target_compile_definitions(codfish_core PRIVATE NO_PEXT)
endif()

include(CTest)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)

    function(codfish_add_test test_source)
        get_filename_component(test_name "${test_source}" NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE codfish_core GTest::gtest)
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endfunction()

    foreach(test_source IN LISTS CODFISH_TEST_SOURCES)
        codfish_add_test("${test_source}")
    endforeach()

    set_tests_properties(search_coordinator_smoke_test PROPERTIES LABELS smoke)
endif()
