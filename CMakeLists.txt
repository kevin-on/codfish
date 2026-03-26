cmake_minimum_required(VERSION 3.23)

project(codfish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CODFISH_ENABLE_PEXT "Enable BMI2/PEXT-based move generation" ON)
option(CODFISH_BUILD_LEARNER_PYTHON "Build Python learner bindings" OFF)

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
    src/learner/sample_facade_public_test.cc
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

if(CODFISH_BUILD_LEARNER_PYTHON)
    set(Python3_FIND_VIRTUALENV FIRST)
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
        "Selected Python interpreter for nanobind" FORCE)
    find_package(Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
        COMPONENTS Interpreter Development.Module REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.9.2
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nanobind)

    set(CODFISH_PYTHON_PACKAGE_DIR "${PROJECT_BINARY_DIR}/python")
    set(CODFISH_PYTHON_LEARNER_DIR
        "${CODFISH_PYTHON_PACKAGE_DIR}/codfish/learner")
    set(CODFISH_PYTHON_SOURCES
        "${PROJECT_SOURCE_DIR}/python/codfish/__init__.py"
        "${PROJECT_SOURCE_DIR}/python/codfish/learner/__init__.py"
        "${PROJECT_SOURCE_DIR}/python/codfish/learner/_api.py"
        "${PROJECT_SOURCE_DIR}/python/codfish/learner/_replay.py")
    set(CODFISH_PYTHON_STAMP
        "${CODFISH_PYTHON_PACKAGE_DIR}/.codfish_python_staged")

    add_custom_command(
        OUTPUT "${CODFISH_PYTHON_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CODFISH_PYTHON_LEARNER_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PROJECT_SOURCE_DIR}/python/codfish/__init__.py"
                "${CODFISH_PYTHON_PACKAGE_DIR}/codfish/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PROJECT_SOURCE_DIR}/python/codfish/learner/__init__.py"
                "${CODFISH_PYTHON_LEARNER_DIR}/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PROJECT_SOURCE_DIR}/python/codfish/learner/_api.py"
                "${CODFISH_PYTHON_LEARNER_DIR}/_api.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PROJECT_SOURCE_DIR}/python/codfish/learner/_replay.py"
                "${CODFISH_PYTHON_LEARNER_DIR}/_replay.py"
        COMMAND ${CMAKE_COMMAND} -E touch "${CODFISH_PYTHON_STAMP}"
        DEPENDS ${CODFISH_PYTHON_SOURCES}
        VERBATIM)
    add_custom_target(codfish_python_package ALL
        DEPENDS "${CODFISH_PYTHON_STAMP}")

    nanobind_add_module(_native src/learner/python_bindings.cc)
    target_link_libraries(_native PRIVATE codfish_core)
    add_dependencies(_native codfish_python_package)
    set_target_properties(
        _native
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CODFISH_PYTHON_LEARNER_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CODFISH_PYTHON_LEARNER_DIR}")
endif()

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

    if(CODFISH_BUILD_LEARNER_PYTHON)
        add_executable(
            learner_raw_chunk_golden_writer
            src/learner/raw_chunk_golden_writer.cc)
        target_link_libraries(
            learner_raw_chunk_golden_writer
            PRIVATE codfish_core)

        add_test(
            NAME learner_python_test
            COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CODFISH_PYTHON_PACKAGE_DIR}"
                    "CODFISH_LEARNER_GOLDEN_CHUNK_WRITER=$<TARGET_FILE:learner_raw_chunk_golden_writer>"
                    "${Python3_EXECUTABLE}" -m unittest discover
                    -s "${PROJECT_SOURCE_DIR}/python/tests" -p "test_*.py")
    endif()
endif()
