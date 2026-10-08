cmake_minimum_required(VERSION 3.24)

# sm_80 = A100, sm_90 = H100 (NVL). sm_89 (L40) is opportunistic and we don't
# tune for it. CUDA 12.6 supports all three. This must precede project():
# enabling CUDA there initializes CMAKE_CUDA_ARCHITECTURES to the compiler
# default (52, PTX only), so a later "if(NOT DEFINED)" never fires -- which is
# what a plain `pip install .` used to build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;90" CACHE STRING "CUDA architectures")
endif()

project(gpu_bpe_tokenizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# ===========================================================================
# Vocab selection. Single primary knob; family + slot policy derive from it.
#
#   -DGBPE_VOCABS=ALL                              (default set; Qwen excluded)
#   -DGBPE_VOCABS=GPT2                             (smallest, fastest GPT-2 build)
#   -DGBPE_VOCABS=LLAMA3                           (Llama-3 only)
#   -DGBPE_VOCABS=QWEN25                           (Qwen2.5 only)
#   -DGBPE_VOCABS=DEEPSEEK_V3                      (DeepSeek-V3 only)
#   -DGBPE_VOCABS=GEMMA3                           (Gemma 3 only, SP-family)
#   -DGBPE_VOCABS="GPT2;LLAMA3"                    (multi-select)
#
# In single-vocab builds the kernel is non-templated on Slot policy and only
# the matching pre-tokenizer regex set is compiled in. Source files that
# belong to a non-selected family are excluded from compilation entirely.
# ===========================================================================

set(GBPE_KNOWN_VOCABS GPT2 LLAMA3 QWEN25 DEEPSEEK_V3 GEMMA3)
set(GBPE_ALL_VOCABS
  GPT2
  LLAMA3
  # QWEN25  # NFC normalization currently runs on the host via utf8proc.
  DEEPSEEK_V3
  GEMMA3
)
set(GBPE_VOCABS "ALL" CACHE STRING
    "Which vocabs to compile support for. Use 'ALL' (Qwen excluded) or a ;-separated subset of: ${GBPE_KNOWN_VOCABS}")
option(GBPE_GEMMA_GPU_PRETOK
       "Use the graph-resident Gemma 3 pre-tokenizer instead of SPPreTokenizer" ON)

# Host-encoder diagnostics and cache geometry. All default to off/empty so the
# shipped build is uninstrumented and uses the in-source defaults.
option(GBPE_HOST_STATS
       "Compile pre-token cache hit/miss/eviction counters into the host encoder" OFF)
set(GBPE_CACHE_BITS "" CACHE STRING
    "log2 of the hashed pre-token cache entry count (default 18; 104 B/entry)")
set(GBPE_SHORT_BITS "" CACHE STRING
    "log2 of the short-key pre-token cache entry count (default 18; 32 B/entry)")
set(GBPE_BATCH_DIRECT_NUMPY "" CACHE STRING
    "1 (default) = batch D2H writes straight into the returned NumPy array; 0 = legacy vector+memcpy, A/B only")
set(GBPE_HOST_STOP_AFTER "" CACHE STRING
    "truncate the host encoder for stage attribution: 1 = after classify, 2 = after the boundary scan. Emits WRONG TOKENS; measurement only")

# Resolve ALL → default list. Qwen remains available through QWEN25 or an
# explicit multi-vocab list, but is excluded here until NFC is GPU-resident.
if(GBPE_VOCABS STREQUAL "ALL")
  set(_gbpe_vocabs ${GBPE_ALL_VOCABS})
else()
  set(_gbpe_vocabs ${GBPE_VOCABS})
endif()

# Validate.
foreach(v IN LISTS _gbpe_vocabs)
  if(NOT v IN_LIST GBPE_KNOWN_VOCABS)
    message(FATAL_ERROR "Unknown vocab '${v}' in GBPE_VOCABS. Known: ${GBPE_KNOWN_VOCABS}")
  endif()
endforeach()

# Derive family flags.
set(_have_bytelevel 0)  # GPT-2, Llama-3, Qwen2.5, DeepSeek-V3
set(_have_sp 0)         # Gemma 3 (and future SentencePiece-derived BPE)
set(_have_qwen25 0)     # the only supported byte-level vocab with NFC
foreach(v IN LISTS _gbpe_vocabs)
  if(v STREQUAL "GPT2" OR v STREQUAL "LLAMA3" OR v STREQUAL "QWEN25" OR v STREQUAL "DEEPSEEK_V3")
    set(_have_bytelevel 1)
    if(v STREQUAL "QWEN25")
      set(_have_qwen25 1)
    endif()
  elseif(v STREQUAL "GEMMA3")
    set(_have_sp 1)
  endif()
endforeach()

if(GBPE_GEMMA_GPU_PRETOK AND _have_sp)
  set(_gemma_gpu_pretok 1)
else()
  set(_gemma_gpu_pretok 0)
endif()

# Derive slot policy. GPT-2 alone -> Slot64; everything else -> Slot128;
# anything with mixed sizes (including any non-GPT2 vocab) -> AUTO (both).
list(LENGTH _gbpe_vocabs _nvocabs)
set(_slot_only "")
if(_nvocabs EQUAL 1)
  list(GET _gbpe_vocabs 0 _only)
  if(_only STREQUAL "GPT2")
    set(_slot_only "64")
  else()
    set(_slot_only "128")
  endif()
endif()

message(STATUS "GBPE vocabs:     ${_gbpe_vocabs}")
message(STATUS "GBPE family:     bytelevel=${_have_bytelevel} sp=${_have_sp}")
message(STATUS "Gemma pretok:    gpu=${_gemma_gpu_pretok}")
if(_slot_only STREQUAL "")
  message(STATUS "GBPE slot:       AUTO (Slot64 + Slot128, runtime dispatch)")
else()
  message(STATUS "GBPE slot:       Slot${_slot_only} ONLY (no runtime dispatch)")
endif()

# Unicode normalization is needed only when Qwen-2.5 support is compiled in.
# Keeping this conditional preserves dependency-free GPT-2/Llama/DeepSeek/
# Gemma specialized builds. The source is guarded by the same compile-time
# vocabulary flag.
if(_have_qwen25)
  find_path(UTF8PROC_INCLUDE_DIR utf8proc.h REQUIRED)
  find_library(UTF8PROC_LIB utf8proc REQUIRED)
endif()

# nlohmann/json vendored under third_party/ (single-header).
set(NLOHMANN_JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party")

# Everything that links the tokenizer core needs the same vocab/family macros,
# include dirs, optimization flags and CUDA separable-compilation settings.
# Factored out so test executables cannot drift from the shipping binary — a
# mismatched -DGBPE_* between the CLI and a bit-exactness gate would make the
# gate meaningless.
function(gbpe_configure_target tgt)
  # Per-vocab and per-family compile macros — the source code reads these via
  # #if GBPE_HAVE_VOCAB_*, #if GBPE_HAVE_FAMILY_*, #ifdef GBPE_SLOT_ONLY.
  foreach(v IN LISTS GBPE_KNOWN_VOCABS)
    if(v IN_LIST _gbpe_vocabs)
      target_compile_definitions(${tgt} PRIVATE GBPE_HAVE_VOCAB_${v}=1)
    else()
      target_compile_definitions(${tgt} PRIVATE GBPE_HAVE_VOCAB_${v}=0)
    endif()
  endforeach()
  target_compile_definitions(${tgt} PRIVATE
    GBPE_HAVE_FAMILY_BYTELEVEL=${_have_bytelevel}
    GBPE_HAVE_FAMILY_SP=${_have_sp}
    GBPE_GEMMA_GPU_PRETOK=${_gemma_gpu_pretok})
  if(NOT _slot_only STREQUAL "")
    target_compile_definitions(${tgt} PRIVATE GBPE_SLOT_ONLY=${_slot_only})
  endif()
  # GPU pre-tokenizer stage is now the ONLY byte-level path (legacy host PCRE2
  # byte-level pipeline removed); it is unconditional via build_config.h. The
  # classification-table header lives in generated/.
  target_include_directories(${tgt} PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${UTF8PROC_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
  )
  target_link_libraries(${tgt} PRIVATE ${UTF8PROC_LIB} CUDA::cudart)
  # Aggressive optimization on the CUDA side; we want the kernel inlined hard.
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --extended-lambda -lineinfo -Xcompiler=-Wall>
  )
  set_target_properties(${tgt} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endfunction()

add_executable(gpu_bpe_tokenize
  src/main.cc
  src/vocab.cc
  src/pretokenize.cc
  src/host_encode.cc
  src/special_tokens.cc
  src/tokenizer.cu
)
# Position output binary at build/gpu_bpe_tokenize (default).
gbpe_configure_target(gpu_bpe_tokenize)
# The stage-profile collector uses host NVTX ranges to delimit individual
# graph replays for Nsight Systems.  Keep this dependency on the CLI only so
# the library-like smoke targets retain their minimal link surface.
if(TARGET CUDA::nvToolsExt)
  target_link_libraries(gpu_bpe_tokenize PRIVATE CUDA::nvToolsExt)
else()
  find_library(GBPE_NVTOOLSEXT_LIBRARY NAMES nvToolsExt
    HINTS "${CUDAToolkit_LIBRARY_DIR}")
  if(NOT GBPE_NVTOOLSEXT_LIBRARY)
    message(FATAL_ERROR "NVTX is required for the stage-profile CLI option")
  endif()
  target_link_libraries(gpu_bpe_tokenize PRIVATE "${GBPE_NVTOOLSEXT_LIBRARY}")
endif()

# ===========================================================================
# CPU route engine: gigatoken (rs-gigatoken 0.10.0, MIT; vendored under
# third_party/rs-gigatoken) linked statically through rust/gtok_cpu.
#
#   -DGBPE_GIGATOKEN=AUTO  (default) link it when a nightly cargo is found,
#                          otherwise build with the host encoder only
#   -DGBPE_GIGATOKEN=ON    require it
#   -DGBPE_GIGATOKEN=OFF   never link it
# ===========================================================================
set(GBPE_GIGATOKEN "AUTO" CACHE STRING "Link gigatoken as the CPU route engine (AUTO|ON|OFF)")
set(_gbpe_gigatoken 0)
if(NOT GBPE_GIGATOKEN STREQUAL "OFF")
  find_program(GBPE_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin")
  set(_gtok_cpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/rust/gtok_cpu")
  if(GBPE_CARGO)
    # rust-toolchain.toml in the crate selects nightly; probe it from there.
    execute_process(COMMAND "${GBPE_CARGO}" --version
                    WORKING_DIRECTORY "${_gtok_cpu_dir}"
                    RESULT_VARIABLE _cargo_rc OUTPUT_VARIABLE _cargo_ver
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cargo_rc EQUAL 0 AND _cargo_ver MATCHES "nightly")
      set(_gbpe_gigatoken 1)
    endif()
  endif()
  if(NOT _gbpe_gigatoken AND GBPE_GIGATOKEN STREQUAL "ON")
    message(FATAL_ERROR "GBPE_GIGATOKEN=ON needs a nightly cargo (found: '${_cargo_ver}')")
  endif()
endif()
if(_gbpe_gigatoken)
  set(_gtok_cpu_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
  set(_gtok_cpu_lib "${_gtok_cpu_target_dir}/release/libgtok_cpu.a")
  file(GLOB_RECURSE _gtok_cpu_sources CONFIGURE_DEPENDS
       "${_gtok_cpu_dir}/src/*.rs"
       "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rs-gigatoken/src/*.rs")
  add_custom_command(
    OUTPUT "${_gtok_cpu_lib}"
    COMMAND "${CMAKE_COMMAND}" -E env CARGO_TARGET_DIR=${_gtok_cpu_target_dir}
            "${GBPE_CARGO}" build --release --locked
            --manifest-path "${_gtok_cpu_dir}/Cargo.toml"
    WORKING_DIRECTORY "${_gtok_cpu_dir}"
    DEPENDS ${_gtok_cpu_sources} "${_gtok_cpu_dir}/Cargo.toml" "${_gtok_cpu_dir}/Cargo.lock"
    COMMENT "Building the gigatoken CPU engine (cargo, nightly)"
    VERBATIM)
  add_custom_target(gtok_cpu_build DEPENDS "${_gtok_cpu_lib}")
  add_library(gtok_cpu_rs STATIC IMPORTED)
  set_target_properties(gtok_cpu_rs PROPERTIES IMPORTED_LOCATION "${_gtok_cpu_lib}")
  add_dependencies(gtok_cpu_rs gtok_cpu_build)
  message(STATUS "CPU route:       gigatoken (${_cargo_ver}) + host encoder fallback")
else()
  message(STATUS "CPU route:       host encoder only (gigatoken not linked)")
endif()

# Link the CPU route into a target: cpu_route.cc always, gigatoken when built.
function(gbpe_link_cpu_route tgt)
  target_sources(${tgt} PRIVATE src/cpu_route.cc src/dispatcher.cc)
  target_compile_definitions(${tgt} PRIVATE GBPE_HAVE_GIGATOKEN=${_gbpe_gigatoken})
  if(_gbpe_gigatoken)
    target_link_libraries(${tgt} PRIVATE gtok_cpu_rs pthread dl m)
    # Keep the Rust archive's symbols out of the dynamic symbol table: hosts
    # such as Dynamo are Rust processes themselves.
    target_link_options(${tgt} PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
endfunction()

# C ABI shared library (libgtok.so) for the serving integrations: the vLLM
# adapter loads it with ctypes and NVIDIA Dynamo's native tokenizer backend
# with dlopen. Same sources and flags as the Python module, with the C shim in
# place of the pybind11 bindings; whole-program CUDA compilation for the same
# reason (a shared library loaded into another process).
add_library(gtok SHARED
  src/gtok_shim.cc
  src/vocab.cc
  src/pretokenize.cc
  src/host_encode.cc
  src/tokenizer.cu
)
gbpe_configure_target(gtok)
gbpe_link_cpu_route(gtok)
target_link_options(gtok PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/gtok.map")
set_property(TARGET gtok APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gtok.map")
set_target_properties(gtok PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# encode_batch seam gate (tests/test_encode_batch_smoke.cc).
#
# Asserts TokenizerCtx::encode_batch() is bit-exact with the concatenation of
# per-document encode() calls across the adversarial seam corpus. Byte-level
# only, so it is not built for SP-only configurations (e.g. the gemma3 preset).
# Run it as:  ./build/<preset>/test_encode_batch_smoke <tokenizer.json>
# ---------------------------------------------------------------------------
option(GBPE_BUILD_TESTS "Build the C++ correctness-gate executables" ON)
if(GBPE_BUILD_TESTS AND _have_bytelevel)
  # The smoke test uses device-side declarations from tokenizer.cuh; compile it
  # as CUDA exactly as the ad-hoc `nvcc -x cu` recipe did.
  set_source_files_properties(tests/test_encode_batch_smoke.cc
                              PROPERTIES LANGUAGE CUDA)
  add_executable(test_encode_batch_smoke
    tests/test_encode_batch_smoke.cc
    src/vocab.cc
    src/pretokenize.cc
    src/host_encode.cc
    src/special_tokens.cc
    src/tokenizer.cu
  )
  gbpe_configure_target(test_encode_batch_smoke)

  # Host-only: the CPU route's AVX2 content classifier must count exactly what
  # its scalar form counts.
  add_executable(test_cpu_route_classifier tests/test_cpu_route_classifier.cc
    src/vocab.cc src/pretokenize.cc src/host_encode.cc)
  gbpe_configure_target(test_cpu_route_classifier)
  target_compile_definitions(test_cpu_route_classifier PRIVATE GBPE_HAVE_GIGATOKEN=0)

  # Host-only: the AVX2 boundary fast path must answer exactly what the scalar
  # one answers, on every lane it claims to resolve. Needs no GPU and no data
  # files (the random trials are the gate); pass a corpus directory to also
  # scan real text.
  add_executable(test_boundary_simd_equiv tests/test_boundary_simd_equiv.cc)
  target_include_directories(test_boundary_simd_equiv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${CMAKE_CURRENT_SOURCE_DIR}/generated)
endif()

message(STATUS "CUDA arch:       ${CMAKE_CUDA_ARCHITECTURES}")
if(_have_qwen25)
  message(STATUS "utf8proc (Qwen): ${UTF8PROC_LIB}")
else()
  message(STATUS "utf8proc (Qwen): not required")
endif()
message(STATUS "nlohmann/json:   ${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")

# ===========================================================================
# Optional Python bindings (pybind11). Built only when requested explicitly
# via -DGBPE_BUILD_PYTHON=ON or when scikit-build-core drives the build
# (which sets SKBUILD). CLI presets are unaffected.
# ===========================================================================

option(GBPE_BUILD_PYTHON "Build the gpu_bpe_tokenizer Python extension module" OFF)
if(DEFINED SKBUILD OR SKBUILD)
  set(GBPE_BUILD_PYTHON ON CACHE BOOL "" FORCE)
endif()

if(GBPE_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(gpu_bpe_tokenizer_py MODULE
    src/python_bindings.cc
    src/vocab.cc
    src/pretokenize.cc
    src/host_encode.cc
    src/tokenizer.cu
  )

  gbpe_link_cpu_route(gpu_bpe_tokenizer_py)
  set_target_properties(gpu_bpe_tokenizer_py PROPERTIES
    OUTPUT_NAME "_gpu_bpe_tokenizer"
    # Whole-program compilation: avoids the device-link `fatbinData` ABI
    # issue when loading a MODULE library into the Python interpreter, and
    # tokenizer.cu doesn't span TUs so we lose nothing.
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE ON
  )

  foreach(v IN LISTS GBPE_KNOWN_VOCABS)
    if(v IN_LIST _gbpe_vocabs)
      target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_HAVE_VOCAB_${v}=1)
    else()
      target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_HAVE_VOCAB_${v}=0)
    endif()
  endforeach()
  target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE
    GBPE_HAVE_FAMILY_BYTELEVEL=${_have_bytelevel}
    GBPE_HAVE_FAMILY_SP=${_have_sp}
    GBPE_GEMMA_GPU_PRETOK=${_gemma_gpu_pretok})
  # Diagnostic / tuning knobs, all OFF by default so the shipped build carries
  # no instrumentation and the default cache geometry.
  if(GBPE_HOST_STATS)
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_HOST_STATS=1)
  endif()
  if(NOT GBPE_CACHE_BITS STREQUAL "")
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_CACHE_BITS=${GBPE_CACHE_BITS})
  endif()
  if(NOT GBPE_SHORT_BITS STREQUAL "")
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_SHORT_BITS=${GBPE_SHORT_BITS})
  endif()
  if(NOT GBPE_HOST_STOP_AFTER STREQUAL "")
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE
      GBPE_HOST_STOP_AFTER=${GBPE_HOST_STOP_AFTER})
  endif()
  if(NOT GBPE_BATCH_DIRECT_NUMPY STREQUAL "")
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_BATCH_DIRECT_NUMPY=${GBPE_BATCH_DIRECT_NUMPY})
  endif()
  if(NOT _slot_only STREQUAL "")
    target_compile_definitions(gpu_bpe_tokenizer_py PRIVATE GBPE_SLOT_ONLY=${_slot_only})
  endif()
  target_include_directories(gpu_bpe_tokenizer_py PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${UTF8PROC_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
  )
  target_link_libraries(gpu_bpe_tokenizer_py PRIVATE ${UTF8PROC_LIB} CUDA::cudart)
  target_compile_options(gpu_bpe_tokenizer_py PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --extended-lambda -lineinfo -Xcompiler=-Wall -Xcompiler=-fPIC>
  )

  # NOTE: scikit-build-core's `wheel.install-dir = "gpu_bpe_tokenizer"` already
  # places this destination inside the package directory; using "." here
  # avoids a doubled gpu_bpe_tokenizer/gpu_bpe_tokenizer/ layout.
  install(TARGETS gpu_bpe_tokenizer_py LIBRARY DESTINATION .)

  message(STATUS "Python bindings: ENABLED (target gpu_bpe_tokenizer_py)")
endif()
