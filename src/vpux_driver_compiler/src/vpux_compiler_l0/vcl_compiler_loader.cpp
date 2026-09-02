//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_compiler_loader.hpp"

#include <openvino/util/file_util.hpp>
#include <openvino/util/shared_object.hpp>

#include <memory>
#include <mutex>

namespace VPUXDriverCompiler {

namespace {
constexpr auto CREATE_NPU_COMPILER_FUNC_NAME = "CreateNPUCompiler";
constexpr auto COMPILER_LIBRARY_NAME = "openvino_intel_npu_compiler";
using CreateFuncT = void (*)(std::shared_ptr<vpux::ICompiler>&);

// Loads the compiler library once per process and returns its create-compiler entry point.
CreateFuncT loadCreateCompilerFunc() {
    // compilerSO and createNPUCompilerFunc are static and kept alive for the lifetime of the process
    // to ensure the compiler library remains loaded and the function pointer stays valid.
    static std::shared_ptr<void> compilerSO = nullptr;
    static CreateFuncT createNPUCompilerFunc = nullptr;

    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    if (!compilerSO || !createNPUCompilerFunc) {
        compilerSO = ov::util::load_shared_object(
                ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), COMPILER_LIBRARY_NAME));
        createNPUCompilerFunc =
                reinterpret_cast<CreateFuncT>(ov::util::get_symbol(compilerSO, CREATE_NPU_COMPILER_FUNC_NAME));
    }
    return createNPUCompilerFunc;
}
}  // namespace

void CompilerLoader::ensureLoaded() {
    try {
        (void)loadCreateCompilerFunc();
    } catch (const std::exception& err) {
        VPUX_THROW("Failed to load compiler library: {0}", err.what());
    }
}

std::shared_ptr<vpux::ICompiler> CompilerLoader::createCompiler() try {
    auto createNPUCompilerFunc = loadCreateCompilerFunc();
    std::shared_ptr<vpux::ICompiler> compiler;
    createNPUCompilerFunc(compiler);
    return compiler;
} catch (const std::exception& err) {
    VPUX_THROW("Failed to initialize compiler: {0}", err.what());
}

std::shared_ptr<vpux::ICompiler> CompilerLoader::getCompiler() {
    std::call_once(_compilerInitOnce, [this]() {
        _compiler = createCompiler();
    });
    return _compiler;
}
}  // namespace VPUXDriverCompiler
