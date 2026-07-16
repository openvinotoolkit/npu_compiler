//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_compiler_loader.hpp
 * @brief Define CompilerLoader which loads and manages the VPUX compiler instance
 */

#pragma once

#include "vpux/compiler/icompiler.hpp"

#include <memory>
#include <mutex>

namespace VPUXDriverCompiler {

class CompilerLoader {
public:
    CompilerLoader() = default;
    ~CompilerLoader() = default;

    CompilerLoader(const CompilerLoader&) = delete;
    CompilerLoader(CompilerLoader&&) = delete;
    CompilerLoader& operator=(const CompilerLoader&) = delete;
    CompilerLoader& operator=(CompilerLoader&&) = delete;

    // Loads the compiler library and resolves its entry point.
    void ensureLoaded();

    std::shared_ptr<vpux::ICompiler> getCompiler();

private:
    static std::shared_ptr<vpux::ICompiler> createCompiler();

    std::shared_ptr<vpux::ICompiler> _compiler = nullptr;
    std::once_flag _compilerInitOnce;
};

}  // namespace VPUXDriverCompiler
