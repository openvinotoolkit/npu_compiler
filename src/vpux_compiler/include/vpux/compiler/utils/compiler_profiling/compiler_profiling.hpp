//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadicDetails.h>
#include <llvm/Support/raw_ostream.h>

#include <variant>

namespace mlir {
class MLIRContext;
}

namespace vpux::compiler_profiling {

struct MlirProfilerOptions {
    static constexpr llvm::StringLiteral key() {
        return "mlir-profiler";
    }

    std::string path{};
};

struct CallgrindProfilerOptions {
    static constexpr llvm::StringLiteral key() {
        return "callgrind";
    }

    std::string regex{};
    bool separateDumps{false};
};

struct VTuneProfilerOptions {
    static constexpr llvm::StringLiteral key() {
        return "vtune";
    }

    std::string regex{};
};

class CompilerProfiler {
    std::variant<std::monostate, MlirProfilerOptions, CallgrindProfilerOptions, VTuneProfilerOptions> _tool{
            std::monostate{}};

public:
    static CompilerProfiler createFromString(llvm::StringRef param);
    void configureActionHandler(mlir::MLIRContext* ctx) const;

    llvm::StringRef name() const;
    bool empty() const {
        return std::holds_alternative<std::monostate>(_tool);
    }
};

}  // namespace vpux::compiler_profiling

namespace llvm {
template <>
struct format_provider<::vpux::compiler_profiling::CompilerProfiler> {
    static void format(const ::vpux::compiler_profiling::CompilerProfiler& profiler, raw_ostream& stream, StringRef) {
        stream << profiler.name();
    }
};
}  // namespace llvm
