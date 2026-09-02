//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/compiler_profiling.hpp"
#include "vpux/compiler/utils/compiler_profiling/callgrind_profiler.hpp"
#include "vpux/compiler/utils/compiler_profiling/mlir_profiler.hpp"
#include "vpux/compiler/utils/compiler_profiling/vtune_profiler.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/MLIRContext.h>

namespace {
template <typename... Callables>
struct Overloaded : Callables... {
    using Callables::operator()...;
};
template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

struct MlirProfilerOptionsToParse : mlir::PassPipelineOptions<MlirProfilerOptionsToParse> {
    vpux::StrOption path{*this, "path", llvm::cl::desc("Path to the output file")};
};

struct CallgrindProfilerOptionsToParse : mlir::PassPipelineOptions<CallgrindProfilerOptionsToParse> {
    vpux::StrOption regex{*this, "regex", llvm::cl::desc("Regex to filter actions")};
    vpux::BoolOption separateDumps{*this, "separate-dumps",
                                   llvm::cl::desc("Whether to create separate dumps for each action"),
                                   llvm::cl::init(false)};
};

struct VTuneProfilerOptionsToParse : mlir::PassPipelineOptions<VTuneProfilerOptionsToParse> {
    vpux::StrOption regex{*this, "regex", llvm::cl::desc("Regex pattern to match actions")};
};

static constexpr llvm::StringLiteral NONE_TOOL = "none";
}  // namespace
namespace vpux::compiler_profiling {
CompilerProfiler CompilerProfiler::createFromString(llvm::StringRef param) {
    CompilerProfiler profilerTool;
    if (param.empty()) {
        return profilerTool;
    }

    // An example value for param can be:
    // "compiler-profiler-tool=mlir-profiler='path=dump.json'" where
    // compiler-profiler-tool is already parsed

    auto [toolName, toolOptions] = param.split('=');
    if (toolName == NONE_TOOL) {
        return profilerTool;
    }

    toolOptions = toolOptions.trim('\'');  // remove quotes if present

    if (toolName == MlirProfilerOptions::key()) {
        auto options = MlirProfilerOptionsToParse::createFromString(toolOptions);
        VPUX_THROW_WHEN(options == nullptr, "Failed to parse MLIR profiler tool options: {0}", toolOptions);
        profilerTool._tool = MlirProfilerOptions{options->path};
    }

    if (toolName == CallgrindProfilerOptions::key()) {
        auto options = CallgrindProfilerOptionsToParse::createFromString(toolOptions);
        VPUX_THROW_WHEN(options == nullptr, "Failed to parse Callgrind profiler tool options: {0}", toolOptions);
        profilerTool._tool = CallgrindProfilerOptions{options->regex, options->separateDumps};
    }

    if (toolName == VTuneProfilerOptions::key()) {
        auto options = VTuneProfilerOptionsToParse::createFromString(toolOptions);
        VPUX_THROW_WHEN(options == nullptr, "Failed to parse VTune profiler tool options: {0}", toolOptions);
        profilerTool._tool = VTuneProfilerOptions{options->regex};
    }

    VPUX_THROW_WHEN(profilerTool.empty(), "Unknown compiler profiler tool: {0}", toolName);
    return profilerTool;
}

void CompilerProfiler::configureActionHandler(mlir::MLIRContext* ctx) const {
    auto& handler = getActionHandler(*ctx);
    std::visit(Overloaded{[](const std::monostate&) {},
                          [&](const MlirProfilerOptions& options) {
                              handler.registerObserver(std::make_unique<MlirProfiler>(options.path));
                          },
                          [&](const CallgrindProfilerOptions& options) {
                              handler.registerObserver(
                                      std::make_unique<CallgrindProfiler>(options.regex, options.separateDumps));
                          },
                          [&](const VTuneProfilerOptions& options) {
                              handler.registerObserver(std::make_unique<VTuneProfiler>(options.regex));
                          }},
               _tool);
}

llvm::StringRef CompilerProfiler::name() const {
    return std::visit(Overloaded{[](const std::monostate&) -> llvm::StringRef {
                                     return NONE_TOOL;
                                 },
                                 [](const MlirProfilerOptions&) -> llvm::StringRef {
                                     return MlirProfilerOptions::key();
                                 },
                                 [](const CallgrindProfilerOptions&) -> llvm::StringRef {
                                     return CallgrindProfilerOptions::key();
                                 },
                                 [](const VTuneProfilerOptions&) -> llvm::StringRef {
                                     return VTuneProfilerOptions::key();
                                 }},
                      _tool);
}

}  // namespace vpux::compiler_profiling
