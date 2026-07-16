//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassOptions.h>
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"
namespace vpux {
namespace HostExec {

//
// HostExecOptions
//

struct HostExecOptions : mlir::PassPipelineOptions<HostExecOptions> {
    BoolOption enablePipelinedCmdListRecording{
            *this, "enable-pipelined-cmd-list-recording",
            llvm::cl::desc("Enable pipelined command list recording and inference execution"),
            llvm::cl::init(vpux::HostExec::defaultEnablePipelinedCmdListRecording)};
    HostExecOptions() = default;

    template <class OtherOptions>
    HostExecOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }

    HostExecOptions(config::ArchKind) {
    }

    static std::unique_ptr<HostExecOptions> createFromString(StringRef options, config::ArchKind) {
        auto result = std::make_unique<HostExecOptions>();
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }
};

//
// Passes
//

std::unique_ptr<mlir::Pass> createSerializeELFToBinaryPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToLLVMUMDCallsPass(
        bool enablePipelinedCmdListRecording = vpux::HostExec::defaultEnablePipelinedCmdListRecording,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInlineMainBatchingCallsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPrepareHostFuncForAsyncExecutionPass(bool removeReturnValues = true,
                                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeMemRefCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReplaceAllocsWithSingleAllocAndViewsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSerializeNetworkMetadataPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExtractReturnShapesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlineDimOperationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGenerateExecutionContextFuncsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapFuncCallPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateDynamicShapesPass(Logger log = Logger::global());

void buildHostExecPipeline(mlir::OpPassManager& pm, bool enablePipelinedCmdListRecording,
                           Logger log = Logger::global());
void buildOutputShapePredictPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildBytecodeBackendPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// Registration
//

void registerHostExecPipelines();
void registerPasses();

}  // namespace HostExec
}  // namespace vpux
