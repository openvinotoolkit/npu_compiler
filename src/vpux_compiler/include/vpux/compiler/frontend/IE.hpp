//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

#include <cpp/ie_cnn_network.h>
#include <vpux_compiler.hpp>

namespace vpux {
namespace IE {

mlir::OwningModuleRef importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet,
                                    std::vector<VPUXPreProcessInfo::Ptr>& preProcInfo, bool sharedConstants,
                                    mlir::TimingScope& rootTiming, bool enableProfiling, const Config& config,
                                    Logger log = Logger::global());

std::unordered_set<std::string> queryNetwork(const InferenceEngine::CNNNetwork& cnnNet,
                                             std::vector<VPUXPreProcessInfo::Ptr>& preProcInfo,
                                             mlir::TimingScope& rootTiming, const Config& config,
                                             Logger log = Logger::global());

}  // namespace IE
}  // namespace vpux
