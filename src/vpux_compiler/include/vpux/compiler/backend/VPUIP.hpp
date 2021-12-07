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
#include "vpux_compiler.hpp"

#include <flatbuffers/flatbuffers.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

namespace vpux {
namespace VPUIP {

flatbuffers::DetachedBuffer exportToBlobGraphFile(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                                  const std::vector<PreProcessInfo>& preprocessInfo,
                                                  Logger log = Logger::global());

std::vector<char> exportToBlobELF(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                  const std::vector<PreProcessInfo>& preprocessInfo,
                                  Logger log = Logger::global());

}  // namespace VPUIP
}  // namespace vpux
