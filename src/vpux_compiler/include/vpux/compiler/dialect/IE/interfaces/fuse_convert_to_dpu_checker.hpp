//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

/*
   Class for fusion of Convert to DPU op checker
*/
class FuseConvertToDPUCheckerBase {
public:
    FuseConvertToDPUCheckerBase() = default;
    FuseConvertToDPUCheckerBase(const FuseConvertToDPUCheckerBase&) = default;
    FuseConvertToDPUCheckerBase(FuseConvertToDPUCheckerBase&&) = default;
    virtual ~FuseConvertToDPUCheckerBase() = default;

    FuseConvertToDPUCheckerBase& operator=(const FuseConvertToDPUCheckerBase&) = default;
    FuseConvertToDPUCheckerBase& operator=(FuseConvertToDPUCheckerBase&&) = default;

    virtual bool isFusionToParentDPUOpSupported(mlir::Operation* /*dpuOp*/, Logger /*log*/) const {
        return true;
    };
};

/*
   Find right class to verify whether fusion of Convert F16 -> F32 to parent DPU is feasible
*/
std::unique_ptr<FuseConvertToDPUCheckerBase> createFuseConvertToDPUChecker(vpux::VPU::ArchKind arch);

}  // namespace IE
}  // namespace vpux
