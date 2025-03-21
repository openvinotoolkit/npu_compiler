//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/fuse_convert_to_dpu_checker.hpp"

namespace vpux::IE::arch37xx {

/*
   Class for Convert to DPU fusion checker for NPU37XX & NPU40XX
*/
class FuseConvertToDPUChecker : public FuseConvertToDPUCheckerBase {
public:
    bool isFusionToParentDPUOpSupported(mlir::Operation* dpuOp, Logger log) const override;
};

}  // namespace vpux::IE::arch37xx
