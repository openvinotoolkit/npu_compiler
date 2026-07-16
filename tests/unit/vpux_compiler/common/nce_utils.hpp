//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

// see src/vpux_translate_utils/src/hwtest/hwtest_utils.cpp for a more detailed implementation of this function
// outStart and outEnd represent the {W, H, C} dimensions
inline vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder& builder, vpux::ArrayRef<int64_t> outStart,
                                              vpux::ArrayRef<int64_t> outEnd) {
    auto pad = vpux::VPU::getPaddingAttr(builder.getContext(), 0, 0, 0, 0);

    return builder.create<vpux::VPUIP::DPUTaskOp>(builder.getUnknownLoc(), vpux::getIntArrayAttr(builder, outStart),
                                                  vpux::getIntArrayAttr(builder, outEnd), pad,
                                                  vpux::VPU::MPEMode::CUBOID_16x16);
}
