//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/scale_shift_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult isBeneficialConvertScaleShiftToDW(IE::ScaleShiftOp scaleShiftOp, Logger log) {
    if (scaleShiftOp.getBiases() != nullptr) {
        if (mlir::failed(IE::getConstParentOp(scaleShiftOp.getBiases()))) {
            log.nest().trace("Cannot convert ScaleShift to DW, since it has non constant biases");
            return mlir::failure();
        }
    }

    // If sub-graph like: input -> SHAVEs -> ScaleShift -> Conv. It is better not convert to DWConv.
    // If layer before ScaleShift is NCE op or with NHWC layout. It should convert to DWConv.
    auto onlySupportNHWCLayout = [&](mlir::Operation* op) -> bool {
        if (auto iface = mlir::dyn_cast_or_null<IE::LayoutInfoOpInterface>(op)) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
            return orderInfo.hasChanges();
        }
        return false;
    };

    auto prevOp = scaleShiftOp.getInput().getDefiningOp();
    if (onlySupportNHWCLayout(prevOp)) {
        return mlir::success();
    }

    return mlir::success();
}
}  // namespace IE
}  // namespace vpux
