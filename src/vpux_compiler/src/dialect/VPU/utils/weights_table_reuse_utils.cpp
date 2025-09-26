//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/weights_table_reuse_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/VPU/function_outlining_splitter.hpp"
#include "vpux/compiler/utils/options.hpp"

using namespace vpux;

WeightsTableReuseMode VPU::getWeightsTableReuseMode(mlir::Operation* op) {
    return static_cast<WeightsTableReuseMode>(VPU::getConstraint(op, VPU::WEIGHTS_TABLE_REUSE_MODE));
}

bool VPU::isWeightsTableReuseEnabled(mlir::Operation* op) {
    mlir::func::FuncOp func;
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
        func = funcOp;
    } else {
        func = op->getParentOfType<mlir::func::FuncOp>();
    }
    VPUX_THROW_WHEN(func == nullptr, "Cannot find parent function for operation '{0}'", op->getName());
    const auto weightsTableReuseMode = getWeightsTableReuseMode(func);
    return weightsTableReuseMode == WeightsTableReuseMode::ENABLED ||
           (weightsTableReuseMode == WeightsTableReuseMode::VF_ENABLED &&
            func->hasAttr(VPU::PureVerticalFusionRegionAttrName));
}
