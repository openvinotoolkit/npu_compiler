//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

size_t vpux::VPUIP::CopyOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

    // TODO: Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel, numDMAPorts));
}

mlir::LogicalResult vpux::VPUIP::CopyOp::verify() {
    const auto op = getOperation();
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape();
    const auto outShape = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).getShape();

    if (inShape != outShape) {
        return errorAt(op, "Input shape '{0}' doesn't match output shape '{1}'", inShape, outShape);
    }

    return mlir::success();
}
