//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

size_t vpux::VPUIP::CopyOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numDMAPorts = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();

    // TODO: Expose API to get arch from cost model
    const auto vpuDevice = vpux::VPU::getVPUDeviceType(module);
    return checked_cast<size_t>(
            getDMACost(getInput(), getOutput(), config::getArch(module), vpuDevice, costModel, numDMAPorts));
}

mlir::LogicalResult vpux::VPUIP::CopyOp::verify() {
    const auto op = getOperation();
    const auto inShape = getBoundedShape(getInput());
    const auto outShape = getBoundedShape(getOutput());

    if (inShape != outShape) {
        return errorAt(op, "Input shape '{0}' doesn't match output shape '{1}'", inShape, outShape);
    }

    return mlir::success();
}
