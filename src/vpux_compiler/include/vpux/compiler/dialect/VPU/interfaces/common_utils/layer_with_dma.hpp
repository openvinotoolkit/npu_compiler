//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Operation.h>

#include <climits>

namespace vpux::VPU {

//
// LayerWithDmaInterfaceFwlm
// Attaches LayerWithDmaInterface to SW ops whose only requirement is FWLM enabled.
// Customize via a derived class when an op needs extra checks.
//

template <class MainOpType>
class LayerWithDmaInterfaceFwlm final :
        public IE::LayerWithDmaInterface::ExternalModel<LayerWithDmaInterfaceFwlm<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* origOp) const {
        auto module = origOp->getParentOfType<mlir::ModuleOp>();
        return config::getWorkloadManagementMode(module) == WorkloadManagementMode::FWLM_V1_PAGES;
    }
};

//
// LayerWithDmaInterfaceActivation
// Attaches LayerWithDmaInterface to SW activation ops that have dynamic tensors exceeding the size threshold.
//

template <class MainOpType>
class LayerWithDmaInterfaceActivation final :
        public IE::LayerWithDmaInterface::ExternalModel<LayerWithDmaInterfaceActivation<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* origOp) const {
        auto module = origOp->getParentOfType<mlir::ModuleOp>();
        if (config::getWorkloadManagementMode(module) != WorkloadManagementMode::FWLM_V1_PAGES) {
            return false;
        }
        if (vpux::IE::hasDynamicTensors(origOp) == false) {
            return false;
        }
        auto input = origOp->getOperand(0);
        const auto boundedShape = getBoundedShape(input);
        const int64_t threshold = 2 * 1024 * 1024;
        if (vpux::details::calcTotalShapeSize(boundedShape) > threshold) {
            return true;
        }
        return false;
    }
};

//
// LayerWithDmaInterfaceTopK
// Attaches LayerWithDmaInterface to IE::TopKOp; routes to VPU::TopKDmaOp when
// the input is large, K is small, and the sort axis is the innermost dimension.
//

class LayerWithDmaInterfaceTopK final :
        public IE::LayerWithDmaInterface::ExternalModel<LayerWithDmaInterfaceTopK, IE::TopKOp> {
public:
    bool isSupported(mlir::Operation* origOp) const {
        auto module = origOp->getParentOfType<mlir::ModuleOp>();
        if (config::getWorkloadManagementMode(module) != WorkloadManagementMode::FWLM_V1_PAGES) {
            return false;
        }
        if (config::getCompilationMode(origOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        auto topKOp = mlir::cast<IE::TopKOp>(origOp);
        if (!topKOp.getKValue().has_value()) {
            return false;
        }
        const auto input = topKOp.getInput();
        const auto inputShape = getShape(input).raw();
        const auto rank = static_cast<int64_t>(inputShape.size());
        const auto axis = topKOp.getAxis();
        const auto k = topKOp.getKValue().value();
        const auto isInnerAxis = (axis == (rank - 1));
        constexpr int64_t minAxisSize = 24000;
        constexpr int64_t maxK = 2 * 1024;
        return isInnerAxis && (inputShape[axis] >= minAxisSize) && (k < maxK);
    }
};

//
// LayerWithDmaInterfaceScatterUpdate
//

class LayerWithDmaInterfaceScatterUpdate final :
        public IE::LayerWithDmaInterface::ExternalModel<LayerWithDmaInterfaceScatterUpdate, IE::ScatterUpdateOp> {
public:
    bool isSupported(mlir::Operation* origOp) const {
        auto module = origOp->getParentOfType<mlir::ModuleOp>();
        if (config::getWorkloadManagementMode(module) != WorkloadManagementMode::FWLM_V1_PAGES) {
            return false;
        }
        if (config::getCompilationMode(origOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }
        const auto indicesType =
                mlir::cast<mlir::ShapedType>(mlir::cast<IE::ScatterUpdateOp>(origOp).getIndices().getType());
        if (!indicesType.hasStaticShape()) {
            return false;
        }
        const auto arch = config::getArch(origOp);
        const auto maxIndices = static_cast<int64_t>(VPU::getGatherDMAMaxIndicesListLength(arch));
        if (indicesType.getNumElements() > maxIndices) {
            return false;
        }
        return true;
    }
};

}  // namespace vpux::VPU
