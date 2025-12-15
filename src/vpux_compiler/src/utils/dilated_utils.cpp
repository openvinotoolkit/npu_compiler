//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/dilated_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

NDTypeInterface vpux::getDilatedType(vpux::NDTypeInterface type, vpux::ShapeRef dilations) {
    const auto targetRank = 4;
    VPUX_THROW_UNLESS(type.getRank() == targetRank, "Got invalid tensor rank '{0}'", targetRank);

    const auto origShape = type.getShape();

    const auto OC = origShape[vpux::Dims4D::Filter::OC];
    const auto IC = origShape[vpux::Dims4D::Filter::IC];
    const auto KY = origShape[vpux::Dims4D::Filter::KY];
    const auto KX = origShape[vpux::Dims4D::Filter::KX];

    // Calculate dilated kernel shape
    const auto dKY = KY + (KY - 1) * (dilations[Dim(0)] - 1);
    const auto dKX = KX + (KX - 1) * (dilations[Dim(1)] - 1);

    const auto dilatedShape = Shape({OC, IC, dKY, dKX});

    const auto newType = type.changeShape(dilatedShape);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

bool vpux::isSEPDWConv(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::NCEDepthConvolutionOp>(op)) {
        return false;
    }
    const auto sparseInputTensor = mlir::dyn_cast<VPU::SparseTensorType>(op->getOperand(0).getType());
    if (sparseInputTensor == nullptr) {
        return false;
    }
    auto seAttr = sparseInputTensor.getSeAttr();
    return seAttr != nullptr;
}
