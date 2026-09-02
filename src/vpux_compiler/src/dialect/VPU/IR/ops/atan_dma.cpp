//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    constexpr int KER_WSZ = 512;  // kernel cmx workspace size [KB]
    return mlir::RankedTensorType::get({1, 1, 1, KER_WSZ * 1024}, getUInt8Type(module.getContext()));
}

}  // namespace

mlir::LogicalResult vpux::VPU::AtanDmaOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AtanDmaOpAdaptor atan(operands, attrs, prop);
    if (mlir::failed(atan.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = atan.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

void vpux::VPU::AtanDmaOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input) {
    auto loc = odsState.location;
    auto module = getModuleOp(odsBuilder);
    auto auxBufferType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, loc, auxBufferType);
    build(odsBuilder, odsState, input.getType(), input, auxBuffer, nullptr);
}

mlir::LogicalResult vpux::VPU::AtanDmaOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}
