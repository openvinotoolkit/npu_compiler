// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    constexpr int KER_MISC = 1;
    constexpr int KER_WSZ = 1024 + KER_MISC;  // [KB]
    return mlir::RankedTensorType::get({1, 1, 1, KER_WSZ * 1024}, getUInt8Type(module.getContext()));
}

}  // namespace

//
// build
//

void vpux::VPU::PadDmaOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                mlir::ArrayAttr padsBeginAttr, mlir::ArrayAttr padsEndAttr,
                                mlir::FloatAttr padValueAttr, IE::PadModeAttr mode) {
    auto module = getModuleOp(odsBuilder);
    const auto auxType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxType);
    build(odsBuilder, odsState, input, auxBuffer, padsBeginAttr, padsEndAttr, padValueAttr, mode,
          /*multiClusterStrategy=*/nullptr);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::PadDmaOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PadDmaOpAdaptor pad(operands, attrs, prop);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    if (!pad.getPadsBeginAttr().has_value() || !pad.getPadsEndAttr().has_value()) {
        return mlir::emitError(loc, "VPU::PadDmaOp requires static pads_begin_attr and pads_end_attr");
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(pad.getInput().getType());
    const auto padsBegin = parseIntArrayAttr<int64_t>(pad.getPadsBeginAttr().value());
    const auto padsEnd = parseIntArrayAttr<int64_t>(pad.getPadsEndAttr().value());

    auto outputType = inType.pad(ShapeRef(padsBegin), ShapeRef(padsEnd));
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}
