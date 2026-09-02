//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    constexpr int KER_MISC = 1;               // miscellaneous small allocs (dma-descriptors, sync-buffs) [KB]
    constexpr int KER_WSZ = 1024 + KER_MISC;  // kernel cmx workspace size [KB]
    return mlir::RankedTensorType::get({1, 1, 1, KER_WSZ * 1024}, getUInt8Type(module.getContext()));
}

}  // namespace

void VPU::TopKDmaOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                           mlir::IntegerAttr kValue, mlir::IntegerAttr axis, IE::TopKModeAttr mode,
                           IE::TopKSortTypeAttr sort, mlir::TypeAttr indexElementType,
                           VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    auto module = getModuleOp(odsBuilder);
    const auto auxBuffType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, input, auxBuffer, kValue, axis, mode, sort, indexElementType, multiClusterStrategy);
}

mlir::LogicalResult vpux::VPU::TopKDmaOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TopKDmaOpAdaptor topK(operands, attrs, prop);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(topK.getInput().getType());
    const auto inputShape = inType.getShape().raw();
    const auto kValue = topK.getKValueAttr().getValue().getSExtValue();

    SmallVector<int64_t> outShape(inputShape);
    int64_t axis = topK.getAxis();
    outShape[axis] = kValue;

    auto outValueType =
            mlir::RankedTensorType::get(outShape, inType.getElementType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outValueType);

    auto outIndexType =
            mlir::RankedTensorType::get(outShape, topK.getIndexElementType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outIndexType);

    return mlir::success();
}
