//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/type/float16.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::IncrementalSDPAOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::IncrementalSDPAOpAdaptor incrementalSdpa(operands, attrs, prop);
    if (mlir::failed(incrementalSdpa.verify(loc))) {
        return mlir::failure();
    }

    inferredReturnTypes.push_back(incrementalSdpa.getInputPartialOutput().getType());
    inferredReturnTypes.push_back(incrementalSdpa.getInputRunningMax().getType());
    inferredReturnTypes.push_back(incrementalSdpa.getInputRunningSum().getType());

    return mlir::success();
}

namespace {

mlir::Value createAuxiliaryBuffer(mlir::OpBuilder& rewriter, mlir::Location loc, ArrayRef<int64_t> shape) {
    const auto auxIndicesType = mlir::RankedTensorType::get(shape, getFp16Type(rewriter.getContext()));
    return Const::createConst(rewriter, appendLoc(loc, "auxiliaryBuffer"), auxIndicesType,
                              ArrayRef<type::float16>{0.0});
}

}  // namespace

void vpux::VPU::IncrementalSDPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                         mlir::Value query, mlir::Value key, mlir::Value value,
                                         mlir::Value inputPartialOutput, mlir::Value inputRunningMax,
                                         mlir::Value inputRunningSum, mlir::Value attentionMask, mlir::Value scale) {
    auto queryShape = getShape(query);
    auto keyShape = getShape(key);

    VPUX_THROW_UNLESS(queryShape.size() >= 2 && keyShape.size() >= 2,
                      "Expected rank of Query and Key tensors to be at least 2D, got: {0}, {1}", queryShape, keyShape);
    auto sourceSeqLen = keyShape[Dim(keyShape.size() - 2)];
    auto bufferShape = to_small_vector(queryShape);
    bufferShape[bufferShape.size() - 1] = sourceSeqLen;

    auto auxBuffer = createAuxiliaryBuffer(odsBuilder, odsState.location, bufferShape);

    build(odsBuilder, odsState, query, key, value, auxBuffer, inputPartialOutput, inputRunningMax, inputRunningSum,
          attentionMask, scale);
}
