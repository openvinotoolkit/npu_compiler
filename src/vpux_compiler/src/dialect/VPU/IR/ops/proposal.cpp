//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::Type getAuxiliaryBufferType(mlir::Value classProbs) {
    constexpr int32_t proposalBoxSize = 10;         // see: sw_runtime_kernels/kernels/src/proposal.cpp (proposalBox)
    constexpr int32_t anchorsBuffElementSize = 16;  // see: sw_runtime_kernels / kernels / src / proposal.cpp (anchors)

    const auto inType = mlir::cast<vpux::NDTypeInterface>(classProbs.getType());
    const auto inShape = inType.getShape().raw();
    // [ num_batches, 2 * K, H, W ]
    const auto rank = inShape.size();
    VPUX_THROW_UNLESS(rank == 4, "Unsupported rank {0}", rank);
    const auto k = inShape[rank - 3] / 2;
    const auto h = inShape[rank - 2];
    const auto w = inShape[rank - 1];
    const auto numProposals = k * h * w;
    const auto auxiliaryBuffSize = alignValUp(numProposals * proposalBoxSize, static_cast<int64_t>(7)) +
                                   alignValUp(k * anchorsBuffElementSize, static_cast<int64_t>(7));
    const auto auxBuffType = mlir::RankedTensorType::get({auxiliaryBuffSize}, getUInt8Type(classProbs.getContext()));
    return auxBuffType;
}

void VPU::ProposalOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value classProbs,
                            mlir::Value bboxDelta, mlir::Value imageShape, IE::ProposalAttr proposalAttr) {
    const auto auxBuffType = getAuxiliaryBufferType(classProbs);
    auto auxBuffer = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, classProbs, bboxDelta, imageShape, auxBuffer, proposalAttr);
}

mlir::LogicalResult VPU::ProposalOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                      mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                      mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                      mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ProposalOpAdaptor proposal(operands, attrs, prop);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(proposal.getClassProbs().getType());

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{
            inType.getShape().front() * proposal.getProposalAttrs().getPostNmsTopN().getInt(), 5};
    const SmallVector<int64_t> probsShape{inType.getShape().front() *
                                          proposal.getProposalAttrs().getPostNmsTopN().getInt()};

    const auto outType = inType.changeShape(ShapeRef(outShape));
    const auto probsType = inType.changeShape(ShapeRef(probsShape));
    inferredReturnTypes.push_back(outType);
    inferredReturnTypes.push_back(probsType);

    return mlir::success();
}

llvm::LogicalResult VPU::ProposalOp::verify() {
    auto auxBufferType = mlir::cast<NDTypeInterface>(getAuxiliary().getType());
    auto expectedType = mlir::cast<NDTypeInterface>(getAuxiliaryBufferType(getClassProbs()));
    return VPU::compareTypes(getOperation()->getLoc(), auxBufferType, expectedType);
}

SmallVector<mlir::Value> VPU::ProposalOp::getAuxiliaryBuffers() {
    return {getAuxiliary()};
}
