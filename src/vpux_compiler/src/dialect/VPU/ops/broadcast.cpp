//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

SmallVector<int64_t> getResultShapeBidirectional(SmallVector<int64_t>& inShape, SmallVector<int64_t>& targetShape) {
    const auto targetPaddedRank = std::max(inShape.size(), targetShape.size());

    SmallVector<int64_t> resultShape(targetPaddedRank);

    while (inShape.size() < targetPaddedRank) {
        inShape.insert(inShape.begin(), 1);
    }

    while (targetShape.size() < targetPaddedRank) {
        targetShape.insert(targetShape.begin(), 1);
    }

    for (size_t i = 0; i < targetPaddedRank; ++i) {
        VPUX_THROW_UNLESS(inShape[i] == 1 || targetShape[i] == 1 || inShape[i] == targetShape[i],
                          "Broadcast incorrect target shape. Expecting either 1 or {0}. Got {1}", inShape[i],
                          targetShape[i]);

        resultShape[i] = std::max(inShape[i], targetShape[i]);
    }

    return resultShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::BroadcastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::BroadcastOpAdaptor broadcast(operands, attrs);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = broadcast.input().getType().cast<vpux::NDTypeInterface>();
    auto inShape = to_small_vector(inType.getShape().raw());
    auto targetShape = IE::constInputToData(loc, broadcast.target_shape()).getValue();
    const auto broadcastMode = broadcast.mode().getValue();

    SmallVector<int64_t> outShape;

    if (broadcastMode == IE::BroadcastType::NUMPY || broadcastMode == IE::BroadcastType::EXPLICIT) {
        outShape = targetShape;
    } else if (broadcastMode == IE::BroadcastType::BIDIRECTIONAL) {
        outShape = getResultShapeBidirectional(inShape, targetShape);
    }

    auto outType = inType.changeShape(Shape(outShape));
    outType = outType.changeDimsOrder(DimsOrder::fromNumDims(outShape.size()));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::BroadcastOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::BroadcastParamsBuilder builder(writer);

    MVCNN::BroadcastMode mvcnn_mode;

    if (this->mode() == IE::BroadcastType::NUMPY) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_NUMPY;
    } else if (this->mode() == IE::BroadcastType::BIDIRECTIONAL) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_BIDIRECTIONAL;
    } else if (this->mode() == IE::BroadcastType::EXPLICIT) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_EXPLICIT;
    } else {
        VPUX_THROW("Unsupported broadcast mode {0}", this->mode());
    }

    builder.add_mode(mvcnn_mode);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_BroadcastParams});
}
