//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NonMaxSuppressionUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::NMSParamsBuilder builder(writer);

    builder.add_box_encoding(box_encoding() == IE::BoxEncodingType::CENTER);
    builder.add_sort_result_descending(sort_result_descending());
    builder.add_max_output_boxes_per_class(checked_cast<int32_t>(max_output_boxes_per_class_value()));
    builder.add_iou_threshold(static_cast<float>(iou_threshold_value().convertToDouble()));
    builder.add_score_threshold(static_cast<float>(score_threshold_value().convertToDouble()));
    builder.add_soft_nms_sigma(static_cast<float>(soft_nms_sigma_valueAttr().getValueAsDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NMSParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseNonMaxSuppression(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                 ArrayRef<mlir::Value> outputs,
                                                                 const MVCNN::UPALayerTask* task) {
    const auto params = task->softLayerParams_as_NMSParams();
    const auto boxEncoding = params->box_encoding() ? IE::BoxEncodingType::CENTER : IE::BoxEncodingType::CORNER;
    const auto boxEncodingAttr = IE::BoxEncodingTypeAttr::get(_ctx, boxEncoding);
    const auto sortResultDescending = params->sort_result_descending() ? mlir::UnitAttr::get(_ctx) : nullptr;
    const auto maxOutBoxexPerClassAttr = getIntAttr(_ctx, params->max_output_boxes_per_class());
    const auto iouThresholdAttr = getFPAttr(_ctx, params->iou_threshold());
    const auto scoreThresholdAttr = getFPAttr(_ctx, params->score_threshold());
    const auto softNMSSigmaAttr = getFPAttr(_ctx, params->soft_nms_sigma());

    return builder.create<VPUIP::NonMaxSuppressionUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0], outputs[1], outputs[2], boxEncodingAttr,
            sortResultDescending, maxOutBoxexPerClassAttr, iouThresholdAttr, scoreThresholdAttr, softNMSSigmaAttr);
}
