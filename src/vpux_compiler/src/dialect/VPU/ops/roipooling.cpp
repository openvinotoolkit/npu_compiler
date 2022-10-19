//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ROIPoolingOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ROIPoolingOpAdaptor roiPooling(operands, attrs);
    if (mlir::failed(roiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputSize = parseIntArrayAttr<int64_t>(roiPooling.output_size());
    const auto inTypeFeatureMap = roiPooling.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiPooling.coords().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (outputSize.size() != 2) {
        return errorAt(loc, "Dimension of pooled size is expected to be equal to 2. Got {0}", outputSize.size());
    }

    if (outputSize[0] <= 0 && outputSize[1] <= 0) {
        return errorAt(loc, "Pooled size attributes pooled_h and pooled_w should should be positive.");
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inShapeCoord.raw()[0]);
    outputShape.push_back(inShapeFeatureMap.raw()[1]);
    outputShape.push_back(outputSize[0]);
    outputShape.push_back(outputSize[1]);

    const auto outType = inTypeFeatureMap.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ROIPoolingOp::serialize(EMU::BlobWriter& writer) {
    const float spatial_scale_val = static_cast<float>(spatial_scale().convertToDouble());
    uint32_t num_rois = checked_cast<uint32_t>(coords().getType().cast<mlir::ShapedType>().getShape()[0]);
    const auto output_size = parseIntArrayAttr<int64_t>(output_sizeAttr());

    MVCNN::ROIPoolingParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_roi_pooling_method(VPUIP::convertVPUXROIPoolingMethod2Int32(method()));
    builder.add_num_rois(num_rois);
    builder.add_pooled_h(checked_cast<uint32_t>(output_size[0]));
    builder.add_pooled_w(checked_cast<uint32_t>(output_size[1]));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIPoolingParams});
}
