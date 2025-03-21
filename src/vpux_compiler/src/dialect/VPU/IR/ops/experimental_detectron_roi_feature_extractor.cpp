//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExperimentalDetectronROIFeatureExtractorOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExperimentalDetectronROIFeatureExtractorOpAdaptor exp(operands, attrs, prop);
    if (mlir::failed(exp.verify(loc))) {
        return mlir::failure();
    }

    size_t totalInputs = exp.getInputs().size();
    if (totalInputs > 4) {
        return errorAt(loc, "The total number of supported inputs is 4. Got {0}. ", totalInputs);
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(exp.getInputs().front().getType());
    const auto outputSize = exp.getAttr().getOutputSize().getInt();

    SmallVector<int64_t> outputShapeFeatures, outputShapeROI;

    const auto inputShapeROI = getShape(exp.getInputs()[0]);
    const auto inputShapeFeatures = getShape(exp.getInputs()[1]);

    if (inputShapeFeatures.size() != 4) {
        return errorAt(loc, "Feature inputs shape should be 4D. Got {0}D", inputShapeFeatures.size());
    }

    outputShapeFeatures.push_back(inputShapeROI[Dim(0)]);
    outputShapeFeatures.push_back(inputShapeFeatures[Dim(1)]);
    outputShapeFeatures.push_back(outputSize);
    outputShapeFeatures.push_back(outputSize);

    const auto outTypeFeatures = inType.changeShape(Shape(outputShapeFeatures));
    inferredReturnTypes.push_back(outTypeFeatures);

    outputShapeROI.push_back(inputShapeROI[Dim(0)]);
    outputShapeROI.push_back(inputShapeROI[Dim(1)]);

    const auto outTypeROI = inType.changeShape(Shape(outputShapeROI));
    inferredReturnTypes.push_back(outTypeROI);

    return mlir::success();
}
