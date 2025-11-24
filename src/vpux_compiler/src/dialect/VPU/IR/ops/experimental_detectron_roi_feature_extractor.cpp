//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult VPU::ExperimentalDetectronROIFeatureExtractorOp::inferReturnTypes(
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

    const auto outTypeFeatures = inType.changeShape(ShapeRef(outputShapeFeatures));
    inferredReturnTypes.push_back(outTypeFeatures);

    outputShapeROI.push_back(inputShapeROI[Dim(0)]);
    outputShapeROI.push_back(inputShapeROI[Dim(1)]);

    const auto outTypeROI = inType.changeShape(ShapeRef(outputShapeROI));
    inferredReturnTypes.push_back(outTypeROI);

    return mlir::success();
}

SmallVector<mlir::Value> VPU::ExperimentalDetectronROIFeatureExtractorOp::getAuxiliaryBuffers() {
    return {getReorderedRois(), getOriginalRoiMap(), getOutputRoisFeaturesTemp(), getLevels()};
}

mlir::LogicalResult VPU::ExperimentalDetectronROIFeatureExtractorOp::setAuxiliaryBuffers(
        ArrayRef<mlir::Value> buffers) {
    if (buffers.size() != 4 || llvm::any_of(buffers, [](mlir::Value buffer) {
            return buffer == nullptr;
        })) {
        return mlir::failure();
    }
    getReorderedRoisMutable().assign(buffers[0]);
    getOriginalRoiMapMutable().assign(buffers[1]);
    getOutputRoisFeaturesTempMutable().assign(buffers[2]);
    getLevelsMutable().assign(buffers[3]);
    return mlir::success();
}

SmallVector<mlir::Type> VPU::ExperimentalDetectronROIFeatureExtractorOp::getBufferTypes() {
    const auto shapeROI = getShape(getInputs()[0]);
    const auto shapeFeature = getShape(getInputs()[1]);
    const auto outputSize = getAttr().getOutputSize().getInt();

    const auto reorderedRoisBuffSize = static_cast<int32_t>(4 * shapeROI[Dim(0)]);
    const auto reorderedRoisType =
            mlir::RankedTensorType::get(reorderedRoisBuffSize, mlir::Float32Type::get(getContext()));

    const auto originalRoiMapBuffSize = static_cast<int32_t>(shapeROI[Dim(0)]);
    const auto originalRoiMapType = mlir::RankedTensorType::get(originalRoiMapBuffSize, getUInt32Type(getContext()));

    const auto outputRoisFeaturesTempBuffSize =
            static_cast<int32_t>(shapeFeature[Dim(1)] * outputSize * outputSize * shapeROI[Dim(0)]);
    const auto outputRoisFeaturesTempType =
            mlir::RankedTensorType::get(outputRoisFeaturesTempBuffSize, mlir::Float32Type::get(getContext()));

    const auto levelsBuffSize = static_cast<int32_t>(shapeROI[Dim(0)]);
    const auto levelsType = mlir::RankedTensorType::get(levelsBuffSize, getUInt32Type(getContext()));

    return {reorderedRoisType, originalRoiMapType, outputRoisFeaturesTempType, levelsType};
}
