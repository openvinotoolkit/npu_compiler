//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

SmallVector<mlir::Type> getAuxiliaryBufferTypes(mlir::ValueRange inputs,
                                                IE::ExperimentalDetectronROIFeatureExtractorAttr attr) {
    const auto shapeROI = getShape(inputs[0]);
    const auto shapeFeature = getShape(inputs[1]);
    const auto outputSize = attr.getOutputSize().getInt();

    const auto reorderedRoisBuffSize = static_cast<int32_t>(4 * shapeROI[Dim(0)]);
    const auto reorderedRoisType =
            mlir::RankedTensorType::get(reorderedRoisBuffSize, mlir::Float32Type::get(attr.getContext()));

    const auto originalRoiMapBuffSize = static_cast<int32_t>(shapeROI[Dim(0)]);
    const auto originalRoiMapType =
            mlir::RankedTensorType::get(originalRoiMapBuffSize, getUInt32Type(attr.getContext()));

    const auto outputRoisFeaturesTempBuffSize =
            static_cast<int32_t>(shapeFeature[Dim(1)] * outputSize * outputSize * shapeROI[Dim(0)]);
    const auto outputRoisFeaturesTempType =
            mlir::RankedTensorType::get(outputRoisFeaturesTempBuffSize, mlir::Float32Type::get(attr.getContext()));

    const auto levelsBuffSize = static_cast<int32_t>(shapeROI[Dim(0)]);
    const auto levelsType = mlir::RankedTensorType::get(levelsBuffSize, getUInt32Type(attr.getContext()));

    return {reorderedRoisType, originalRoiMapType, outputRoisFeaturesTempType, levelsType};
}

void VPU::ExperimentalDetectronROIFeatureExtractorOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                                            mlir::ValueRange inputs,
                                                            IE::ExperimentalDetectronROIFeatureExtractorAttr attr) {
    const auto auxBufferTypes = getAuxiliaryBufferTypes(inputs, attr);
    VPUX_THROW_WHEN(auxBufferTypes.size() != 4, "Expected 4 auxiliary buffer types, got {0}", auxBufferTypes.size());
    auto reorderedRois = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[0]);
    auto originalRoiMap = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[1]);
    auto outputRoisFeaturesTemp = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[2]);
    auto levels = VPU::createAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[3]);
    build(odsBuilder, odsState, inputs, reorderedRois, originalRoiMap, outputRoisFeaturesTemp, levels, attr);
}

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

llvm::LogicalResult VPU::ExperimentalDetectronROIFeatureExtractorOp::verify() {
    auto expectedAuxBuffTypes = getAuxiliaryBufferTypes(getInputs(), getAttr());
    if (expectedAuxBuffTypes.size() != 4) {
        return errorAt(getOperation(), "Expected four reference auxiliary buffer types, but got {0}",
                       expectedAuxBuffTypes.size());
    }
    auto loc = getOperation()->getLoc();
    if (mlir::failed(VPU::compareTypes(loc, getReorderedRois().getType(), expectedAuxBuffTypes[0]))) {
        return errorAt(getOperation(), "Invalid reordered ROIs auxiliary buffer");
    }
    if (mlir::failed(VPU::compareTypes(loc, getOriginalRoiMap().getType(), expectedAuxBuffTypes[1]))) {
        return errorAt(getOperation(), "Invalid original ROI map auxiliary buffer");
    }
    if (mlir::failed(VPU::compareTypes(loc, getOutputRoisFeaturesTemp().getType(), expectedAuxBuffTypes[2]))) {
        return errorAt(getOperation(), "Invalid output ROI features auxiliary buffer");
    }
    if (mlir::failed(VPU::compareTypes(loc, getLevels().getType(), expectedAuxBuffTypes[3]))) {
        return errorAt(getOperation(), "Invalid levels auxiliary buffer");
    }
    return mlir::success();
}

SmallVector<mlir::Value> VPU::ExperimentalDetectronROIFeatureExtractorOp::getAuxiliaryBuffers() {
    return {getReorderedRois(), getOriginalRoiMap(), getOutputRoisFeaturesTemp(), getLevels()};
}
