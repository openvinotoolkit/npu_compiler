//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

//
// build
//

void VPUIP::ShapeCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ShapeRef shape) {
    build(builder, state, input, shape.raw());
}

void VPUIP::ShapeCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ArrayRef<int64_t> shape) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), shape));
}

void VPUIP::ShapeCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               mlir::ArrayAttr shape) {
    build(builder, state, input, shape, nullptr, nullptr, nullptr);
}

void VPUIP::ShapeCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               mlir::ArrayAttr shape, mlir::ArrayAttr explicitOutputShapes,
                               mlir::ArrayAttr explicitOutputOffsets) {
    build(builder, state, input, shape, explicitOutputShapes, explicitOutputOffsets, nullptr);
}

void VPUIP::ShapeCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, NDTypeInterface result,
                               mlir::Value input, mlir::ArrayAttr shape, mlir::ArrayAttr explicitOutputShapes,
                               mlir::ArrayAttr explicitOutputOffsets) {
    build(builder, state, result, input, shape, explicitOutputShapes, explicitOutputOffsets, nullptr);
}

void VPUIP::ShapeCastOp::build(mlir::OpBuilder&, mlir::OperationState& state, NDTypeInterface result, mlir::Value input,
                               mlir::ArrayAttr shape, mlir::ArrayAttr explicitOutputShapes,
                               mlir::ArrayAttr explicitOutputOffsets, mlir::ArrayAttr explicitOutputAlignment) {
    state.addOperands(input);
    state.addTypes(result);
    state.addAttribute("shape", shape);
    if (explicitOutputShapes != nullptr) {
        state.addAttribute("explicit_output_shapes", explicitOutputShapes);
    }
    if (explicitOutputOffsets != nullptr) {
        state.addAttribute("explicit_output_offsets", explicitOutputOffsets);
    }
    if (explicitOutputAlignment != nullptr) {
        state.addAttribute("explicit_output_alignment", explicitOutputAlignment);
    }
}

mlir::Value VPUIP::ShapeCastOp::getViewSource() {
    return getSource();
}

namespace {

bool isSupportedShapeCastType(VPUIP::ShapeCastOp op, mlir::Type inputType, mlir::Type outputType,
                              LogCb logCb = emptyLogCb) {
    auto inputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(inputType);
    auto outputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(outputType);
    if (inputNDType == nullptr || outputNDType == nullptr) {
        logCb(formatv("ShapeCast input and output must be ND types: in type = {0}, out type = {1}", inputType,
                      outputType));
        return false;
    }

    if (inputNDType.getDimsOrder() != outputNDType.getDimsOrder()) {
        logCb(formatv("Input dims order '{0}' doesn't match output dims order '{1}'", inputNDType.getDimsOrder(),
                      outputNDType.getDimsOrder()));
        return false;
    }
    if (inputNDType.getRank() != outputNDType.getRank()) {
        logCb(formatv("Input rank '{0}' doesn't match output rank '{1}'", inputNDType.getRank(),
                      outputNDType.getRank()));
        return false;
    }
    if (inputNDType.getElementType() != outputNDType.getElementType()) {
        logCb(formatv("Input element type '{0}' doesn't match output element type '{1}'", inputNDType.getElementType(),
                      outputNDType.getElementType()));
        return false;
    }
    if (inputNDType.getMemSpace() != outputNDType.getMemSpace()) {
        logCb(formatv("Input mem space '{0}' doesn't match output mem space '{1}'", inputNDType.getMemSpace(),
                      outputNDType.getMemSpace()));
        return false;
    }
    if (op.getExplicitOutputShapes().has_value() != op.getExplicitOutputOffsets().has_value()) {
        logCb(formatv("Only explicit output shape or offset is assigned"));
        return false;
    }
    if (op.getExplicitOutputAlignment().has_value()) {
        const auto explicitOutputAlignment = parseIntArrayAttr<int64_t>(op.getExplicitOutputAlignment().value());
        if (checked_cast<int64_t>(explicitOutputAlignment.size()) != outputNDType.getRank()) {
            logCb(formatv("Explicit output alignment rank '{0}' doesn't match output rank '{1}'",
                          explicitOutputAlignment.size(), outputNDType.getRank()));
            return false;
        }
    }
    return true;
}

bool isShapeCastSupportedForBackInfer(VPUIP::ShapeCastOp op) {
    auto origInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getSource().getType());
    auto origOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getResult().getType());
    if (origInputNDType == nullptr || origOutputNDType == nullptr) {
        return false;
    }

    // CompressConv uses ShapeCast to expose a padded DPU view, for example
    // 1x4xHxW -> 1x16xHxW, while the real memory still stores the compressed shape.
    // That is not a regular reshape and should not be crossed by generic back-inference.
    return origInputNDType.getNumElements() == origOutputNDType.getNumElements();
}

std::optional<vpux::NDTypeInterface> inferShapeCastOutputNDType(mlir::MLIRContext* ctx,
                                                                vpux::NDTypeInterface inputNDType, ShapeRef outShape,
                                                                config::ArchKind arch,
                                                                mlir::ArrayAttr explicitOutputShapes,
                                                                mlir::ArrayAttr explicitOutputOffsets,
                                                                mlir::ArrayAttr explicitOutputAlignment) {
    const auto hasExplicitOutputShapes = explicitOutputShapes != nullptr;
    const auto hasExplicitOutputOffsets = explicitOutputOffsets != nullptr;
    if (hasExplicitOutputShapes != hasExplicitOutputOffsets) {
        return std::nullopt;
    }

    auto outElemType = inputNDType.getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType) &&
        inputNDType.getDimsOrder() == DimsOrder::NHWC) {
        // NHWC per-axis quantization stores the quantized dimension in the
        // logical type. Recompute it after the shape cast so channel scales
        // still line up with the new logical shape.
        auto inferredElemType = vpux::inferPerAxisQuantizedTypeAfterShapeCastOrNull(inputNDType, outShape.raw());
        if (!inferredElemType.has_value()) {
            return std::nullopt;
        }
        outElemType = inferredElemType.value();
        if (!vpux::isSupportedElemTypeQuantization(outElemType, outShape)) {
            return std::nullopt;
        }
    }

    vpux::NDTypeInterface outNDType;
    const auto distributedIn = mlir::dyn_cast<VPU::DistributedTypeInterface>(inputNDType);
    if (distributedIn != nullptr && distributedIn.containsDistributedTypes()) {
        // Distributed ShapeCast must rebuild the distribution for the requested
        // output shape. Example: input 1x16x64x1 SEGMENTED over H can become
        // output 1x16x8x8 with the tiling moved to a shape-compatible axis.
        // When explicit output shapes/offsets are present, they are already the
        // intended per-cluster view and are copied into compute and memory view.
        const auto distBufType = mlir::cast<VPUIP::DistributedBufferType>(distributedIn.getDistributedTypes().front());
        const auto origDistribution = distBufType.getDistribution();

        VPU::DistributionInfoAttr newDistAttr;
        if (hasExplicitOutputShapes) {
            const auto mode = origDistribution.getMode().getValue();
            if (mode != VPU::DistributionMode::SEGMENTED) {
                return std::nullopt;
            }
            newDistAttr = VPU::DistributionInfoAttr::get(
                    ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                    origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                    origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(),
                    explicitOutputShapes, explicitOutputOffsets, explicitOutputShapes, explicitOutputOffsets,
                    origDistribution.getEqualMemoryAndComputeView(), origDistribution.getMemoryNumTiles());
        } else {
            auto distributedOutShape = Shape(outShape);
            if (auto sparseBufferType = mlir::dyn_cast<VPUIP::SparseBufferType>(distributedIn)) {
                if (auto seAttr = sparseBufferType.getSeAttr()) {
                    distributedOutShape = seAttr.backInferInputShape(outShape);
                }
            }
            if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                        distBufType, ShapeRef(distributedOutShape), inputNDType.getDimsOrder(), arch)) {
                return std::nullopt;
            }
            newDistAttr = VPUIP::getDistributedAttrAfterShapeCast<VPUIP::DistributedBufferType>(
                    distributedIn, outShape.raw(), arch, explicitOutputAlignment);
        }
        if (newDistAttr == nullptr) {
            return std::nullopt;
        }

        outNDType =
                outElemType == inputNDType.getElementType()
                        ? distributedIn.changeShapeForExplicitDistribution(outShape, newDistAttr)
                        : distributedIn.changeShapeElemTypeForExplicitDistribution(outShape, outElemType, newDistAttr);
    } else {
        outNDType = outElemType == inputNDType.getElementType()
                            ? inputNDType.changeShape(outShape)
                            : inputNDType.changeShapeElemType(outShape, outElemType);
    }

    auto strideUpdatedOutType = VPUIP::updateStridesForReshape(inputNDType, outNDType);
    if (mlir::failed(strideUpdatedOutType)) {
        return std::nullopt;
    }
    return strideUpdatedOutType.value();
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::ShapeCastOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr) {
        return std::nullopt;
    }
    if (!isShapeCastSupportedForBackInfer(*this)) {
        return std::nullopt;
    }

    const auto outShape = parseIntArrayAttr<int64_t>(getShape());
    if (newInputNDType.getNumElements() != ShapeRef(outShape).totalSize()) {
        return std::nullopt;
    }

    const auto arch = config::getArch(getOperation());
    const auto explicitOutputShapes = getExplicitOutputShapes().value_or(nullptr);
    const auto explicitOutputOffsets = getExplicitOutputOffsets().value_or(nullptr);
    const auto explicitOutputAlignment = getExplicitOutputAlignment().value_or(nullptr);
    const auto outputNDType =
            inferShapeCastOutputNDType(getContext(), newInputNDType, ShapeRef(outShape), arch, explicitOutputShapes,
                                       explicitOutputOffsets, explicitOutputAlignment);
    if (!outputNDType.has_value()) {
        return std::nullopt;
    }
    const auto outputType = mlir::cast<mlir::Type>(outputNDType.value());
    if (!isSupportedShapeCastType(*this, newInputType, outputType)) {
        return std::nullopt;
    }
    return outputType;
}

std::optional<mlir::Type> VPUIP::ShapeCastOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    auto desiredOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(desiredOutputType);
    if (desiredOutputNDType == nullptr) {
        return std::nullopt;
    }
    if (!isShapeCastSupportedForBackInfer(*this)) {
        return std::nullopt;
    }
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(getSource().getType());
    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(getResult().getType());
    const auto inputType =
            VPUIP::inferReshapeInputType(getOperation(), desiredOutputNDType, origInputNDType, origOutputNDType);
    if (!inputType.has_value() || !isSupportedShapeCastType(*this, inputType.value(), desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

mlir::LogicalResult vpux::VPUIP::ShapeCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedShapeCastType(*this, getSource().getType(), getResult().getType(), logCb));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties props, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::ShapeCastOpAdaptor shapeCast(operands, attrs, props);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }
    const auto arch = config::getArch(mlir::isa<mlir::BlockArgument>(operands[0])
                                              ? operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>()
                                              : operands[0].getDefiningOp());
    const auto inType = mlir::cast<vpux::NDTypeInterface>(shapeCast.getSource().getType());
    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());

    const auto explicitOutputShapes = shapeCast.getExplicitOutputShapes().value_or(nullptr);
    const auto explicitOutputOffsets = shapeCast.getExplicitOutputOffsets().value_or(nullptr);
    const auto explicitOutputAlignment = shapeCast.getExplicitOutputAlignment().value_or(nullptr);
    const auto outType = inferShapeCastOutputNDType(ctx, inType, ShapeRef(outShape), arch, explicitOutputShapes,
                                                    explicitOutputOffsets, explicitOutputAlignment);
    if (!outType.has_value()) {
        return mlir::failure();
    }
    inferredReturnTypes.push_back(outType.value());

    return mlir::success();
}
