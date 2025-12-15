//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RollOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RollOpAdaptor roll(operands, attrs, prop);
    if (mlir::failed(roll.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = mlir::cast<vpux::NDTypeInterface>(roll.getData().getType());

    auto getConstSource = [](mlir::Value value) {
        // If the value comes from an UnrolledTypeOp,
        // update the value to its input for the next check.
        if (auto unrollOp = value.getDefiningOp<VPU::UnrolledTypeOp>()) {
            value = unrollOp.getInput();
        }
        while (auto parentOp = value.getDefiningOp<VPU::CopyOp>()) {
            value = parentOp->getOperand(0);
        }
        return value.getDefiningOp<Const::DeclareOp>();
    };
    auto constShiftSource = getConstSource(roll.getShift());
    auto constAxesSource = getConstSource(roll.getAxes());

    if (constShiftSource != nullptr && constAxesSource != nullptr) {
        const bool shiftContentIsSplat = constShiftSource.getContentAttr().isSplat();
        auto shiftAndAxesOrFail =
                IE::getShiftAndAxesForRollOp(loc, constShiftSource, constAxesSource, inDataType.getShape());
        if (mlir::failed(shiftAndAxesOrFail)) {
            return mlir::failure();
        }
        const auto shiftAndAxes = shiftAndAxesOrFail.value();
        const auto inShapeShift = shiftAndAxes.shift;

        if (!shiftContentIsSplat && inShapeShift.size() == 1) {
            auto shiftData = VPU::extractConstData(loc, roll.getShift());
            if (mlir::failed(shiftData)) {
                return mlir::failure();
            }

            auto axesData = VPU::extractConstData(loc, roll.getAxes());
            if (mlir::failed(axesData)) {
                return mlir::failure();
            }

            auto shiftShape = shiftData.value();
            auto axesShape = axesData.value();

            if (shiftShape.size() != axesShape.size()) {
                return errorAt(
                        loc,
                        "If shift is a 1D vector, axes must be a 1D tensor of the same size. Got shift size {0} and "
                        "axes size {1}.",
                        shiftShape.size(), axesShape.size());
            }
        }
    }

    auto outType = mlir::RankedTensorType::get(inDataType.getShape(), inDataType.getElementType(),
                                               createTensorAttrFromType(inDataType));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::RollOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    TileInfo shiftTile(getShape(getShift()));
    TileInfo axesTile(getShape(getAxes()));

    return InputTiling{{outputTile, std::move(shiftTile), std::move(axesTile)}};
}

mlir::FailureOr<OutputTiling> vpux::VPU::RollOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

void vpux::VPU::RollOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // No attributes - do nothing
}

//
// ClusteredOpInterface
//

bool vpux::VPU::RollOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t /*numTiles*/) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getData().getType());
    const auto inShape = inputType.getShape();

    // Accuracy issue is found for the case where dataRank != TARGET_TENSOR_DIM with MC and MS
    // Tracked by #E167088
    if (inShape.size() != 4) {
        return false;
    }

    auto shiftAndAxesOrFail = IE::getShiftAndAxesForRollOp(getLoc(), getShift(), getAxes(), inShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return false;
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto axes = shiftAndAxes.axes;

    const auto noShiftOnDim{[&](auto dim) {
        for (auto axis : axes) {
            if (axis == dim.ind()) {
                return false;
            }
        }
        return true;
    }};

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth && inShape[Dims4D::Act::W] > 1) {
        return noShiftOnDim(Dims4D::Act::W);
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && inShape[Dims4D::Act::H] > 1) {
        return noShiftOnDim(Dims4D::Act::H);
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && inShape[Dims4D::Act::C] > 1) {
        return noShiftOnDim(Dims4D::Act::C);
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::RollOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::RollOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                        bool hasExplicitDistributedAttr,
                                                                        SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<RollOp>(getOperation());

    if (operand.get() == origOp.getData()) {
        return getSwDistributedTypeForOpOperand(clusteredOp, operand, siblingsAnalysis, hasExplicitDistributedAttr);
    } else if (operand.get() == origOp.getShift() || operand.get() == origOp.getAxes()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    VPUX_THROW("Failed to compute distributed type for op operand {0}", clusteredOp);
    return nullptr;
}

vpux::NDTypeInterface vpux::VPU::RollOp::getDistributedTypeForOpResult(mlir::Value result,
                                                                       VPU::MultiClusterStrategy strategy,
                                                                       SiblingOpsAnalysis& siblingsAnalysis,
                                                                       bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());

    return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis, strategy,
                                          hasExplicitDistributedAttr);
}

//
// SWOpInterface
//

bool vpux::VPU::RollOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 4, "RollOp requires 3 inputs and 1 outputs, but the number of buffer is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::RollOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::RollOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::RollOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value data,
                              ::mlir::Value shift, ::mlir::Value axes) {
    build(builder, state, data, shift, axes, nullptr);
}
