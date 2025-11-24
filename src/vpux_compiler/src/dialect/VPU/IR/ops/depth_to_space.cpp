//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/IR/Location.h>

using namespace vpux;

//
// ClusteredOpInterface
//

bool vpux::VPU::DepthToSpaceOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inShape = getBoundedShape(inputType);

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return false;
    }

    // Optimized DepthToSpace SW kernel implementation has no restriction on W and H (can be tiled on these dims), but
    // cannot split C-dim on multiple shaves, see E#86460.
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight &&
        inShape[Dims4D::Act::H] >= checked_cast<int64_t>(numTiles)) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth &&
        inShape[Dims4D::Act::W] >= checked_cast<int64_t>(numTiles)) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::DepthToSpaceOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

void vpux::VPU::DepthToSpaceOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value input, ::mlir::IntegerAttr blockSize,
                                      vpux::IE::DepthToSpaceModeAttr mode,
                                      /*optional*/ vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, blockSize, mode, padded_channels, {});
}

//
// SWOpInterface
//

bool vpux::VPU::DepthToSpaceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "DepthToSpaceOp requires 1 input and 1 output, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::DepthToSpaceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DepthToSpaceOp::supportCycleCostCalculation() {
    return false;
}

mlir::LogicalResult vpux::VPU::DepthToSpaceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DepthToSpaceOpAdaptor depthToSpace(operands, attrs, prop);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.getInput());
    const auto inType = mlir::cast<vpux::NDTypeInterface>(depthToSpace.getInput().getType());
    const auto blockSize = depthToSpace.getBlockSize();

    const auto elemType = inType.getElementType();
    if (!(elemType.isF16() || elemType.isF32() || elemType.isUnsignedInteger(8) ||
          mlir::isa<mlir::quant::QuantizedType>(elemType))) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8 data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (blockSize <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", blockSize);
    }

    if (inShape[Dims4D::Act::C] == mlir::ShapedType::kDynamic) {
        return errorAt(loc, "Input channels dimension is dynamic, cannot infer output shape");
    }
    if (inShape[Dims4D::Act::N] == mlir::ShapedType::kDynamic) {
        return errorAt(loc, "Input batch size dimension is dynamic, cannot infer output shape");
    }

    if (inShape[Dims4D::Act::C] % (blockSize * blockSize) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", blockSize,
                       inShape[Dims4D::Act::C]);
    }

    int64_t paddedIC = 0;
    int64_t paddedOC = 0;

    auto blockSizeSquare = blockSize * blockSize;
    auto paddedChannels = depthToSpace.getPaddedChannels();
    if (paddedChannels.has_value()) {
        paddedIC = paddedChannels.value().getInput() ? paddedChannels.value().getInput().getInt() : 0;
        paddedOC = paddedChannels.value().getOutput() ? paddedChannels.value().getOutput().getInt() : 0;

        auto unpaddedChannels = inShape[Dims4D::Act::C] - paddedIC;
        if (unpaddedChannels % blockSizeSquare != 0) {
            return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", blockSize,
                           unpaddedChannels);
        }
    }

    const int64_t outW = inShape[Dims4D::Act::W] == mlir::ShapedType::kDynamic
                                 ? mlir::ShapedType::kDynamic
                                 : checked_cast<int64_t>(inShape[Dims4D::Act::W] * blockSize);
    const int64_t outH = inShape[Dims4D::Act::H] == mlir::ShapedType::kDynamic
                                 ? mlir::ShapedType::kDynamic
                                 : checked_cast<int64_t>(inShape[Dims4D::Act::H] * blockSize);
    const int64_t outC = checked_cast<int64_t>((inShape[Dims4D::Act::C] - paddedIC) / blockSizeSquare + paddedOC);
    const int64_t outN = checked_cast<int64_t>(inShape[Dims4D::Act::N]);

    auto [outDesc, outShape] = callOnShapeOf(inType, [&](const auto& shape) {
        const SmallVector<int64_t> outputShape{outN, outC, outH, outW};
        if constexpr (std::is_same_v<std::decay_t<decltype(shape)>, BoundedShape>) {
            const auto boundedShape = getBoundedShape(inType);
            SmallVector<int64_t> outBounds{outputShape[Dims4D::Act::N.ind()], outputShape[Dims4D::Act::C.ind()],
                                           static_cast<int64_t>(boundedShape[Dims4D::Act::H] * blockSize),
                                           static_cast<int64_t>(boundedShape[Dims4D::Act::W] * blockSize)};
            auto desc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), BoundsRef(outBounds));
            return std::make_pair(std::move(desc), std::move(outputShape));
        } else if constexpr (std::is_same_v<std::decay_t<decltype(shape)>, DimsMaskedShape>) {
            auto inDynamicDimsMaskType = mlir::cast<Core::DynamicDimsMaskTensorType>(inType);
            auto desc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), {},
                                            inDynamicDimsMaskType.getDynamicDimsMask());
            return std::make_pair(std::move(desc), std::move(outputShape));
        } else {
            auto desc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
            return std::make_pair(std::move(desc), std::move(outputShape));
        }
    });

    auto outType = mlir::RankedTensorType::get(outShape, elemType, outDesc);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::DepthToSpaceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(getInput());

    int64_t blockSize = 0;
    if (getBlockSizeAttr() != nullptr) {
        blockSize = getBlockSizeAttr().getValue().getSExtValue();
    }
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");

    TileInfo inputTile(origInputShape);
    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] / blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] / blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] / blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] / blockSize;

    return InputTiling{inputTile};
}

void vpux::VPU::DepthToSpaceOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> getD2STilingStrategy(mlir::Operation* op, TilingMode tilingMode, bool useDMA,
                                                   Logger log) {
    auto origOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = getBoundedShape(outputType);

    int64_t blockSize = 0;
    if (origOp.getBlockSizeAttr() != nullptr) {
        blockSize = origOp.getBlockSizeAttr().getValue().getSExtValue();
    }
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");

    auto newShape = to_small_vector(outputShape);
    newShape[Dims4D::Act::H.ind()] /= blockSize;
    newShape[Dims4D::Act::W.ind()] /= blockSize;
    auto outputShapeReducedHW = ShapeRef(newShape);

    Shape nTilesOnDimforDepthToSpace(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    auto tileDimOrder = getTileDimOrder(op, tilingMode, log);

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);

        if (mlir::failed(tiles)) {
            return false;
        }

        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t maxTile = 1;

    while (tileDimIter < tileDimOrder.end()) {
        if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            while ((maxTile <= outputShapeReducedHW[dimToTile]) &&
                   (!isSupportedTileSize(nTilesOnDimforDepthToSpace, tilingModeToCheck))) {
                nTilesOnDimforDepthToSpace[dimToTile] = maxTile;
                maxTile++;
            }
            dimToTile = *(++tileDimIter);
            maxTile = 1;
        } else if (dimToTile == Dims4D::Act::C) {
            while (!isSupportedTileSize(nTilesOnDimforDepthToSpace, tilingModeToCheck)) {
                if (nTilesOnDimforDepthToSpace[dimToTile] >= outputShape[dimToTile]) {
                    break;
                } else {
                    ++nTilesOnDimforDepthToSpace[dimToTile];
                }
            }
            dimToTile = *(++tileDimIter);
        }
    }

    // Explicit tiling not needed, op will be converted to multicluster DMA
    if (useDMA && vpux::VPUIP::isCompatibleWithMultiClusterNNDMA(origOp, nTilesOnDimforDepthToSpace)) {
        nTilesOnDimforDepthToSpace = vpux::Shape(outputShape.size(), 1);
    } else {
        // Update tiling with multiCluster
        auto strategyAttr = origOp.getMultiClusterStrategy();
        if (strategyAttr.has_value()) {
            auto numClusters = VPU::getOptimalNumClusters(op, outputShape, strategyAttr.value());
            auto strategy = strategyAttr.value();
            if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
                nTilesOnDimforDepthToSpace[Dims4D::Act::H] =
                        (nTilesOnDimforDepthToSpace[Dims4D::Act::H] + numClusters - 1) / numClusters;
            }

            if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
                nTilesOnDimforDepthToSpace[Dims4D::Act::W] =
                        (nTilesOnDimforDepthToSpace[Dims4D::Act::W] + numClusters - 1) / numClusters;
            }
        }
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforDepthToSpace, outputShape);
    return origTiles;
}

mlir::FailureOr<OutputTiling> vpux::VPU::DepthToSpaceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto useDMA = vpux::VPUIP::isLegalAndBeneficialConvertToDMA(op, log);
    return getD2STilingStrategy(op, tilingMode, useDMA, log);
}

bool vpux::VPU::DepthToSpaceOp::isVFSupported() {
    return true;
}

mlir::LogicalResult VPU::DepthToSpaceOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "Block size should be a positive integer, while it is {0}", getBlockSize());
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::DepthToSpaceOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();
    // Parse attributes
    auto blockSize = getBlockSize();

    const auto inputShapedType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());

    VPUX_THROW_WHEN(inputShapedType.getRank() != 4 || outputShapedType.getRank() != 4,
                    "reify D2S: Unsupported input or output rank: {0} , {1}", inputShapedType.getRank(),
                    outputShapedType.getRank());

    auto makeIndex = [&](int64_t value) {
        return builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, value);
    };

    auto getInputDimVal = [&](int64_t idx, mlir::Location dimLoc) {
        auto inputDim = reifyDim(builder, getInput(), idx, dimLoc);
        auto inputDimVal = mlir::dyn_cast<mlir::Value>(inputDim);
        VPUX_THROW_WHEN(inputDimVal == nullptr, "Failed to reify input dimension {0} for input {1} at location {2}",
                        idx, getInput(), loc);

        return inputDimVal;
    };

    // Use generator functions based on index for each output dimension
    auto computeShapeForDim = [&](int64_t idx) -> mlir::OpFoldResult {
        auto dimLoc = appendLoc(loc, llvm::StringLiteral("_dim_{0}"), idx);

        if (idx == Dims4D::Act::N.ind()) {
            return reifyDim(builder, getInput(), idx, dimLoc);
        } else if (idx == Dims4D::Act::C.ind()) {
            // outC = inC / (blockSize * blockSize)
            auto inputDimVal = getInputDimVal(idx, dimLoc);
            return builder.createOrFold<mlir::arith::DivSIOp>(dimLoc, inputDimVal, makeIndex(blockSize * blockSize));
        } else if (idx == Dims4D::Act::H.ind() || idx == Dims4D::Act::W.ind()) {
            // outHW = inHW * blockSize
            auto inputDimVal = getInputDimVal(idx, dimLoc);

            return builder.createOrFold<mlir::arith::MulIOp>(dimLoc, inputDimVal, makeIndex(blockSize));
        } else {
            VPUX_THROW("Unexpected dimension index {0}", idx);
        }
    };

    SmallVector<mlir::OpFoldResult> outShape;
    for (const auto dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dim)) {
            outShape.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, computeShapeForDim(dim)));
        } else {
            outShape.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        }
    }

    reifiedReturnShapes.emplace_back(std::move(outShape));
    return mlir::success();
}
