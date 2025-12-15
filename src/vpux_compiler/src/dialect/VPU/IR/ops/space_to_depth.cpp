//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

using namespace vpux;

bool isCompatibleWithMultiClusterNNDMA(VPU::SpaceToDepthOp op, vpux::ShapeRef numTilesOnDim) {
    if (op.getMode() != IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (inputType.getDimsOrder() != DimsOrder::NHWC || outputType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto tileExecOp = config::getTileExecutor(moduleOp);
    const auto numAvailableTiles = tileExecOp.getCount();
    if (numTilesOnDim.totalSize() > numAvailableTiles) {
        return false;
    }

    mlir::DenseSet<VPU::MultiClusterStrategy> allowedStrategies = {
            VPU::MultiClusterStrategy::SplitOverHeight,
            VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
            VPU::MultiClusterStrategy::HKSwitch,
    };

    // Check next ops
    VPU::DistributedTensorType userDistType = nullptr;
    for (const auto& userOp : op->getUsers()) {
        auto userNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(userOp);
        auto userClusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(userOp);
        if (userNCEOp == nullptr || userClusteredOp == nullptr) {
            return false;
        }
        auto userStrategy = userClusteredOp.getMultiClusterStrategy();
        if (!userStrategy.has_value() || !allowedStrategies.contains(userStrategy.value())) {
            return false;
        }
        // Check distributed tensor types are aligned
        auto userInputType = mlir::cast<vpux::NDTypeInterface>(userOp->getOperand(0).getType());
        auto userOutputType = mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType());
        auto numClusters = VPU::getOptimalNumClusters(userOp, userOutputType.getShape(), userStrategy.value());
        auto userInputDistType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(getDistributedActivationTypeFromOp(
                userClusteredOp, userOp->getOperand(0), userInputType, numClusters, userStrategy.value()));
        if (userInputDistType == nullptr) {
            return false;
        }
        if (userDistType == nullptr) {
            userDistType = userInputDistType;
            continue;
        }
        if (areDistributionAttrsCompatible(userDistType, userInputDistType).failed()) {
            return false;
        }
    }

    if (userDistType == nullptr) {
        return false;
    }

    // Currently MultiCluster S2D only supports SingleClusterCMX->MultiClusterCMX, so need to
    // check if the first cluster fits in CMX.
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(op.getOperation()).to<Byte>().count();
    // Input data is all on CMX0
    const auto inputBytes = inputType.getShape().totalSize() * inputType.getElemTypeSize().to<Byte>().count();
    // The only risk is CMX0 because input is also on it
    const auto perClusterOutMemShape = userDistType.getPerClusterMemoryShapes();
    const auto firstClusterOutputBytes =
            perClusterOutMemShape.front().totalSize() * userDistType.getElemTypeSize().to<Byte>().count();

    return (inputBytes + firstClusterOutputBytes) < cmxAvailableBytes;
}

mlir::LogicalResult vpux::VPU::SpaceToDepthOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SpaceToDepthOpAdaptor spd(operands, attrs, prop);
    if (mlir::failed(spd.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(spd.getInput().getType());

    const auto elementType = inputType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8) ||
          mlir::isa<mlir::quant::QuantizedType>(elementType))) {
        return errorAt(loc, "SpaceToDepth only supports FP16, FP32, U8 or Quantized input data type");
    }

    const auto inputShape = inputType.getShape().raw();
    const auto block_size = spd.getBlockSize();

    if (inputShape.size() < 3) {
        return errorAt(loc, "Input tensor rank must be greater than 2. Got {0}D tensor", inputShape.size());
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (inputShape[H.ind()] % block_size || inputShape[W.ind()] % block_size) {
        return errorAt(loc, "Invalid block_size {0} , height {1} and width {2} must be divisible by block_size",
                       block_size, inputShape[H.ind()], inputShape[W.ind()]);
    }

    const auto outN = inputShape[N.ind()];
    const auto outC = inputShape[C.ind()] * block_size * block_size;
    const auto outH = inputShape[H.ind()] / block_size;
    const auto outW = inputShape[W.ind()] / block_size;

    SmallVector<int64_t> outShape{outN, outC, outH, outW};

    auto outType = mlir::RankedTensorType::get(outShape, elementType, createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::SpaceToDepthOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(getInput());

    int64_t blockSize = 0;

    VPUX_THROW_UNLESS(getBlockSizeAttr() != nullptr, "Got NULL block_size");
    blockSize = mlir::dyn_cast<mlir::IntegerAttr>(getBlockSizeAttr()).getValue().getSExtValue();

    VPUX_THROW_WHEN(blockSize <= 0, "Unsupported block size: {0}", blockSize);

    TileInfo inputTile(origInputShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] / (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] * blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] * blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] / (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] * blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] * blockSize;

    return InputTiling{inputTile};
}

void vpux::VPU::SpaceToDepthOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::SpaceToDepthOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto origOp = mlir::dyn_cast<VPU::SpaceToDepthOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    int64_t blockSize = 0;

    VPUX_THROW_UNLESS(getBlockSizeAttr() != nullptr, "Got NULL block_size");
    blockSize = mlir::dyn_cast<mlir::IntegerAttr>(getBlockSizeAttr()).getValue().getSExtValue();

    Shape nTilesOnDimforSpaceToDepth(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    SmallVector<Dim> tileDimOrder;

    if (origOp.getMode() == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        tileDimOrder = {Dims4D::Act::W, Dims4D::Act::H};
    } else if (origOp.getMode() == IE::SpaceToDepthMode::DEPTH_FIRST) {
        tileDimOrder = {Dims4D::Act::W, Dims4D::Act::H, Dims4D::Act::C};
    } else {
        VPUX_THROW("Unknown SpaceToDepthMode: {0}. BLOCKS_FIRST and DEPTH_FIRST methods are supported only",
                   origOp.getMode());
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t maxTile = 1;

    VPUX_THROW_WHEN(blockSize <= 0, "Unsupported block size: {0}", blockSize);

    while (tileDimIter < tileDimOrder.end()) {
        if (dimToTile == Dims4D::Act::C) {
            while (((maxTile * blockSize * blockSize) <= outputShape[dimToTile]) &&
                   (!isSupportedTileSize(nTilesOnDimforSpaceToDepth, tilingModeToCheck))) {
                if (outputShape[dimToTile] % (maxTile * blockSize * blockSize) == 0) {
                    nTilesOnDimforSpaceToDepth[dimToTile] = maxTile;
                    maxTile++;
                } else {
                    maxTile++;
                }
            }
            dimToTile = *(++tileDimIter);
            maxTile = 1;
        } else if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            while (!isSupportedTileSize(nTilesOnDimforSpaceToDepth, tilingModeToCheck)) {
                if (nTilesOnDimforSpaceToDepth[dimToTile] >= outputShape[dimToTile]) {
                    break;
                } else {
                    ++nTilesOnDimforSpaceToDepth[dimToTile];
                }
            }
            dimToTile = *(++tileDimIter);
        } else {
            VPUX_THROW("Unsupported dim to tile: {0}", dimToTile);
        }
    }

    // No need to tile if s2d will be converted to MultiClusterNNDMA
    if (isCompatibleWithMultiClusterNNDMA(origOp, nTilesOnDimforSpaceToDepth)) {
        nTilesOnDimforSpaceToDepth = Shape(outputShape.size(), 1);
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforSpaceToDepth, outputShape);
    return origTiles;
}

mlir::LogicalResult VPU::SpaceToDepthOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "Block size should be a positive integer, while it is {0}", getBlockSize());
    }
    return mlir::success();
}
