//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_expand_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_expand_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP::arch37xx {

mlir::LogicalResult SingleClusterExpandDMARewriter::matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Process ExpandDMA op: {0}", expandDmaOp);

    if (expandDmaOp.getDmaDescriptor().has_value()) {
        return mlir::failure();
    }

    auto dmaDescriptorGenerator = VPUIP::ExpandDmaDescriptorGenerator(_ctx, _log);

    const auto input = expandDmaOp.getInput();
    const auto output = expandDmaOp.getOutputBuff();
    const auto inputType = input.getType();
    const auto outputType = output.getType();
    VPUX_THROW_WHEN(mlir::isa<vpux::VPUIP::DistributedBufferType>(inputType),
                    "Cannot unroll input DistributedBuffer type.");

    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

    if (distributedType == nullptr) {
        return unrollSingleTile(expandDmaOp, dmaDescriptorGenerator, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult MultiClusterExpandDMARewriter::matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Process ExpandDMA op: {0}", expandDmaOp);

    if (expandDmaOp.getDmaDescriptor().has_value()) {
        return mlir::failure();
    }

    auto dmaDescriptorGenerator = VPUIP::ExpandDmaDescriptorGenerator(_ctx, _log);

    const auto input = expandDmaOp.getInput();
    const auto output = expandDmaOp.getOutputBuff();
    const auto inputType = input.getType();
    const auto outputType = output.getType();
    VPUX_THROW_WHEN(mlir::isa<vpux::VPUIP::DistributedBufferType>(inputType),
                    "Cannot unroll input DistributedBuffer type.");

    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

    if (distributedType == nullptr) {
        return mlir::failure();
    }

    auto vpurtTask = expandDmaOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
        _log.trace("Process SEGMENTED/OVERLAPPED mode", VPU::stringifyDistributionMode(mode));
        unrollSegmentedOrOverlapped(expandDmaOp->getLoc(), expandDmaOp, vpurtTask, distributedType,
                                    dmaDescriptorGenerator, rewriter);
    } else if (mode == VPU::DistributionMode::DUPLICATED) {
        _log.trace("Process DUPLICATED mode");
        unrollDuplicated(expandDmaOp->getLoc(), expandDmaOp, vpurtTask, distributedType, dmaDescriptorGenerator,
                         rewriter);
    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(mode));
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

void MultiClusterExpandDMARewriter::unrollSegmentedOrOverlapped(
        mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp, VPURT::TaskOp vpurtTask,
        VPUIP::DistributedBufferType distributedType, VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
        mlir::PatternRewriter& rewriter) const {
    const auto input = expandDmaOp.getInput();
    const auto output = expandDmaOp.getOutputBuff();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = distributedType.getCompactType();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();

    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto originInShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' are mismatch", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    const auto tileType = [&](vpux::NDTypeInterface type) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    ExpandDMA::changeShape(type, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };

    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(_ctx, ArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.getByteOffset()};
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        ddrOffset += perClusterShapeOffsets[clusterId][Dim(axis)] * static_cast<Byte>(newType.getStrides()[Dim(axis)]);

        auto section = declBuff.getSection();
        auto sectionIndex = declBuff.getSectionIndex();

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        newType = newType.changeMemSpace(symbolAttr);

        if (sectionIndex.has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), ddrOffset.count(), nullptr);
        }

        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                       ddrOffset.count());
    };

    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    const auto inTypes = tileType(inputType);
    const auto outTypes = tileType(outputType);
    Byte elemTypeSize = inputType.getElemTypeSize();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto dmaDescriptor = dmaDescriptorGenerator.generate(newInputType, newOutType, expandDmaOp.getPadsBeginAttr(),
                                                             expandDmaOp.getPadsEndAttr(), elemTypeSize.count());
        auto newDMAPort = clusterId % _dmaPortCount;
        auto newExpandDMAOp = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                expandDmaOp.getPadsBeginAttr(), expandDmaOp.getPadsEndAttr(), dmaDescriptor,
                getIntAttr(_ctx, newDMAPort), expandDmaOp.getIsOutOfOrderAttr(), expandDmaOp.getIsCriticalAttr(),
                expandDmaOp.getDmaHwpIdAttr(), expandDmaOp.getProfilingMetadataAttr());

        _log.trace("Insert new Expand dma : '{0}'", newExpandDMAOp);
    }
}

void MultiClusterExpandDMARewriter::unrollDuplicated(mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp,
                                                     VPURT::TaskOp vpurtTask,
                                                     VPUIP::DistributedBufferType distributedType,
                                                     VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto input = expandDmaOp.getInput();
    const auto output = expandDmaOp.getOutputBuff();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

    auto newCMXBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(_ctx, clusters), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

    _log.trace("Insert new CMX buffer declaration: '{0}'", newCMXBuffer);

    const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
    auto expandInType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDmaOp.getInput().getType());
    auto expandOutType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDmaOp.getOutput().getType());
    Byte elemTypeSize = expandInType.getElemTypeSize();
    auto dmaDescriptor = dmaDescriptorGenerator.generate(expandInType, expandOutType, expandDmaOp.getPadsBeginAttr(),
                                                         expandDmaOp.getPadsEndAttr(), elemTypeSize.count());
    const auto newExpandDMA = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, input, newCMXBuffer,
            expandDmaOp.getPadsBeginAttr(), expandDmaOp.getPadsEndAttr(), dmaDescriptor,
            /*port=*/vpux::getIntAttr(rewriter, 0), expandDmaOp.getIsOutOfOrderAttr(), expandDmaOp.getIsCriticalAttr(),
            expandDmaOp.getDmaHwpIdAttr(), expandDmaOp.getProfilingMetadataAttr());
    _log.trace("Insert new ExpandDMA op: '{0}'", newExpandDMA);
}

void SingleClusterExpandDMARewriter::createTilesForLargeSize(VPUIP::ExpandDMAOp origOp,
                                                             VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                                             mlir::PatternRewriter& rewriter) const {
    // Currently, tiling is implemented only for 4D shapes.
    const auto origInputShape = getShape(origOp.getInput());
    const auto origOutputShape = getShape(origOp.getOutput());
    VPUX_THROW_UNLESS(origInputShape.size() == 4,
                      "ExpandDMAOpTiling: found shape {0} which is not supported yet (only 4D tensors are)",
                      origInputShape);

    const auto fullCopySize = static_cast<Byte>(getCompactSize(origOp.getInput()));
    // Always split by the first non-batch dimension, regardless the layout
    // NCHW - C, NHWC - H, NWHC - W
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto tileDim = inOrder.toDim(MemDim(Dims4D::Act::N.ind() + 1));

    const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto padEndAxisIter = std::find_if(padEnd.begin(), padEnd.end(), nonZeroAxisPredicate);
    VPUX_THROW_WHEN(padEndAxisIter == padEnd.end(), "Can not find padding axis");

    const auto padEndAxis = std::distance(padEnd.begin(), padEndAxisIter);
    auto isPadOnTileDim = tileDim.ind() == padEndAxis;

    // We cannot divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;

    // Deeply rooted in NPU2.7 representation
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(origOp));
    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();
    const auto numPlanesPerTile = (dmaMaxLength / singlePlaneSize.count());
    VPUX_THROW_UNLESS(numPlanesPerTile != 0,
                      "Couldn't split a ExpandDMAOp with single plane size greater than plane limit");

    auto inputDeclBuff = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outputDeclBuff = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inputDeclBuff != nullptr && outputDeclBuff != nullptr,
                      "Can't get input or output buffer of ExpandDMAOp '{0}'", origOp->getLoc());

    Byte inputOffset{inputDeclBuff.getByteOffset()};
    Byte outputOffset{outputDeclBuff.getByteOffset()};

    const auto expandInputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const Byte elemTypeSize = expandInputType.getElemTypeSize();

    auto vpurtTask = origOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    auto currentTileInShape = Shape(origInputShape.raw());
    auto currentTileOutShape = Shape(origOutputShape.raw());
    auto planesLeftToCopy = numPlanesOfFullShape;
    auto inputInsertionPoint = origOp.getInput().getDefiningOp();
    auto outputInsertionPoint = origOp.getOutputBuff().getDefiningOp();
    for (int64_t tileIdx = 0; planesLeftToCopy > 0; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), "tile {0}", tileIdx);
        currentTileInShape[tileDim] = std::min(numPlanesPerTile, planesLeftToCopy);
        currentTileOutShape[tileDim] = std::min(numPlanesPerTile, planesLeftToCopy);
        auto newPadEnd = padEnd;
        newPadEnd[padEndAxis] += (planesLeftToCopy - currentTileInShape[tileDim]);

        // Create new input buffer
        auto inputNewType = mlir::cast<vpux::NDTypeInterface>(inputDeclBuff.getType()).changeShape(currentTileInShape);
        auto inputNewBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, inputInsertionPoint, tileLoc, inputNewType, inputDeclBuff.getSection(), inputOffset.count());
        inputInsertionPoint = inputNewBuffer.getResult().getDefiningOp();
        inputOffset += Byte(currentTileInShape.totalSize() * elemTypeSize.count());

        // Create new output buffer
        auto outputNewBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, outputInsertionPoint, tileLoc, outputDeclBuff.getType(), outputDeclBuff.getSection(),
                outputOffset.count());
        outputInsertionPoint = outputNewBuffer.getResult().getDefiningOp();
        outputOffset += Byte(currentTileOutShape.totalSize() * elemTypeSize.count());

        // Create Descriptor
        auto expandInType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getInput().getType());
        auto expandOutType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        auto dmaDescriptor = dmaDescriptorGenerator.generate(expandInType.changeShape(currentTileInShape),
                                                             expandOutType, origOp.getPadsBeginAttr(),
                                                             origOp.getPadsEndAttr(), elemTypeSize.count());

        // Create tile ExpandDMAOp
        auto newDMAPort = tileIdx % _dmaPortCount;
        auto newExpandDMAOp = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), tileLoc, inputNewBuffer,
                outputNewBuffer, origOp.getPadsBeginAttr(),
                isPadOnTileDim ? getIntArrayAttr(_ctx, newPadEnd) : origOp.getPadsEndAttr(), dmaDescriptor,
                getIntAttr(_ctx, newDMAPort), origOp.getIsOutOfOrderAttr(), origOp.getIsCriticalAttr(),
                origOp.getDmaHwpIdAttr(), origOp.getProfilingMetadataAttr());

        _log.trace("New tile '{0}' Expand dma : '{1}'", tileIdx, newExpandDMAOp);

        planesLeftToCopy -= currentTileInShape[tileDim];
    }

    VPUX_THROW_UNLESS(planesLeftToCopy == 0,
                      "ExpandDMAOpTiling: a part of the original shape was not covered by ExpandDMA tiles");

    rewriter.eraseOp(vpurtTask);
}

mlir::LogicalResult SingleClusterExpandDMARewriter::unrollSingleTile(
        VPUIP::ExpandDMAOp expandDmaOp, VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
        mlir::PatternRewriter& rewriter) const {
    _log.trace("ExpandDMA's result is not DistributedBufferType");

    const auto dmaSize = static_cast<Byte>(getCompactSize(expandDmaOp.getInput()));
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(expandDmaOp));
    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();
    if (dmaSize > Byte(dmaMaxLength)) {
        _log.trace("ExpandDMA with input size '{0}' large than limitation '{1}' and need to tile", dmaSize,
                   dmaMaxLength);
        createTilesForLargeSize(expandDmaOp, dmaDescriptorGenerator, rewriter);
        return mlir::success();
    }

    auto expandInType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDmaOp.getInput().getType());
    auto expandOutType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDmaOp.getOutput().getType());
    Byte elemTypeSize = expandInType.getElemTypeSize();
    auto dmaDescriptor = dmaDescriptorGenerator.generate(expandInType, expandOutType, expandDmaOp.getPadsBeginAttr(),
                                                         expandDmaOp.getPadsEndAttr(), elemTypeSize.count());
    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(
            expandDmaOp, expandDmaOp.getInput(), expandDmaOp.getOutputBuff(), expandDmaOp.getPadsBeginAttr(),
            expandDmaOp.getPadsEndAttr(), dmaDescriptor, vpux::getIntAttr(rewriter, 0),
            expandDmaOp.getIsOutOfOrderAttr(), expandDmaOp.getIsCriticalAttr(), expandDmaOp.getDmaHwpIdAttr(),
            expandDmaOp.getProfilingMetadataAttr());

    return mlir::success();
}

UnrollExpandDMAStrategy::UnrollExpandDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollExpandDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::arch37xx::SingleClusterExpandDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::arch37xx::MultiClusterExpandDMARewriter>(ctx, _dmaPortCount, log);
}

}  // namespace vpux::VPUIP::arch37xx
