//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

using namespace vpux;

namespace {

//
// ClusterExpandDMARewriter
//

class ClusterExpandDMARewriter final : public mlir::OpRewritePattern<VPUIP::ExpandDMAOp> {
public:
    ClusterExpandDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("ClusterExpandDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::ExpandDMAOp origOp, VPUIP::NCEClusterTilingOp clusterOp,
                                     VPURT::TaskOp vpurtTask, VPUIP::DistributedBufferType distributedType,
                                     VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                     mlir::PatternRewriter& rewriter) const;
    void unrollDuplicated(mlir::Location loc, VPUIP::ExpandDMAOp origOp, VPUIP::NCEClusterTilingOp clusterOp,
                          VPURT::TaskOp vpurtTask, VPUIP::DistributedBufferType distributedType,
                          VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                          mlir::PatternRewriter& rewriter) const;
    void createTilesForLargeSize(VPUIP::ExpandDMAOp origOp, VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                 mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef outShape, ShapeRef offset) {
    auto inShape = to_small_vector(outShape);
    // After Expand fuse into Permute and got one PermuteDMA Op
    // The channel size of input and output are not same
    // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
    // The channel size need align with the input
    inShape[Dims4D::Act::C.ind()] = originType.getShape()[Dims4D::Act::C];
    const auto elemType = originType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, Shape(inShape), offset);
        return originType.changeShapeElemType(Shape(inShape), newQType);
    }

    return originType.changeShape(Shape(inShape));
}

void ClusterExpandDMARewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp,
                                                           VPUIP::NCEClusterTilingOp clusterOp, VPURT::TaskOp vpurtTask,
                                                           VPUIP::DistributedBufferType distributedType,
                                                           VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();

    const auto innerInput = *clusterOp.getInnerInputs().begin();
    const auto innerOutput = *clusterOp.getInnerOutputs().begin();
    const auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
    const auto innerOutputType = innerOutput.getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();

    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    const auto originInShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' are mismatch", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    const auto tileInnerType = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(innerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };

    const auto isValidTile = [](auto dim) {
        return dim > 1;
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
                                                           getIntArrayAttr(_ctx, makeArrayRef({clusterId})),
                                                           declBuff.byteOffset(), declBuff.swizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.byteOffset()};
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

        ddrOffset += perClusterShapeOffsets[clusterId][Dim(axis)] * static_cast<Byte>(newType.getStrides()[Dim(axis)]);

        auto section = declBuff.section();
        auto sectionIndex = declBuff.sectionIndex();

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        newType = newType.changeMemSpace(symbolAttr);

        if (sectionIndex.hasValue()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.getValue(), ddrOffset.count(), nullptr);
        }

        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                       ddrOffset.count());
    };

    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    const auto inTypes = tileInnerType(innerInputType);
    const auto outTypes = tileInnerType(innerOutputType);
    Byte elemTypeSize = innerInputType.getElemTypeSize();
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
        auto dmaDescriptor = dmaDescriptorGenerator.generate(newInputType, newOutType, expandDmaOp.pads_beginAttr(),
                                                             expandDmaOp.pads_endAttr(), elemTypeSize.count());
        auto newDMAPort = clusterId % _dmaPortCount;
        auto newExpandDMAOp = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffer, outBuffer,
                expandDmaOp.pads_beginAttr(), expandDmaOp.pads_endAttr(), dmaDescriptor, getIntAttr(_ctx, newDMAPort));
        _log.trace("Insert new Expand dma : '{0}'", newExpandDMAOp);
        auto newTaskOp = newExpandDMAOp->getParentOfType<VPURT::TaskOp>();
        if (cycleBeginAttr) {
            newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newTaskOp->setAttr(cycleEnd, cycleEndAttr);
        }
    }
}

void ClusterExpandDMARewriter::unrollDuplicated(mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp,
                                                VPUIP::NCEClusterTilingOp clusterOp, VPURT::TaskOp vpurtTask,
                                                VPUIP::DistributedBufferType distributedType,
                                                VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                                mlir::PatternRewriter& rewriter) const {
    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();
    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

    auto newCMXBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(_ctx, clusters), outDeclBuff.byteOffset(), outDeclBuff.swizzlingKeyAttr());

    _log.trace("Insert new CMX buffer declaration: '{0}'", newCMXBuffer);

    const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
    auto expandInType = expandDmaOp.input().getType().dyn_cast<NDTypeInterface>();
    auto expandOutType = expandDmaOp.output().getType().dyn_cast<NDTypeInterface>();
    Byte elemTypeSize = expandInType.getElemTypeSize();
    auto dmaDescriptor = dmaDescriptorGenerator.generate(expandInType, expandOutType, expandDmaOp.pads_beginAttr(),
                                                         expandDmaOp.pads_endAttr(), elemTypeSize.count());
    const auto newExpandDMA = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
            rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, input, newCMXBuffer,
            expandDmaOp.pads_beginAttr(), expandDmaOp.pads_endAttr(), dmaDescriptor);
    _log.trace("Insert new ExpandDMA op: '{0}'", newExpandDMA);

    auto newVpurtTask = newExpandDMA->getParentOfType<VPURT::TaskOp>();
    if (vpurtTask->getAttr(cycleBegin)) {
        newVpurtTask->setAttr(cycleBegin, vpurtTask->getAttr(cycleBegin));
    }
    if (vpurtTask->getAttr(cycleEnd)) {
        newVpurtTask->setAttr(cycleEnd, vpurtTask->getAttr(cycleEnd));
    }
}

void ClusterExpandDMARewriter::createTilesForLargeSize(VPUIP::ExpandDMAOp origOp,
                                                       VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                                       mlir::PatternRewriter& rewriter) const {
    // Currently, tiling is implemented only for 4D shapes.
    const auto origInputShape = getShape(origOp.input());
    const auto origOutputShape = getShape(origOp.output());
    VPUX_THROW_UNLESS(origInputShape.size() == 4,
                      "ExpandDMAOpTiling: found shape {0} which is not supported yet (only 4D tensors are)",
                      origInputShape);

    const auto fullCopySize = static_cast<Byte>(getCompactSize(origOp.input()));
    // Always split by the first non-batch dimension, regardless the layout
    // NCHW - C, NHWC - H, NWHC - W
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto tileDim = inOrder.toDim(MemDim(Dims4D::Act::N.ind() + 1));

    // We cannot divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    const auto numPlanesPerTile = (VPUIP::DMA_LIMIT.count() / singlePlaneSize.count());
    VPUX_THROW_UNLESS(numPlanesPerTile != 0,
                      "Couldn't split a ExpandDMAOp with single plane size greater than DMA_LIMIT");

    auto inputDeclBuff = origOp.input().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outputDeclBuff = origOp.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inputDeclBuff != nullptr && outputDeclBuff != nullptr,
                      "Can't get input or output buffer of ExpandDMAOp '{0}'", origOp->getLoc());

    Byte inputOffset{inputDeclBuff.byteOffset()};
    Byte outputOffset{outputDeclBuff.byteOffset()};

    const auto expandInputType = origOp.input().getType().cast<NDTypeInterface>();
    const Byte elemTypeSize = expandInputType.getElemTypeSize();

    auto vpurtTask = origOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    auto currentTileInShape = Shape(origInputShape.raw());
    auto currentTileOutShape = Shape(origOutputShape.raw());
    auto planesLeftToCopy = numPlanesOfFullShape;
    auto inputInsertionPoint = origOp.input().getDefiningOp();
    auto outputInsertionPoint = origOp.output_buff().getDefiningOp();
    for (int64_t tileIdx = 0; planesLeftToCopy > 0; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), "tile {0}", tileIdx);
        currentTileInShape[tileDim] = std::min(numPlanesPerTile, planesLeftToCopy);
        currentTileOutShape[tileDim] = std::min(numPlanesPerTile, planesLeftToCopy);

        // Create new input buffer
        auto inputNewType = inputDeclBuff.getType().cast<NDTypeInterface>().changeShape(currentTileInShape);
        auto inputNewBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, inputInsertionPoint, tileLoc, inputNewType, inputDeclBuff.section(), inputOffset.count());
        inputInsertionPoint = inputNewBuffer.getResult().getDefiningOp();
        inputOffset += Byte(currentTileInShape.totalSize() * elemTypeSize.count());

        // Create new output buffer
        auto outputNewBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, outputInsertionPoint, tileLoc,
                                                                       outputDeclBuff.getType(),
                                                                       outputDeclBuff.section(), outputOffset.count());
        outputInsertionPoint = outputNewBuffer.getResult().getDefiningOp();
        outputOffset += Byte(currentTileOutShape.totalSize() * elemTypeSize.count());

        // Create Descriptor
        auto expandInType = origOp.input().getType().dyn_cast<NDTypeInterface>();
        auto expandOutType = origOp.output().getType().dyn_cast<NDTypeInterface>();
        auto dmaDescriptor =
                dmaDescriptorGenerator.generate(expandInType.changeShape(currentTileInShape), expandOutType,
                                                origOp.pads_beginAttr(), origOp.pads_endAttr(), elemTypeSize.count());

        // Create tile ExpandDMAOp
        auto newDMAPort = tileIdx % _dmaPortCount;
        auto newExpandDMAOp = VPURT::wrapIntoTaskOp<VPUIP::ExpandDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), tileLoc, inputNewBuffer,
                outputNewBuffer, origOp.pads_beginAttr(), origOp.pads_endAttr(), dmaDescriptor,
                getIntAttr(_ctx, newDMAPort));

        _log.trace("New tile '{0}' Expand dma : '{1}'", tileIdx, newExpandDMAOp);

        planesLeftToCopy -= currentTileInShape[tileDim];

        auto newVpurtTask = newExpandDMAOp->getParentOfType<VPURT::TaskOp>();
        if (vpurtTask->getAttr(cycleBegin)) {
            newVpurtTask->setAttr(cycleBegin, vpurtTask->getAttr(cycleBegin));
        }
        if (vpurtTask->getAttr(cycleEnd)) {
            newVpurtTask->setAttr(cycleEnd, vpurtTask->getAttr(cycleEnd));
        }
    }

    VPUX_THROW_UNLESS(planesLeftToCopy == 0,
                      "ExpandDMAOpTiling: a part of the original shape was not covered by ExpandDMA tiles");

    rewriter.eraseOp(vpurtTask);
}

mlir::LogicalResult ClusterExpandDMARewriter::matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Process ExpandDMA op: {0}", expandDmaOp);

    if (expandDmaOp.dma_descriptor().hasValue()) {
        return mlir::failure();
    }

    auto dmaDescriptorGenerator = VPUIP::ExpandDmaDescriptorGenerator(_ctx, _log);

    auto clusterOp = expandDmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        _log.trace("ExpandDMA is not a child of NCEClusterTiling op");

        const auto dmaSize = static_cast<Byte>(getCompactSize(expandDmaOp.input()));
        if (dmaSize > VPUIP::DMA_LIMIT) {
            _log.trace("ExpandDMA with input size '{0}' large than limitation '{1}' and need to tile", dmaSize,
                       VPUIP::DMA_LIMIT);
            createTilesForLargeSize(expandDmaOp, dmaDescriptorGenerator, rewriter);
            return mlir::success();
        }

        auto expandInType = expandDmaOp.input().getType().dyn_cast<NDTypeInterface>();
        auto expandOutType = expandDmaOp.output().getType().dyn_cast<NDTypeInterface>();
        Byte elemTypeSize = expandInType.getElemTypeSize();
        auto dmaDescriptor = dmaDescriptorGenerator.generate(expandInType, expandOutType, expandDmaOp.pads_beginAttr(),
                                                             expandDmaOp.pads_endAttr(), elemTypeSize.count());
        rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandDmaOp, expandDmaOp.input(), expandDmaOp.output_buff(),
                                                        expandDmaOp.pads_beginAttr(), expandDmaOp.pads_endAttr(),
                                                        dmaDescriptor);

        return mlir::success();
    }

    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();
    const auto inputType = input.getType();
    const auto outputType = output.getType();
    VPUX_THROW_WHEN(inputType.isa<VPUIP::DistributedBufferType>() && outputType.isa<VPUIP::DistributedBufferType>(),
                    "Only one operand can have DistributedBuffer type");

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Output must have DistributedBuffer type");

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
        _log.trace("Process SEGMENTED/OVERLAPPED mode", VPU::stringifyDistributionMode(mode));
        unrollSegmentedOrOverlapped(expandDmaOp->getLoc(), expandDmaOp, clusterOp, vpurtTask, distributedType,
                                    dmaDescriptorGenerator, rewriter);
    } else if (mode == VPU::DistributionMode::DUPLICATED) {
        _log.trace("Process DUPLICATED mode");
        unrollDuplicated(expandDmaOp->getLoc(), expandDmaOp, clusterOp, vpurtTask, distributedType,
                         dmaDescriptorGenerator, rewriter);
    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(mode));
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

//
// UnrollExpandDMAPass
//

class UnrollExpandDMAPass final : public VPUIP::UnrollExpandDMABase<UnrollExpandDMAPass> {
public:
    explicit UnrollExpandDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollExpandDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ClusterExpandDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollExpandDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollExpandDMAPass(Logger log) {
    return std::make_unique<UnrollExpandDMAPass>(log);
}
