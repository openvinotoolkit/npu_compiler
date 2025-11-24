//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_permute_dma.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {

NDTypeInterface changeShape(NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        return originType.changeShapeElemType(shape, newQType);
    }

    return originType.changeShape(shape);
}

NDTypeInterface getPerClusterInputType(NDTypeInterface inputType, NDTypeInterface outputType, mlir::AffineMap memPerm,
                                       ShapeRef outShape, ShapeRef offset) {
    auto inputShape = inputType.getShape();

    // Back infer the input shape from output shape and mem_perm attribute
    // For example: Input: 1x8x1x32xfp16, #NHWC -> 1x32x1x8xfp16, #NHWC, memPerm: [0, 1, 3, 2]
    // If want get right input shape from per cluster output shape. There are three step:
    //   1) Get the output real physical shape: 1x32x1x8xfp16, #NHWC -> 1x1x8x32
    //   2) Using memPerm to back infer the input real physical shape: 1x1x8x32 -> 1x1x32x8
    //   3) Got the input logic shape: 1x1x32x8 -> 1x8x1x32xfp16, #NHWC
    const auto inOrder = inputType.getDimsOrder();
    const auto outOrder = outputType.getDimsOrder();
    auto backInferInputShape = [&](ShapeRef subOutShape) -> Shape {
        // After Expand fuse into Permute and got one PermuteDMA Op
        // The channel size of input and output are not same
        // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
        // The channel size need align with the input
        auto inLogicShape = to_small_vector(subOutShape);
        if (inputType.getShape().totalSize() != outputType.getShape().totalSize()) {
            VPUX_THROW_UNLESS(subOutShape[Dims4D::Act::C] != inputShape[Dims4D::Act::C],
                              "Got unexpected input {0} output {1} type of PermuteDMA", inputType, outputType);
            inLogicShape[Dims4D::Act::C.ind()] = inputShape[Dims4D::Act::C];
        }

        Shape outPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            outPhysicalShape[Dim(idx)] = inLogicShape[outOrder.dimAt(idx).ind()];
        }

        Shape inPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            inPhysicalShape[DimsOrder::fromAffineMap(memPerm).dimAt(idx)] = outPhysicalShape[Dim(idx)];
        }

        for (const auto idx : irange(inLogicShape.size())) {
            inLogicShape[inOrder.dimAt(idx).ind()] = inPhysicalShape[Dim(idx)];
        }

        return Shape(inLogicShape);
    };

    const auto inferredShape = backInferInputShape(outShape);
    return changeShape(inputType, inferredShape, offset);
}

mlir::AffineMap getLogicalTransposeFromMemPermute(NDTypeInterface inType, NDTypeInterface outType,
                                                  mlir::AffineMap memPermute) {
    VPUX_THROW_WHEN(inType.getRank() != outType.getRank(), "Rank mismatch between input type and output type");

    auto ctx = inType.getContext();
    const auto mappingInMemToLogical = VPUIP::getSmallVectorFromAffineMap(inType.getDimsOrder().toAffineMap(ctx));
    const auto mappingOutMemToLogical = VPUIP::getSmallVectorFromAffineMap(outType.getDimsOrder().toAffineMap(ctx));
    const auto mappingOutToInMem = VPUIP::getSmallVectorFromAffineMap(memPermute);

    auto mappingOutToInLogical = mlir::SmallVector<int64_t>(inType.getRank());

    for (auto index : irange(inType.getRank())) {
        mappingOutToInLogical[mappingOutMemToLogical[index]] = mappingInMemToLogical[mappingOutToInMem[index]];
    }

    return mlir::AffineMap::getPermutationMap(mappingOutToInLogical, ctx);
}

bool isMultiClusterPermuteDMA(VPUIP::PermuteDMAOp permuteDMAOp) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(permuteDMAOp.getInput().getType());
    auto outDistributedType =
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(permuteDMAOp.getOutputBuff().getType());

    // Dispatch between single cluster and multi-cluster tasks.
    //  - multi-cluster tasks have at least one distributed buffer and do not have DMADescriptorAttr
    //  - single cluster tasks either do not have any distributed buffers or have DMADescriptorAttr resulted
    //  from multi-cluster task unrolling
    //  - only form of single-cluster tasks with distributed buffers is with DUPLICATED output buffer
    return ((inDistributedType || outDistributedType) && !permuteDMAOp.getDmaDescriptorAttr());
}

SingleClusterPermuteDMARewriter::SingleClusterPermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                 Logger log)
        : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("SingleClusterPermuteDMARewriter");
}

mlir::LogicalResult SingleClusterPermuteDMARewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteDMAOp.getMemPermAttr() == nullptr) {
        return mlir::failure();
    }

    if (!isMultiClusterPermuteDMA(permuteDMAOp)) {
        _log.trace("Got PermuteDMAOp '{0}' at '{1}'", permuteDMAOp->getName(), permuteDMAOp->getLoc());
        return unroll(permuteDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterPermuteDMARewriter::unroll(VPUIP::PermuteDMAOp permuteDMAOp,
                                                            mlir::PatternRewriter& rewriter) const {
    VPUX_THROW_WHEN(permuteDMAOp.getInternalDataFlowAttr(), "Already unrolled");
    VPUX_THROW_WHEN(_dmaPortCount < 1, "Invalid number of ports (expected at least 1, but got {0})", _dmaPortCount);

    auto ctx = permuteDMAOp->getContext();

    auto origTaskOp = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(origTaskOp != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(origTaskOp);

    auto origInType = mlir::cast<NDTypeInterface>(permuteDMAOp.getInput().getType());
    auto origInBuffer = permuteDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto origInBufferOffset = origInBuffer.getByteOffset();

    auto origOutType = mlir::cast<NDTypeInterface>(permuteDMAOp.getOutput().getType());
    auto origOutBuffer = permuteDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto origOutBufferOffset = origOutBuffer.getByteOffset();

    // Super paranoid check
    VPUX_THROW_WHEN(origInType.getRank() != origOutType.getRank(), "Rank mismatch between input type and output type");

    // mem_perm attribute maps out memory dims to in memory dims.
    //
    // The modes of operation of PermuteDMA could be summarized as follows:
    //  - logical transpose -> logical shape changes, but the memory layout remains unchanged
    //      1x2x3x4 NHWC -> 1x3x2x4 NHWC
    //  - memory layout change -> logical shape is unchanged, but the memory layout is different
    //      1x2x3x4 NHWC to 1x2x3x4 NCHW
    //  - logical transpose + memory layout change -> both logical shape and memory layout are different
    //      1x2x3x4 NHWC -> 1x3x2x4 NCHW
    //
    // To make things more explicit:
    //  - memory layout change is represented through the memory layout of the input and output types
    //  - logical transpose is represented through mappingOrder (AffineMap) which maps output logical dims to input
    //  logical dims
    auto mappingOrder =
            getLogicalTransposeFromMemPermute(origInType, origOutType, permuteDMAOp.getMemPermAttr().getAffineMap());
    auto mappingOutToInLogical = VPUIP::getSmallVectorFromAffineMap(mappingOrder);
    auto mappingInToOutLogical = VPUIP::getSmallVectorFromAffineMap(mlir::inversePermutation(mappingOrder));

    auto workingInShape = Shape(origInType.getShape().raw());
    auto origInStrides = origInType.getStrides();

    auto workingOutShape = Shape(origOutType.getShape().raw());
    auto origOutStrides = origOutType.getStrides();

    // Temporary solution to treat case where Expand is fused with Permute, which results in output size > input size
    // (see E#173193). Use input dim sizes and keep output strides.
    for (auto index : irange(origOutType.getRank())) {
        workingOutShape[Dim(index)] = workingInShape[Dim(mappingOutToInLogical[index])];
    }

    // Initialize properly here indexes and task count in case no splittable dim is found
    int64_t inSplitDimIndex = 0;
    int64_t outSplitDimIndex = mappingInToOutLogical[inSplitDimIndex];
    int64_t newTaskCount = 1;

    auto hasPortAssigned = permuteDMAOp.getPort().has_value();

    // Initialize new port
    // All cluster tasks will have port assigned
    // In case of single cluster task, if splitting to all ports is not possible, always use port 0
    int64_t newPort = 0;

    // Split only byte-aligned element types
    if (origInType.getElemTypeSize() % Byte(1).to<Bit>() == 0) {
        // Find a split candidate (search in mem order to find largest continuous chunk)
        for (auto index : VPUIP::getSmallVectorFromAffineMap(origInType.getDimsOrder().toAffineMap(ctx))) {
            // Find the first non-trivial dim (i.e. dim size > 1) that is evenly divided by number of ports
            // This is needed to ensure load balancing, particularly when reading from DDR
            if (workingInShape[Dim(index)] > 1 && workingInShape[Dim(index)] % _dmaPortCount == 0) {
                inSplitDimIndex = index;
                newTaskCount = _dmaPortCount;
                outSplitDimIndex = mappingInToOutLogical[inSplitDimIndex];
                break;
            }
        }
    }

    auto origSplitDimSize = workingInShape[Dim(inSplitDimIndex)];
    auto initialSplitDimSize = workingInShape[Dim(inSplitDimIndex)] / newTaskCount;
    auto currentSplitDimSize = initialSplitDimSize;

    // Offsets needed for per-axis quant type updates
    auto workingInShapeOffsets = Shape(workingInShape.size());
    auto workingOutShapeOffsets = Shape(workingOutShape.size());

    const auto getNewElementType = [](NDTypeInterface origType, ShapeRef newShape, ShapeRef newOffset) {
        auto elemType = origType.getElementType();
        if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            // tileScalesAndZP will update type only when we actually tiled over quantized axis
            elemType = tileScalesAndZP(qType, newShape, newOffset);
        }

        return elemType;
    };

    const auto createNewBuffer = [](mlir::PatternRewriter& rewriter, VPURT::TaskOp taskOp,
                                    VPURT::DeclareBufferOp existingBuffer, NDTypeInterface newType, int64_t newOffset) {
        if (newType.getMemSpace().getIndex().has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, existingBuffer, taskOp.getLoc(), newType,
                                                           existingBuffer.getSection(),
                                                           newType.getMemSpace().getIndex().value(), newOffset);
        } else {
            if (existingBuffer.getSectionIndex().has_value()) {
                return VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, existingBuffer, taskOp.getLoc(), newType, existingBuffer.getSection(),
                        parseIntArrayAttr<int64_t>(existingBuffer.getSectionIndex().value()), newOffset);
            }

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, existingBuffer, taskOp.getLoc(), newType,
                                                           existingBuffer.getSection(), newOffset);
        }
    };

    const auto getNewInType = [&getNewElementType](NDTypeInterface origType, ShapeRef newShape, ShapeRef newOffsets,
                                                   StridesRef newStrides) -> NDTypeInterface {
        auto newElementType = getNewElementType(origType, newShape, newOffsets);

        VPUX_THROW_UNLESS(mlir::isa_and_nonnull<mlir::MemRefType>(origType), "Unexpected input type");

        return getMemRefType(newShape, newElementType, origType.getDimsOrder(), origType.getMemSpace(), newStrides);
    };

    const auto getNewOutType = [&getNewElementType](mlir::MLIRContext* ctx, NDTypeInterface origType, ShapeRef newShape,
                                                    ShapeRef newOffsets, StridesRef newStrides) -> NDTypeInterface {
        auto newElementType = getNewElementType(origType, newShape, newOffsets);

        if (auto dstDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(origType)) {
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PermuteNNDMA; Buffer has distributed type != DUPLICATED after unroll");

            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        newShape, distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(), nullptr,
                        distributionAttr.getUniformDistributedSegments(), ctx);
            }

            const auto layout = mlir::AffineMapAttr::get(origType.getDimsOrder().toAffineMap(ctx));
            return mlir::cast<NDTypeInterface>(VPUIP::DistributedBufferType::get(ctx, newShape, newElementType, layout,
                                                                                 origType.getMemSpace(),
                                                                                 distributionAttr))
                    .changeStrides(newStrides);
        }

        return getMemRefType(newShape, newElementType, origType.getDimsOrder(), origType.getMemSpace(), newStrides);
    };

    // Initialize variables here to allow loop to handle single iteration cases
    // We update the buffers and types since we are switching to identity logical to memory mapping
    auto newInType = getNewInType(origInType, workingInShape, workingInShapeOffsets, origInStrides);
    auto newInBuffer = createNewBuffer(rewriter, origTaskOp, origInBuffer, newInType, origInBufferOffset);

    auto newOutType = getNewOutType(ctx, origOutType, workingOutShape, workingOutShapeOffsets, origOutStrides);
    auto newOutBuffer = createNewBuffer(rewriter, origTaskOp, origOutBuffer, newOutType, origOutBufferOffset);

    for (auto index : irange(newTaskCount)) {
        if (newTaskCount > 1) {
            if (index == newTaskCount - 1) {
                // For last iter use remaining size for cases where dim size is not divisible nicely
                currentSplitDimSize = origSplitDimSize - initialSplitDimSize * index;
            }
            newPort = index;

            // Compute new shapes
            workingInShape[Dim(inSplitDimIndex)] = currentSplitDimSize;
            workingOutShape[Dim(outSplitDimIndex)] = currentSplitDimSize;

            // Compute new offsets.
            // For simplicity, the splitting interleaves accesses to the original shapes.
            // Pretty heavy assumption here that strides will turn out to be byte aligned.
            // Jump only over elements in the dimension we split.
            // For a compact shape, if we split over the highest order dim, the access will be continuous.
            auto newInBufferOffset =
                    initialSplitDimSize * origInStrides[Dim(inSplitDimIndex)].to<Byte>().count() * index +
                    origInBufferOffset;
            auto newOutBufferOffset =
                    initialSplitDimSize * origOutStrides[Dim(outSplitDimIndex)].to<Byte>().count() * index +
                    origOutBufferOffset;

            workingInShapeOffsets[Dim(inSplitDimIndex)] = initialSplitDimSize * index;
            newInType = getNewInType(origInType, workingInShape, workingInShapeOffsets, origInStrides);
            newInBuffer = createNewBuffer(rewriter, origTaskOp, newInBuffer, newInType, newInBufferOffset);

            workingOutShapeOffsets[Dim(outSplitDimIndex)] = initialSplitDimSize * index;
            newOutType = getNewOutType(ctx, origOutType, workingOutShape, workingOutShapeOffsets, origOutStrides);
            newOutBuffer = createNewBuffer(rewriter, origTaskOp, newOutBuffer, newOutType, newOutBufferOffset);
        }

        auto loopOrder =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(VPUIP::getLinearMemOrder(newInType), ctx));

        auto internalDataFlowAttr = VPUIP::InternalDataFlowAttr::get(ctx, newInType, newOutType,
                                                                     mlir::AffineMapAttr::get(mappingOrder), loopOrder);

        const auto newLoc = appendLoc(origTaskOp->getLoc(), "_unrolled_permuteDMA");

        // Override port if no splitting can be done and port was already assigned by cluster unrolling
        if (hasPortAssigned && newTaskCount == 1) {
            newPort = permuteDMAOp.getPort().value();
        }

        VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), newLoc, newInBuffer,
                newOutBuffer, getIntAttr(rewriter, newPort), permuteDMAOp.getIsOutOfOrderAttr(),
                permuteDMAOp.getIsCriticalAttr(),
                /*mem_perm*/ nullptr, /* dma_descriptor */ nullptr, permuteDMAOp.getDmaHwpIdAttr(),
                permuteDMAOp.getProfilingMetadataAttr(), internalDataFlowAttr);
    }

    rewriter.eraseOp(origTaskOp);
    return mlir::success();
}

MultiClusterPermuteDMARewriter::MultiClusterPermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
        : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("MultiClusterPermuteDMARewriter");
}

mlir::LogicalResult MultiClusterPermuteDMARewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteDMAOp.getMemPermAttr() == nullptr) {
        return mlir::failure();
    }

    if (isMultiClusterPermuteDMA(permuteDMAOp)) {
        // Unroll multi-cluster task
        _log.trace("Got PermuteDMAOp '{0}' at '{1}'", permuteDMAOp->getName(), permuteDMAOp->getLoc());

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getOutputBuff().getType());

        auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(permuteDMAOp.getInput().getType());
        auto outDistributedType =
                mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(permuteDMAOp.getOutputBuff().getType());

        VPUX_THROW_UNLESS(permuteDMAOp.getMemPerm().has_value(),
                          "Can not get memPerm attribute from PermuteDMA layer at {0}", permuteDMAOp.getLoc());
        const auto memPerm = permuteDMAOp.getMemPerm().value();

        if (inDistributedType != nullptr && outDistributedType != nullptr) {
            return unrollDuplicatedInputAndOutput(permuteDMAOp, memPerm, rewriter);
        } else if (inDistributedType != nullptr) {
            return unrollDuplicatedInput(permuteDMAOp, memPerm, rewriter);
        }

        VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                                  outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Unexpected memory space. Got: input {0}, output {1}", inputType.getMemoryKind(),
                          outputType.getMemoryKind());

        VPUX_THROW_WHEN(outDistributedType == nullptr, "Expect distributed type for permute op output, actual: {0}",
                        outputType);

        VPUX_THROW_UNLESS(VPUIP::doesPermuteDMATileDimSupportWrapInCluster(inputType, outputType, memPerm,
                                                                           outDistributedType, _log),
                          "Unsupported PermuteDMA under cluster tiling at '{0}'", permuteDMAOp->getLoc());

        const auto distributionAttr = outDistributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return unrollSegmentedOrOverlappedOutput(permuteDMAOp, outDistributedType, memPerm, rewriter);
        } else if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            return unrollDuplicatedOutput(permuteDMAOp, outDistributedType, memPerm, rewriter);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    }

    return mlir::failure();
}

mlir::LogicalResult MultiClusterPermuteDMARewriter::unrollSegmentedOrOverlappedOutput(
        VPUIP::PermuteDMAOp permuteDMAOp, VPUIP::DistributedBufferType distributedType, mlir::AffineMap memPerm,
        mlir::PatternRewriter& rewriter) const {
    auto loc = permuteDMAOp->getLoc();
    auto ctx = permuteDMAOp->getContext();

    const auto input = permuteDMAOp.getInput();
    const auto output = permuteDMAOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto originalOutputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto outputType = distributedType.getCompactType();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distributed mode");
    const auto perClusterOutShapes = distributedType.getPerClusterMemoryShapes();
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto vpurtTask = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteDMAOp);

    const auto tileInputType = [&](vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] = getPerClusterInputType(inputType, outputType, memPerm, perClusterOutShapes[clusterId],
                                                         perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    const auto tileOutputType = [&](vpux::NDTypeInterface outputType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(outputType, perClusterOutShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    auto inTypes = tileInputType(inputType, outputType);
    const auto originStride = inputType.getStrides();

    for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
        inTypes[clusterId] = inTypes[clusterId].changeStrides(originStride);
    }

    const auto outTypes = tileOutputType(outputType);

    rewriter.setInsertionPointAfter(vpurtTask);
    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteDMAOp->getLoc(), cst,
                                                     perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterOutShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteDMAOp->getLoc(), newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(ctx, ArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.getByteOffset()};
        ddrOffset += offset;

        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, ddrOffset);
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptors and ddr offsets
    SmallVector<VPUIP::DMADescriptorAttr> subDmaDescriptors;
    SmallVector<Byte> ddrOffsets;
    SmallVector<Shape> subMergedOutputShapes;

    const auto mergedOutputDimList = VPUIP::getPermuteDMAOutputMergedDimList(outputType, mergedOutputShape);
    auto tileDimForMergedOutput =
            VPUIP::getTileDimForPermuteDMA(inputType, outputType, memPerm, distributedType, _log).value();

    const auto numTileSize = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tileDimIter = std::find_if(numTileSize.begin(), numTileSize.end(), [](const int64_t dim) {
        return dim > 1;
    });
    VPUX_THROW_UNLESS(tileDimIter != numTileSize.end(), "Can not find tile dim.");
    auto tileDim = Dim(std::distance(numTileSize.begin(), tileDimIter));

    auto getSrcOffset = [&](vpux::ShapeRef offset) -> vpux::Byte {
        auto outputShape = originalOutputType.getShape();

        const auto splitDimList = mergedOutputDimList[tileDimForMergedOutput.ind()];
        VPUX_THROW_UNLESS(std::any_of(splitDimList.begin(), splitDimList.end(),
                                      [&](vpux::Dim dim) {
                                          return dim == tileDim;
                                      }),
                          "tileDim is not exist in splitDimList.");

        const auto totalOffsetSize = mergedOutputShape[tileDimForMergedOutput];
        return Byte(totalOffsetSize / outputShape[tileDim] * offset[tileDim] * elemTypeSize.count());
    };

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto mergedSubOutputShape =
                VPUIP::getPermuteDMAOutputShape(inTypes[clusterId], outTypes[clusterId], memPerm, _log).value();
        ddrOffsets.push_back(getSrcOffset(perClusterShapeOffsets[clusterId]));
        subMergedOutputShapes.push_back(mergedSubOutputShape);
    }
    subDmaDescriptors = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, subMergedOutputShapes,
                                                        tileDimForMergedOutput, elemTypeSize);

    int64_t dmaPort = 0;
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint, ddrOffsets[clusterId]);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint, Byte(0));
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, dmaPort), permuteDMAOp.getIsOutOfOrderAttr(),
                permuteDMAOp.getIsCriticalAttr(), permuteDMAOp.getMemPermAttr(), subDmaDescriptors[clusterId],
                permuteDMAOp.getDmaHwpIdAttr(), permuteDMAOp.getProfilingMetadataAttr(), /*internalDataFlow=*/nullptr);

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

mlir::LogicalResult MultiClusterPermuteDMARewriter::unrollDuplicatedOutput(VPUIP::PermuteDMAOp permuteDMAOp,
                                                                           VPUIP::DistributedBufferType distributedType,
                                                                           mlir::AffineMap memPerm,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto loc = permuteDMAOp->getLoc();
    auto ctx = permuteDMAOp->getContext();

    const auto input = permuteDMAOp.getInput();
    const auto output = permuteDMAOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getOutputBuff().getType());

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", distributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteDMAOp);

    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::DUPLICATED, "Unsupported distributed mode");

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto perClusterOutShape = distributedType.getPerClusterMemoryShapes().front();
    const auto perClusterShapeOffset = distributedType.getPerClusterMemoryShapeOffsets().front();

    const auto getOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteDMAOp->getLoc(), cst, perClusterShapeOffset.raw(),
                                                     perClusterOutShape.raw());
        }
        auto insertionPoint = operand.getDefiningOp();

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteDMAOp->getLoc(), newType,
                                                           VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, clusters),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }
        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, Byte(0));
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
    const auto newInputType =
            getPerClusterInputType(inputType, outputType, memPerm, perClusterOutShape, perClusterShapeOffset);

    const auto changeShapeElemTypeForDistributedBuff = [](VPUIP::DistributedBufferType buff, ShapeRef shape,
                                                          mlir::Type elemType) {
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(buff.getDistribution())) {
            auto distribution = buff.getDistribution();
            VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                            "DistributedBuffer has mode different from DUPLICATED after unrolling");
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    shape, distribution.getMode(), nullptr, distribution.getNumClusters(), nullptr,
                    distribution.getUniformDistributedSegments(), buff.getContext());
            return buff.changeShapeElemTypeForExplicitDistribution(shape, elemType, newDistribution);
        }

        return buff.changeShapeElemType(shape, elemType);
    };

    auto newOutType = changeShapeElemTypeForDistributedBuff(distributedType, perClusterOutShape,
                                                            distributedType.getElementType());

    const auto inputBuffer = getOperand(input, newInputType);
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);
    const auto outBuffer = getOperand(output, newOutType);
    _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outBuffer,
            vpux::getIntAttr(rewriter, 0), permuteDMAOp.getIsOutOfOrderAttr(), permuteDMAOp.getIsCriticalAttr(),
            permuteDMAOp.getMemPermAttr(), subDmaDescriptor, permuteDMAOp.getDmaHwpIdAttr(),
            permuteDMAOp.getProfilingMetadataAttr(), /*internalDataFlow=*/nullptr);

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

mlir::LogicalResult MultiClusterPermuteDMARewriter::unrollDuplicatedInputAndOutput(
        VPUIP::PermuteDMAOp permuteDMAOp, mlir::AffineMap memPerm, mlir::PatternRewriter& rewriter) const {
    auto loc = permuteDMAOp->getLoc();
    auto ctx = permuteDMAOp->getContext();

    const auto input = permuteDMAOp.getInput();
    const auto output = permuteDMAOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    const auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(input.getType());
    const auto outDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());

    const auto inMode = inDistributedType.getDistribution().getMode().getValue();
    const auto outMode = outDistributedType.getDistribution().getMode().getValue();
    VPUX_THROW_UNLESS(VPU::bitEnumContainsAny(inMode, VPU::DistributionMode::DUPLICATED) &&
                              VPU::bitEnumContainsAny(outMode, VPU::DistributionMode::DUPLICATED),
                      "Unsupported mode");

    const auto distributionAttr = outDistributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", outDistributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteDMAOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);

    // create new input buffer
    auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, 0)});
    const auto inType = mlir::cast<vpux::NDTypeInterface>(inDistributedType.getCompactType());
    const auto newInType = inType.changeMemSpace(symbolAttr);
    auto inputBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({0})),
            inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

    // create new output buffer
    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer offset");
    auto outBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(ctx, ArrayRef(clusters)), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outBuffer,
            vpux::getIntAttr(rewriter, 0), permuteDMAOp.getIsOutOfOrderAttr(), permuteDMAOp.getIsCriticalAttr(),
            permuteDMAOp.getMemPermAttr(), subDmaDescriptor, permuteDMAOp.getDmaHwpIdAttr(),
            permuteDMAOp.getProfilingMetadataAttr(), /*internalDataFlow=*/nullptr);

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

mlir::LogicalResult MultiClusterPermuteDMARewriter::unrollDuplicatedInput(VPUIP::PermuteDMAOp permuteDMAOp,
                                                                          mlir::AffineMap memPerm,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto loc = permuteDMAOp->getLoc();
    auto ctx = permuteDMAOp->getContext();

    const auto input = permuteDMAOp.getInput();
    const auto output = permuteDMAOp.getOutputBuff();

    const auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(input.getType());
    const auto inMode = inDistributedType.getDistribution().getMode().getValue();
    VPUX_THROW_UNLESS(VPU::bitEnumContainsAny(inMode, VPU::DistributionMode::DUPLICATED), "Unsupported mode");

    const auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(inDistributedType.getCompactType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    auto vpurtTask = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteDMAOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);

    // create new input buffer
    auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, 0)});
    const auto newInType = inputType.changeMemSpace(symbolAttr);
    auto inputBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({0})),
            inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

    // create new output buffer
    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outDeclBuff,
            vpux::getIntAttr(rewriter, 0), permuteDMAOp.getIsOutOfOrderAttr(), permuteDMAOp.getIsCriticalAttr(),
            permuteDMAOp.getMemPermAttr(), subDmaDescriptor, permuteDMAOp.getDmaHwpIdAttr(),
            permuteDMAOp.getProfilingMetadataAttr(), /*internalDataFlow=*/nullptr);

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

}  // namespace vpux::VPUIP
