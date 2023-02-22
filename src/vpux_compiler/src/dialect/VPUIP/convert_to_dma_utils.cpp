//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/dma_descriptor_generator.hpp"

using namespace vpux;
namespace {
// Get correct permute from reversed permute value. The reversed permute value is expect from SW.Kernel op's attribute.
// For example, The perm [0,1,2,3] -> [0,2,3,1] is represented by [2,0,1,3] in SW.Kernel op, so the output is
// supposed to be [0,2,3,1]
SmallVector<unsigned> correctPermutation(ArrayRef<unsigned> revPerm) {
    SmallVector<unsigned> origPerm(revPerm.size());
    for (const auto srcInd : irange(revPerm.size())) {
        const auto revSrcInd = revPerm.size() - 1 - srcInd;
        const auto revDstInd = revPerm[revSrcInd];
        origPerm[srcInd] = static_cast<unsigned>(revPerm.size()) - 1 - revDstInd;
    }
    return origPerm;
}

/**
 * Cost function to evaluate whether it's beneficial to implement the operation using DMA rather than UPA for
 * operations like MemPermute.
 * @return true if it's beneficial for using DMA, otherwise false.
 */
bool isBeneficialForUsingPermuteDMA(NDTypeInterface inType, NDTypeInterface outType, mlir::AffineMap memPerm,
                                    VPU::ArchKind arch, vpux::Logger log) {
    auto subShapes = VPUIP::getPermuteDMASubInputShapes(inType, outType, memPerm, log);
    if (!subShapes.hasValue()) {
        return false;
    }

    if (arch == VPU::ArchKind::VPUX30XX) {
        // This is a empirical value to set limitation for DMA number.
        // In some specific case, for example: 1x8x256x256 #NHWC  ->  1x8x256x256 #NCHW
        // This permuteUPA should be replaced with 256 DMA. Each DMA just with size 8x256. It is inefficient.
        // It's supposed to get the related DMA and UPA cost by VPUNN in future, please refer ticket #41221.
        if (subShapes.getValue().size() > VPUIP::PER_PERMUTE_MAX_DMA_NUMBER) {
            log.trace("PermuteDMA number should less than {0}, but got {1}.", VPUIP::PER_PERMUTE_MAX_DMA_NUMBER,
                      subShapes.getValue().size());
            return false;
        }

        return llvm::all_of(subShapes.getValue(), [](ShapeRef shape) {
            const auto C = shape[Dims4D::Act::C];
            // This is a empirical value to set limitation for the src_width in the DMA descriptor. For dst_with
            // fixed as element size, with smaller src_width, performance tends to be better. It's supposed to get the
            // related DMA and UPA cost by VPUNN in future, please refer ticket #41221.
            return C < VPUIP::PERMUTE_DMA_MAX_LENGTH;
        });
    }
    return true;
}

SmallVector<Shape> computeDMASubShape(ShapeRef shape, Dim numPlaneDim) {
    const auto shapeSize = shape.size();
    VPUX_THROW_UNLESS(shapeSize == 2 || shapeSize == 3, "Shape size should be 2 or 3, but got {0}", shapeSize);
    VPUX_THROW_UNLESS(static_cast<size_t>(numPlaneDim.ind()) < shapeSize,
                      "numPlaneDim index {0} doesn't match shape size {1}", numPlaneDim.ind(), shapeSize);
    const auto totalNumPlane = shape[numPlaneDim];
    auto numberDMAs = divUp(totalNumPlane, VPUIP::DMA_MAX_NUMBER_PLANES);
    auto subShape = Shape(shape.raw());
    if (numberDMAs > 1) {
        subShape[numPlaneDim] = VPUIP::DMA_MAX_NUMBER_PLANES;
        SmallVector<Shape> subOutputShapes(numberDMAs - 1, subShape);
        subShape[numPlaneDim] = totalNumPlane - VPUIP::DMA_MAX_NUMBER_PLANES * (numberDMAs - 1);
        subOutputShapes.push_back(subShape);
        return subOutputShapes;
    }
    return SmallVector<Shape>{subShape};
}
}  // namespace

// In order to simplify the difference cases about input layout and mem perm, the merged input shape need to be
// calculated. For example,
// [1, 4, 16, 16] #NCHW -> [1, 4, 16, 16] #NHWC, memPerm=[d0, d2, d3, d1] can be merged into
// [4, 256] #NC -> [256, 4] #NC, memPerm=[d1, d0]
Optional<Shape> vpux::VPUIP::getPermuteDMAInputShape(NDTypeInterface inType, NDTypeInterface outType,
                                                     mlir::AffineMap perm, vpux::Logger log) {
    if (!perm.isPermutation()) {
        log.trace("Permute op with input {0}, output {1} doesn't support DMA with memPerm {2}", inType, outType, perm);
        return None;
    }
    auto inShape = inType.getShape();
    auto outShape = outType.getShape();
    auto inputMemShape = Shape(inType.getMemShape().raw());
    auto memPerm = DimsOrder::fromAffineMap(perm);
    Shape newInputShape;
    int64_t shapeSize = 1;
    for (size_t idx = 0; idx < inputMemShape.size(); idx++) {
        shapeSize *= inputMemShape[memPerm.dimAt(idx)];

        if (idx + 1 == inputMemShape.size() || memPerm.dimAt(idx).ind() + 1 != memPerm.dimAt(idx + 1).ind()) {
            if (shapeSize != 1) {
                newInputShape.push_back(shapeSize);
            }
            shapeSize = 1;
        }
    }

    if (newInputShape.size() == 1) {
        return Shape{1, newInputShape.front()};
    }

    if (newInputShape.size() == 3) {
        auto mergedMemPerm = getPermuteDMAMergedMemPerm(inType, perm);
        auto ctx = inType.getContext();
        // The data order of newInputshape has appiled permutation. So we need reverse it according the permute map.
        if (mergedMemPerm == mlir::AffineMap::getPermutationMap({1, 0, 2}, ctx)) {
            // Check for permute pattern: HWC->WHC
            return Shape{newInputShape[Dim(1)], newInputShape[Dim(0)], newInputShape[Dim(2)]};
        } else if (mergedMemPerm == mlir::AffineMap::getPermutationMap({0, 2, 1}, ctx)) {
            // Check for permute pattern: HWC->HCW
            return Shape{newInputShape[Dim(0)], newInputShape[Dim(2)], newInputShape[Dim(1)]};
        } else {
            return None;
        }
    }

    if (newInputShape.size() > 3) {
        log.trace("Can't convert Permute to DMA with inshape {0}, outshape {1}, memPerm {2}.", inShape, outShape,
                  memPerm);
        return None;
    }

    return Shape{newInputShape.back(), newInputShape.front()};
}

Optional<Shape> vpux::VPUIP::getPermuteDMAOutputShape(NDTypeInterface inType, NDTypeInterface outType,
                                                      mlir::AffineMap perm, vpux::Logger log) {
    auto mergedInputShape = getPermuteDMAInputShape(inType, outType, perm, log);
    if (!mergedInputShape.hasValue()) {
        return None;
    }
    auto mergedOutputShape = getPermuteDMASubOutputShapes({mergedInputShape.getValue()}, inType, outType, perm).front();
    return mergedOutputShape;
}

Optional<SmallVector<Shape>> vpux::VPUIP::getPermuteDMASubInputShapes(NDTypeInterface inType, NDTypeInterface outType,
                                                                      mlir::AffineMap perm, vpux::Logger log) {
    if (!perm.isPermutation()) {
        log.trace("PermuteOp doesn't support permutation.");
        return None;
    }

    auto newInputShape = getPermuteDMAInputShape(inType, outType, perm, log);
    if (!newInputShape.hasValue()) {
        return None;
    }

    auto numPlaneDim = getPermuteDMANumPlaneDim(inType, perm);
    return computeDMASubShape(newInputShape.getValue(), numPlaneDim);
}

SmallVector<vpux::Shape> vpux::VPUIP::getPermuteDMASubOutputShapes(SmallVector<vpux::Shape> subInputShapes,
                                                                   vpux::NDTypeInterface inType,
                                                                   vpux::NDTypeInterface outType,
                                                                   mlir::AffineMap memPerm) {
    SmallVector<vpux::Shape> subOutputShapes;
    auto outputChannel = outType.getShape()[Dims4D::Act::C];

    for (auto subInput : subInputShapes) {
        auto inShape = to_small_vector(subInput);
        // After Expand fuse into Permute and got one PermuteDMA Op
        // The input shape is not same with the output shape
        // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
        // PermuteDMA input (3x1024), output (1024x16)
        if (inType.getShape().totalSize() != outType.getShape().totalSize()) {
            VPUX_THROW_UNLESS(inShape.size() == 2 && inType.getDimsOrder() == DimsOrder::NCHW &&
                                      outType.getDimsOrder() == DimsOrder::NHWC,
                              "PermuteDMA with unsupport input {0} output {1} type.", inType, outType);
            inShape.front() = outputChannel;
        }
        if (inShape.size() == 2) {
            subOutputShapes.push_back(Shape(SmallVector<int64_t>{inShape.back(), inShape.front()}));
        } else if (inShape.size() == 3) {
            auto mergedMemPerm = getPermuteDMAMergedMemPerm(inType, memPerm);
            auto ctx = inType.getContext();
            if (mergedMemPerm == mlir::AffineMap::getPermutationMap({1, 0, 2}, ctx)) {
                subOutputShapes.push_back(Shape(SmallVector<int64_t>{inShape[1], inShape[0], inShape[2]}));
            } else if (mergedMemPerm == mlir::AffineMap::getPermutationMap({0, 2, 1}, ctx)) {
                subOutputShapes.push_back(Shape(SmallVector<int64_t>{inShape[0], inShape[2], inShape[1]}));
            } else {
                VPUX_THROW("unsupported inShape {0} with memPerm {0}", inShape, memPerm);
            }
        } else {
            VPUX_THROW("unsupported inShape {0}", inShape);
        }
    }

    return subOutputShapes;
}

// Get the real permutation map. For the memPerm provided by the permute op, some dims can be merged or ignored.
// If the dim size is 1, so the real permutation can just ignore this dim.
//     [1, 4, 224] -> [1, 224, 4], memPerm [d0, d2, d1] can be converted as [4, 224] -> [224, 4] with memPerm [d1, d0]
// If the permute dims in sequence, those dims can be merged into one.
//     [2, 3, 4] -> [3, 4, 2], memPerm [d1, d2, d0] can be converted as [2, 12] -> [12, 2] with memPerm [d1, d0]
mlir::AffineMap vpux::VPUIP::getPermuteDMAMergedMemPerm(vpux::NDTypeInterface inType, mlir::AffineMap memPerm) {
    auto inputMemShape = Shape(inType.getMemShape().raw());

    // get permute map dim list which dim size not equal to 1
    SmallVector<int64_t> mergedPermuteArray;
    for (size_t idx = 0; idx < memPerm.getNumResults(); idx++) {
        if (inputMemShape[Dim(memPerm.getDimPosition(idx))] != 1) {
            mergedPermuteArray.push_back(memPerm.getDimPosition(idx));
        }
    }

    auto sortIndex = [&mergedPermuteArray]() {
        SmallVector<unsigned> sortIndexArray(mergedPermuteArray.size());
        std::iota(sortIndexArray.begin(), sortIndexArray.end(), 0);
        llvm::sort(sortIndexArray, [&mergedPermuteArray](auto a, auto b) {
            return mergedPermuteArray[a] < mergedPermuteArray[b];
        });
        return sortIndexArray;
    };

    // Sort permute dim index. For example, [d2 d3 d1] wil be sorted as [d1 d2 d0]
    auto sortedPermuteMap = sortIndex();

    // Merge dims in sequence. For example, [d1 d2 d0] wil be merged as [d1 d0]
    mergedPermuteArray.clear();
    for (size_t idx = 0; idx < sortedPermuteMap.size(); idx++) {
        if (idx == 0 || sortedPermuteMap[idx - 1] + 1 != sortedPermuteMap[idx]) {
            mergedPermuteArray.push_back(sortedPermuteMap[idx]);
        }
    }
    return mlir::AffineMap::getPermutationMap(sortIndex(), inType.getContext());
}

// Get the numPlane dim of the merged input shape. Since the dma descriptor has limitation on the value of numPlane, so
// here the compiler needs to get the numPlane dim to find the numPlane size. So that the PermuteDMA ops can be splited
// into several small ones later. For example, merged input shape [3, 112] which means there is 3 planes for dma and the
// numPlane dim could be d0.
Dim vpux::VPUIP::getPermuteDMANumPlaneDim(vpux::NDTypeInterface inType, mlir::AffineMap memPerm) {
    auto ctx = inType.getContext();
    auto mergedPerm = getPermuteDMAMergedMemPerm(inType, memPerm);

    if (mergedPerm == mlir::AffineMap::getPermutationMap({0, 2, 1}, ctx)) {
        return Dim(1);
    }
    return Dim(0);
}

SmallVector<DimArr> vpux::VPUIP::getPermuteDMAOutputMergedDimList(vpux::NDTypeInterface outputType,
                                                                  ShapeRef mergedOutputShape) {
    auto outShape = outputType.getShape();
    auto outOrder = outputType.getDimsOrder();

    auto outRealShape = outOrder.toMemoryOrder(outShape);
    auto outRealShapeVal = to_small_vector(outRealShape);
    auto mergedOutputShapeVal = to_small_vector(mergedOutputShape);

    SmallVector<DimArr> mergedDims;
    // Calculate the multiply result for outRealShapVal[begin, end]
    auto multiplyShapeFunc = [&outRealShapeVal](size_t begin, size_t end) {
        return std::accumulate(outRealShapeVal.begin() + begin, outRealShapeVal.begin() + end + 1,
                               static_cast<int64_t>(1), std::multiplies<int64_t>());
    };

    size_t curIdx = 0;
    for (auto shapeSize : mergedOutputShapeVal) {
        if (curIdx >= outRealShapeVal.size()) {
            break;
        }
        size_t endIdx = curIdx;
        for (; endIdx < outRealShapeVal.size(); endIdx++) {
            if (multiplyShapeFunc(curIdx, endIdx) == shapeSize) {
                break;
            }
        }
        VPUX_THROW_UNLESS(endIdx < outRealShapeVal.size(), "Can not find merged dim size {0} from memShape {1}",
                          shapeSize, outRealShapeVal);
        DimArr dims;
        for (auto idx = curIdx; idx < endIdx + 1; idx++) {
            dims.push_back(outOrder.dimAt(idx));
        }
        mergedDims.push_back(dims);
        curIdx = endIdx + 1;
    }
    return mergedDims;
}

// Get the tiling axis of the merged output shape. When the axis is on the highest dim, the dma descriptor can be
// set correctly. For example, [1, 4, 16, 16] #NCHW @DDR  -> [1, 4, 8, 16] #NHWC [@CMX, 0]
//                                                           [1, 4, 8, 16] #NHWC [@CMX, 1]
// The merged output shape is [128, 4], #NHWC, [@CMX, 0]
//                            [128, 4], #NHWC, [@CMX, 1]
// The merged output dims is [[d0, d2, d3], [d1]], since shape size on d0 is 1, so the tile dim d2 is the highest
// dim on the first dim list [d0, d2, d3].

// [1, 4, 16, 16] #NCHW @DDR  -> [1, 2, 16, 16] #NHWC [@CMX, 0]
//                               [1, 2, 16, 16] #NHWC [@CMX, 1]
// The merged output shape is [256, 2], #NHWC, [@CMX, 0]
//                            [256, 2], #NHWC, [@CMX, 1]
// The merged output dims is [[d0, d2, d3], [d1]], so the tile dim d1 is the highest dim on the second dim list [d1]

// [1, 4, 16, 16] #NCHW @DDR  -> [1, 4, 16, 8] #NHWC [@CMX, 0]
//                               [1, 4, 16, 8] #NHWC [@CMX, 1]
// The merged output shape is [128, 4], #NHWC, [@CMX, 0]
//                            [128, 4], #NHWC, [@CMX, 1]
// The merged output dims is [[d0, d2, d3], [d1]], the tile dim d3 is not the highest dim on the second dim list [d0,
// d2, d3], and we cannot get the dma descriptor for it
Optional<Dim> vpux::VPUIP::getTileDimForPermuteDMA(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                   mlir::AffineMap memPerm,
                                                   VPUIP::DistributedBufferType distributedOutputType,
                                                   vpux::Logger log) {
    const auto distributionAttr = distributedOutputType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unexpected distributed mode {0}", VPU::stringifyEnum(mode));
    const auto outputShape = outType.getShape();
    const auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, log).getValue();
    const auto mergedOutputDimList = VPUIP::getPermuteDMAOutputMergedDimList(outType, mergedOutputShape);

    VPUX_THROW_UNLESS(mergedOutputDimList.size() == 2 || mergedOutputDimList.size() == 3,
                      "Unexpected merged dim list {0}", mergedOutputDimList);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
    const auto findHighestDim = [&outputShape](Dim dim) {
        return outputShape[dim] > 1;
    };
    auto isValidOnDim = [&](ArrayRef<Dim> mergedDims) {
        auto highestDim = llvm::find_if(mergedDims, findHighestDim);
        return highestDim != mergedDims.end() && *highestDim == Dim(axis);
    };
    for (auto idx : irange(mergedOutputDimList.size())) {
        if (isValidOnDim(mergedOutputDimList[idx])) {
            return Dim(idx);
        }
    }
    return None;
}

bool vpux::VPUIP::doesPermuteDMATileDimSupportWrapInCluster(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                            mlir::AffineMap memPerm,
                                                            VPUIP::DistributedBufferType distributedOutputType,
                                                            vpux::Logger log) {
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, log).getValue();
    auto mergedDims = VPUIP::getPermuteDMAOutputMergedDimList(outType, mergedOutputShape);
    VPUX_THROW_UNLESS(mergedDims.size() == 2 || mergedDims.size() == 3, "Invalid dims size, get {0}",
                      mergedDims.size());

    const auto distributionAttr = distributedOutputType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    // Disable duplicate permute since there is performance regression for some models. Need set up a cost function in
    // future to evaluate the dma cost and decide to fuse permute into cluster or not
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return false;
    }

    if (mergedDims.size() == 3) {
        return false;
    }

    auto tileDim = getTileDimForPermuteDMA(inType, outType, memPerm, distributedOutputType, log);
    return tileDim.hasValue();
}

bool vpux::VPUIP::isCombineAtFront(ShapeRef shape, DimsOrder order) {
    for (size_t idx = 0; idx < shape.size(); idx++) {
        if (shape[order.dimAt(idx)] == 1) {
            continue;
        }
        return shape[order.dimAt(idx)] <= DMA_MAX_NUMBER_PLANES;
    }
    return false;
}

bool vpux::VPUIP::doesSWLayerFitIntoCMX(mlir::Operation* op, vpux::Logger log) {
    if (!mlir::isa<VPUIP::PermuteUPAOp, IE::DepthToSpaceOp, VPUIP::DepthToSpaceUPAOp, IE::SpaceToDepthOp,
                   VPUIP::SpaceToDepthUPAOp, VPUIP::PerAxisTileUPAOp, VPUIP::SwKernelOp>(op)) {
        log.trace("unsupported op type at '{0}'", op->getLoc());
        return false;
    }
    if (mlir::isa<VPUIP::SwKernelOp>(op)) {
        // SwKernelOp should be tiled to fit CMX
        return true;
    }
    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    Byte requiredCMX(0);
    requiredCMX += inputType.getTotalAllocSize();
    requiredCMX += outputType.getTotalAllocSize();
    if (requiredCMX > VPU::getTotalCMXSize(op)) {
        log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(op), requiredCMX);
        return false;
    }
    return true;
}

bool vpux::VPUIP::isLegalConvertToDMA(mlir::Operation* op, vpux::Logger log) {
    if (VPU::getCompilationMode(op) != VPU::CompilationMode::DefaultHW) {
        return false;
    }
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<VPUIP::PermuteUPAOp>([&](VPUIP::PermuteUPAOp op) {
                log.trace("Got Permute Op at {0}.", op->getLoc());

                const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
                const auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();
                const auto memPerm = op.order_value();

                if (!VPUIP::getPermuteDMASubInputShapes(inputType, outputType, memPerm, log).hasValue()) {
                    log.trace("MemPermute Op at {0} doesn't support DMA implementation.", op->getLoc());
                    return false;
                }

                if (!VPUIP::doesSWLayerFitIntoCMX(op, log)) {
                    log.trace("Memory size of SW Op at {0} is larger than CMX, can not move to CMX.", op->getLoc());
                    return false;
                }

                log.trace("PermuteOp at {0} can convert to PermuteDMAOp.", op->getLoc());
                return true;
            })
            .Case<IE::DepthToSpaceOp, VPUIP::DepthToSpaceUPAOp>([&](mlir::Operation* op) {
                log.trace("Got DepthToSpaceOp Op at {0}.", op->getLoc());
                const auto arch = VPU::getArch(op);

                // Memory check only for dKMB
                if (arch == VPU::ArchKind::VPUX30XX && !VPUIP::doesSWLayerFitIntoCMX(op, log)) {
                    log.trace("Memory size of SW Op at {0} is larger than CMX cannot convert to DepthToSpaceDMA.",
                              op->getLoc());
                    return false;
                }

                // Testing on E#37479 showed that if the DepthToSpace tensors fit in CMX and the number planes <= 256
                // Then it is always more efficient than performing it on UPA
                const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
                auto inputShape = inputType.getShape();
                if (op->hasAttr("mode")) {
                    auto mode = op->getAttr("mode").cast<IE::DepthToSpaceModeAttr>().getValue();
                    if (mode == IE::DepthToSpaceMode::BLOCKS_FIRST && inputShape[Dims4D::Act::H] > 256) {
                        log.trace("DepthToSpaceOp at {0} is not efficient convert to DepthToSpaceDMA.", op->getLoc());
                        return false;
                    }
                }

                log.trace("DepthToSpaceOp at {0} can convert to DepthToSpaceDMA.", op->getLoc());
                return true;
            })
            .Case<IE::SpaceToDepthOp, VPUIP::SpaceToDepthUPAOp>([&](mlir::Operation* op) {
                log.trace("Got SpaceToDepthOp at {0}.", op->getLoc());

                // TODO: This can be removed after PR2405 merged.
                if (!VPUIP::doesSWLayerFitIntoCMX(op, log)) {
                    log.trace("Memory size of SW Op at {0} is larger than CMX cannot convert to SpaceToDepthDMA.",
                              op->getLoc());
                    return false;
                }

                if (op->hasAttr("mode") && op->hasAttr("block_size")) {
                    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
                    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                    const auto mode = op->getAttr("mode").cast<IE::SpaceToDepthModeAttr>().getValue();
                    const auto blockSize = op->getAttr("block_size").cast<mlir::IntegerAttr>().getInt();
                    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(op->getContext(), log);
                    auto dmaDescriptor = dmaDescriptorGenerator.generate(inputType, outputType, mode, blockSize);
                    auto numPlanes = dmaDescriptor.numPlanes().getInt();
                    return numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES;
                }

                log.trace("SpaceToDepthOp at {0} can convert to SpaceToDepthDMA.", op->getLoc());
                return true;
            })
            .Case<VPUIP::PerAxisTileUPAOp>([&](mlir::Operation* op) {
                log.trace("Got PerAxisTileOp at {0}.", op->getLoc());

                if (!VPUIP::doesSWLayerFitIntoCMX(op, log)) {
                    log.trace("Memory size of SW Op at {0} is larger than CMX cannot convert to PerAxisTileDMA.",
                              op->getLoc());
                    return false;
                }

                log.trace("PerAxisTileOp at {0} can convert to PerAxisTileDMA.", op->getLoc());
                return true;
            })
            .Case<VPUIP::SwKernelOp>([&](VPUIP::SwKernelOp swKernelOp) {
                if (auto memPerm = getMemPermFromSwKernel(swKernelOp)) {
                    // At VPUX37XX: VPU::MemPermuteUPA -> VPUIP::SwKernelOp -> VPUIP::PermuteDMA
                    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2,
                                      "Unexpected operand number for VPUIP.SwKernelOp at '{0}'", swKernelOp);
                    const auto inputType = swKernelOp.getOperand(0).getType().cast<vpux::NDTypeInterface>();
                    const auto outputType = swKernelOp.getOperand(1).getType().cast<vpux::NDTypeInterface>();

                    if (!VPUIP::getPermuteDMASubInputShapes(inputType, outputType, memPerm.getValue(), log)
                                 .hasValue()) {
                        log.trace("SwKernelOp at {0} doesn't support DMA implementation.", op->getLoc());
                        return false;
                    }
                    log.trace("SwKernelOp at {0} can convert to PermuteDMAOp.", op->getLoc());
                    return true;
                } else if (getDepthToSpaceSwKernelAttr(swKernelOp).hasValue()) {
                    // At VPUX37XX: VPU::DepthToSpaceUPA -> VPUIP::SwKernelOp -> VPUIP::DepthToSpaceDMA
                    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2,
                                      "Unexpected operand number for VPUIP.SwKernelOp at '{0}'", swKernelOp);
                    const auto inputType = swKernelOp.getOperand(0).getType().cast<vpux::NDTypeInterface>();
                    auto inputShape = inputType.getShape();

                    if (inputShape[Dims4D::Act::H] > 256) {
                        // TODO: split more DMAs when the numPlanes is larger than 256 [Track number: E#57027]
                        log.trace("DepthToSpaceOp at {0} cannot convert to DepthToSpaceDMA.", op->getLoc());
                        return false;
                    }

                    log.trace("SwKernelOp at {0} can convert to DepthToSpaceDMA.", op->getLoc());
                    return true;
                } else if (getSpaceToDepthSwKernelAttr(swKernelOp).hasValue()) {
                    // At VPUX37XX: VPU::DepthToSpaceUPA -> VPUIP::SwKernelOp -> VPUIP::DepthToSpaceDMA
                    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2,
                                      "Unexpected operand number for VPUIP.SwKernelOp at '{0}'", swKernelOp);

                    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
                    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

                    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
                    VPUX_THROW_UNLESS(spaceToDepthAttrs.hasValue(),
                                      "Cannot extract spaceToDepth attribute from spaceToDepth SwKernel '{0}'.",
                                      swKernelOp.getLoc());
                    auto mode = spaceToDepthAttrs.getValue().first.getValue();
                    auto blockSize = spaceToDepthAttrs.getValue().second.getInt();

                    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(op->getContext(), log);
                    auto dmaDescriptor = dmaDescriptorGenerator.generate(inputType, outputType, mode, blockSize);
                    auto numPlanes = dmaDescriptor.numPlanes().getInt();

                    if (numPlanes > VPUIP::DMA_MAX_NUMBER_PLANES) {
                        log.trace("{0} at {1} cannot convert to DMA due to numPlanes exceeds limit.", op->getName(),
                                  op->getLoc());
                        return false;
                    }

                    log.trace("SwKernelOp at {0} can convert to DMA.", op->getLoc());
                    return true;
                } else if (isTileSwKernel(swKernelOp)) {
                    // At VPUX37XX: VPU::TileUPA -> VPUIP::SwKernelOp -> VPUIP::PerAxisTileDMA
                    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
                    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

                    if (inputType.getRank() != outputType.getRank()) {
                        log.trace("{0} at {1} cannot convert to DMA due to different in/out shape rank.", op->getName(),
                                  op->getLoc());
                        return false;
                    }

                    log.trace("SwKernelOp at {0} can convert to PerAxisTileDMA.", op->getLoc());
                    return true;
                }

                log.trace("SwKernelOp at {0} cannot convert to DMA.", op->getLoc());
                return false;
            })
            .Case<VPUIP::UpsamplingUPAOp>([&](VPUIP::UpsamplingUPAOp op) {
                const auto inType = op.input().getType().cast<NDTypeInterface>();
                if (inType.getDimsOrder() != DimsOrder::NCHW && inType.getDimsOrder() != DimsOrder::NHWC) {
                    return false;
                }

                const auto inputShape = getShape(op.input());
                const auto padLVector = parseIntArrayAttr<int64_t>(op.pad_l());
                const auto padRVector = parseIntArrayAttr<int64_t>(op.pad_r());
                const auto upsamplingFactorVector = parseIntArrayAttr<int64_t>(op.upsampling_factor());

                // UpsamplingDMA only supports 4D Input shape
                // UpsamplingDMA supports pads only for 3 axes
                // UpsamplingDMA supports factors only for 3 axes
                return (inputShape.size() != 4 || padLVector.size() != 3 || padRVector.size() != 3 ||
                        upsamplingFactorVector.size() != 3);
            })
            .Default([&](mlir::Operation* op) -> bool {
                log.trace("Op {0} at {1} cannot convert to DMA.", op->getName(), op->getLoc());
                return false;
            });
}

bool vpux::VPUIP::isLegalAndBeneficialConvertToDMA(mlir::Operation* op, vpux::Logger log) {
    if (!isLegalConvertToDMA(op, log)) {
        return false;
    }
    const auto arch = VPU::getArch(op);
    if (auto permuteUPAOp = mlir::dyn_cast<VPUIP::PermuteUPAOp>(op)) {
        const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto memPerm = permuteUPAOp.order_value();
        return isBeneficialForUsingPermuteDMA(inputType, outputType, memPerm, arch, log);
    }
    if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        if (VPUIP::isDepthToSpaceSwKernel(swKernelOp) || VPUIP::isSpaceToDepthSwKernel(swKernelOp) ||
            VPUIP::isTileSwKernel(swKernelOp)) {
            return true;
        } else if (VPUIP::isMemPermSwKernel(swKernelOp)) {
            auto memPerm = getMemPermFromSwKernel(swKernelOp);
            VPUX_THROW_UNLESS(memPerm.hasValue(), "Cannot extract mem_perm attribute from permute SwKernel '{0}'.",
                              swKernelOp.getLoc());

            const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
            return isBeneficialForUsingPermuteDMA(inputType, outputType, memPerm.getValue(), arch, log);
        }

        return false;
    }
    return true;
}

VPUIP::DmaDescriptorAttr vpux::VPUIP::updateNumPlanes(VPUIP::DmaDescriptorAttr dmaDescriptor, int64_t numPlane) {
    auto ctx = dmaDescriptor.getContext();
    auto numPlaneAttr = vpux::getIntAttr(ctx, numPlane);
    return VPUIP::DmaDescriptorAttr::get(numPlaneAttr, dmaDescriptor.len(), dmaDescriptor.srcWidth(),
                                         dmaDescriptor.srcStride(), dmaDescriptor.srcPlaneStride(),
                                         dmaDescriptor.dstWidth(), dmaDescriptor.dstStride(),
                                         dmaDescriptor.dstPlaneStride(), ctx);
}

bool vpux::VPUIP::isMemPermSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelTask.kernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "reorder_fp16";
}

Optional<mlir::AffineMap> vpux::VPUIP::getMemPermFromSwKernel(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isMemPermSwKernel(swKernelTask)) {
        return None;
    }

    VPUX_THROW_WHEN(swKernelTask.body().getOps<VPUIP::SwKernelRun>().empty(), "Cannot get VPUIP.SwKernelRun at '{0}'",
                    swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.body().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.attrs().hasValue(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get reversed permute value
    const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
    VPUX_THROW_WHEN(arrayAttrs.empty(), "Empty attribute at '{0}'", kernelRun->getLoc());
    auto reversedPermute = parseIntArrayAttr<unsigned>(arrayAttrs.getValue()[0].dyn_cast<mlir::ArrayAttr>());
    auto permute = correctPermutation(reversedPermute);
    return mlir::AffineMap::getPermutationMap(permute, swKernelTask->getContext());
}

bool vpux::VPUIP::isDepthToSpaceSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelTask.kernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "single_shave_depth_to_space";
}

bool vpux::VPUIP::isSpaceToDepthSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelTask.kernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "single_shave_space_to_depth";
}

bool vpux::VPUIP::isTileSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FuncOp>(swKernelTask.kernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "single_shave_tile";
}

bool vpux::VPUIP::isPerAxisTileSwKernel(VPUIP::SwKernelOp swKernelTask) {
    if (!isTileSwKernel(swKernelTask)) {
        return false;
    }

    // Only support TileOp with one Axis expansion
    const auto inType = swKernelTask->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outType = swKernelTask->getOperand(2).getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(inType.getRank() == outType.getRank(),
                      "Tile Op with different inShape '{0}' outShape '{1}' rank.", inType.getRank(), outType.getRank());

    const auto ioShapes = zip(inType.getShape(), outType.getShape());
    const auto dimDiffPredicate = [](const std::tuple<int64_t, int64_t>& ioDims) -> bool {
        const auto& inDim = std::get<0>(ioDims);
        const auto& outDim = std::get<1>(ioDims);
        return inDim != outDim;
    };
    const auto diffAxisCount = llvm::count_if(ioShapes, dimDiffPredicate);

    return diffAxisCount == 1;
}

Optional<VPUIP::DepthToSpaceReturnType> vpux::VPUIP::getDepthToSpaceSwKernelAttr(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isDepthToSpaceSwKernel(swKernelTask)) {
        return None;
    }

    VPUX_THROW_WHEN(swKernelTask.body().getOps<VPUIP::SwKernelRun>().empty(), "Cannot get VPUIP.SwKernelRun at '{0}'",
                    swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.body().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.attrs().hasValue(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get DepthToSpace attrs
    const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
    VPUX_THROW_WHEN(arrayAttrs.empty(), "Empty attribute at '{0}'", kernelRun->getLoc());

    auto blockSizeAttr = arrayAttrs.getValue()[0].dyn_cast<mlir::IntegerAttr>();
    auto mode = IE::DepthToSpaceMode(arrayAttrs.getValue()[1].dyn_cast<mlir::IntegerAttr>().getInt());
    auto modeAttr = IE::DepthToSpaceModeAttr::get(swKernelTask.getContext(), mode);
    auto paddedInChannels =
            arrayAttrs.getValue().size() > 2 ? arrayAttrs.getValue()[2].dyn_cast<mlir::IntegerAttr>() : nullptr;
    auto paddedOutChannels =
            arrayAttrs.getValue().size() > 3 ? arrayAttrs.getValue()[3].dyn_cast<mlir::IntegerAttr>() : nullptr;
    VPUX_THROW_WHEN(blockSizeAttr == nullptr || modeAttr == nullptr, "Empty DepthToSpace attribute at '{0}'",
                    kernelRun->getLoc());

    auto paddedChannels =
            (paddedInChannels != nullptr && paddedOutChannels != nullptr)
                    ? IE::ChannelPadding::get(paddedInChannels, paddedOutChannels, swKernelTask.getContext())
                    : nullptr;

    return VPUIP::DepthToSpaceReturnType(modeAttr, blockSizeAttr, paddedChannels);
}

Optional<std::pair<IE::SpaceToDepthModeAttr, mlir::IntegerAttr>> vpux::VPUIP::getSpaceToDepthSwKernelAttr(
        VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelTask)) {
        return None;
    }

    VPUX_THROW_WHEN(swKernelTask.body().getOps<VPUIP::SwKernelRun>().empty(), "Cannot get VPUIP.SwKernelRun at '{0}'",
                    swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.body().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.attrs().hasValue(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get SpaceToDepth attrs
    const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
    VPUX_THROW_UNLESS(arrayAttrs.size() == 2, "Wrong numbers of attribute at '{0}', expected 2 but got '{1}'",
                      kernelRun->getLoc(), arrayAttrs.size());

    auto blockSizeAttr = arrayAttrs.getValue()[0].dyn_cast<mlir::IntegerAttr>();
    auto modeIntAttr = arrayAttrs.getValue()[1].dyn_cast<mlir::IntegerAttr>();
    VPUX_THROW_UNLESS(blockSizeAttr != nullptr && modeIntAttr != nullptr,
                      "Failed to extract block size and mode at '{0}'", kernelRun->getLoc());

    auto modeAttr =
            IE::SpaceToDepthModeAttr::get(swKernelTask.getContext(), IE::SpaceToDepthMode(modeIntAttr.getInt()));
    VPUX_THROW_WHEN(blockSizeAttr == nullptr || modeAttr == nullptr, "Empty SpaceToDepth attribute at '{0}'",
                    kernelRun->getLoc());

    return std::pair<IE::SpaceToDepthModeAttr, mlir::IntegerAttr>(modeAttr, blockSizeAttr);
}

Optional<VPUIP::PerAxisTileAttr> vpux::VPUIP::getPerAxisTileSwKernelAttr(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isPerAxisTileSwKernel(swKernelTask)) {
        return None;
    }

    // get PerAxisTile attrs
    auto copyOp = swKernelTask->getOperand(1).getDefiningOp<VPUIP::CopyOp>();
    VPUX_THROW_UNLESS(copyOp != nullptr, "Cannot got Copy Op at '{0}'", swKernelTask->getLoc());

    auto repeatsConst = copyOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(repeatsConst != nullptr, "Got non constant repeats parameters at '{0}'", swKernelTask->getLoc());
    const auto repeatsContent = repeatsConst.content();
    const auto repeats = repeatsContent.getValues<int64_t>();

    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };

    const auto axisCount = llvm::count_if(repeats, greaterThanOne);
    VPUX_THROW_UNLESS(axisCount == 1, "PerAxisTile Op should with one Axis expansion, but got '{0}'", axisCount);

    auto axisIter = std::find_if(repeats.begin(), repeats.end(), greaterThanOne);
    VPUX_THROW_UNLESS(axisIter != repeats.end(), "Cannot find axis to expansion");
    auto axis = std::distance(repeats.begin(), axisIter);

    const auto ctx = swKernelTask->getContext();
    return VPUIP::PerAxisTileAttr{mlir::IntegerAttr::get(getInt64Type(ctx), axis),
                                  mlir::IntegerAttr::get(getInt64Type(ctx), repeats[axis])};
}

// No matter what shape size and layout PerAxisTile Op is, It will convert to 3D with MemShape.
// And the expansion Axis always exist in the second dimension.
// It can simplify calculating Descriptor. There are some cases:
// [1, 2, 3, 4] #NHWC -> [1, 6, 3, 4] #NHWC,  Axis = 1, Tiles = 3;
// Merged inShape: [[3x4], 2, 1]
// Merged outShape: [[3x4], 6, 1]
//
// [1, 2, 3, 4] #NCHW -> [6, 2, 3, 4] #NCHW,  Axis = 0, Tiles = 6;
// Merged inShape: [1, 1, [2x3x4]]
// Merged outShape: [1, 6, [2x3x4]]
std::pair<vpux::Shape, vpux::Shape> vpux::VPUIP::getPerAxisTileDMAMergedShape(vpux::NDTypeInterface inType,
                                                                              vpux::NDTypeInterface outType,
                                                                              int64_t axis, int64_t tiles) {
    auto inShape = inType.getShape();
    auto outShape = outType.getShape();
    VPUX_THROW_UNLESS(inShape.size() == outShape.size() && axis < checked_cast<int64_t>(inShape.size()) &&
                              inShape[Dim(axis)] * tiles == outShape[Dim(axis)],
                      "Unexpect PerAxisTile input shape '{0}' and output shape '{1}'", inShape, outShape);

    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = outOrder.toMemoryOrder(outShape);

    const auto getMergedShape = [](MemShape shape, int64_t axis) -> Shape {
        SmallVector<int64_t> mergedShape(3, 1);
        for (auto idx = 0; idx < checked_cast<int64_t>(shape.size()); idx++) {
            const auto mergeAxis = (idx < axis) ? 0 : (idx == axis) ? 1 : 2;
            mergedShape[mergeAxis] *= shape[MemDim(idx)];
        }
        return Shape(mergedShape);
    };

    return std::pair<Shape, Shape>(getMergedShape(inMemShape, inOrder.dimPos(Dim(axis))),
                                   getMergedShape(outMemShape, inOrder.dimPos(Dim(axis))));
}

SmallVector<vpux::Shape> vpux::VPUIP::getPerAxisTileDMASubShapes(vpux::ShapeRef shape) {
    const auto shapeSize = shape.size();
    VPUX_THROW_UNLESS(shapeSize == 3, "PerAxisTile merged Shape size should be 3, but got {0}", shapeSize);

    const auto totalNumPlane = shape[Dim(0)];
    auto numberDMAs = divUp(totalNumPlane, VPUIP::DMA_MAX_NUMBER_PLANES);
    auto subShape = Shape(shape.raw());
    if (numberDMAs > 1) {
        subShape[Dim(0)] = VPUIP::DMA_MAX_NUMBER_PLANES;
        SmallVector<Shape> subOutputShapes(numberDMAs - 1, subShape);
        subShape[Dim(0)] = totalNumPlane - VPUIP::DMA_MAX_NUMBER_PLANES * (numberDMAs - 1);
        subOutputShapes.push_back(subShape);
        return subOutputShapes;
    }
    return SmallVector<Shape>{subShape};
}

VPURT::DeclareBufferOp vpux::VPUIP::createNewDeclareBuffer(mlir::PatternRewriter& rewriter,
                                                           mlir::Operation* insertionPoint,
                                                           VPURT::DeclareBufferOp declBuff,
                                                           vpux::NDTypeInterface newType, int64_t offset) {
    auto ctx = declBuff->getContext();
    auto section = declBuff.section();
    int64_t sectionIndex =
            declBuff.sectionIndex().hasValue() ? parseIntArrayAttr<int64_t>(declBuff.sectionIndex().getValue())[0] : -1;
    const auto symbolAttr =
            sectionIndex == -1
                    ? vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)))
                    : vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)), sectionIndex);
    newType = newType.changeMemSpace(symbolAttr);
    return sectionIndex == -1 ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, declBuff->getLoc(),
                                                                        newType, section, offset)
                              : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, declBuff->getLoc(),
                                                                        newType, section, sectionIndex, offset);
}
