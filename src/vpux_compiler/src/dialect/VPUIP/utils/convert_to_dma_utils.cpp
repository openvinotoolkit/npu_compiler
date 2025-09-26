//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

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

// Mapping DEPTH_FIRST D2S with block size = 4 / block size = 2 to SHAVE is the most optimum for the following reasons:
// 1) DMA: DMA engine bandwidth is 1 element/cycle. Here we're trying to handle depth translations, which generates
// numerous DMA workloads.
// 2) DPU: It is not feasible for the hardware to handle block sizes > 2, see E#83455.

// Optimized DepthToSpace SW kernel has below restrictions:
// 1. Layout must be NHWC
// 2. Data type must be 16-bits
// 3. case blockSize = 4: Only support 128 / 16 input channels; case blockSize = 2: support 16 aligned input channals
// 4. DepthToSpace mode should be DEPTH_FIRST
bool isBeneficialForUsingSWDepthToSpace(vpux::NDTypeInterface inType, vpux::IE::DepthToSpaceMode mode,
                                        int64_t blockSize) {
    const auto inC = inType.getShape()[Dims4D::Act::C];
    const auto inW = inType.getShape()[Dims4D::Act::W];

    const bool isNHWC = (inType.getDimsOrder() == DimsOrder::NHWC);
    const bool is16bit = (inType.getElementType().isF16() || inType.getElementType().isInteger(16));
    const bool isC16C128 = (inC == 16) || (inC == 128);
    const bool isBS4 = (blockSize == 4);
    const bool isC16Align = (inC % 16 == 0);
    const bool isBS2 = (blockSize == 2);
    const bool isDepthFirst = (mode == IE::DepthToSpaceMode::DEPTH_FIRST);

    if ((inType.getElemTypeSize().count() == 8) && (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) && (inC == 16) &&
        (blockSize == 2) && (inW >= 256) && isNHWC) {
        return true;
    }

    return isNHWC && is16bit && ((isBS4 && isC16C128) || (isBS2 && isC16Align)) && isDepthFirst;
}

bool isBeneficialForUsingSWDepthToSpace(VPUIP::SwKernelOp swKernelOp, config::ArchKind /*arch*/) {
    VPUX_THROW_UNLESS(VPUIP::isDepthToSpaceSwKernel(swKernelOp), "SwKernelOp {0} is not DepthToSpace",
                      swKernelOp->getLoc());
    const auto inType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType());
    const auto d2sAttr = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    const auto mode = std::get<0>(d2sAttr.value()).getValue();
    const auto blockSize = std::get<1>(d2sAttr.value()).getInt();

    return isBeneficialForUsingSWDepthToSpace(inType, mode, blockSize);
}

SmallVector<Shape> computeDMASubShape(config::ArchKind arch, ShapeRef shape, Dim numPlaneDim, int64_t dmaPortCount) {
    VPUX_THROW_WHEN(dmaPortCount <= 0, "Invalid number of DMA ports: {0}", dmaPortCount);

    const auto shapeSize = shape.size();
    VPUX_THROW_UNLESS(shapeSize == 2 || shapeSize == 3 || shapeSize == 4,
                      "Shape size should be 2 or 3 or 4, but got {0}", shapeSize);
    VPUX_THROW_UNLESS(static_cast<size_t>(numPlaneDim.ind()) < shapeSize,
                      "numPlaneDim index {0} doesn't match shape size {1}", numPlaneDim.ind(), shapeSize);

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(arch);
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

    const auto totalPlaneCount = shape[numPlaneDim];
    // Enforce hardware limitation: Each DMA plane number must be less than 256
    //  - 'requiredDMACount' represents the minimum number of DMAs needed to handle the data
    // Given multiple DMA ports can operate in parallel to move data
    //  - 'optimizedDMACount' adjusts the DMA count to fully utilize available DMA sources
    // Example scenario:
    // Input: 520x16, Output: 16x520, Plane number is 520, DMA Port number is 2
    // With 2 available DMA ports, the initial 'requiredDMACount' is 3, allowing only the first two PermuteDMAs to run
    // concurrently. 'optimizedDMACount' adjusts this to 4 to ensure all DMAs are utilized efficiently
    auto requiredDMACount = divUp(totalPlaneCount, dmaMaxNumPlanes);
    auto optimizedDMACount = divUp(requiredDMACount, dmaPortCount) * dmaPortCount;

    // Aim to distribute the data size as evenly as possible across PermuteDMAs
    // Example scenario:
    // - Input: 521x16, Output: 16x521, Plane number is 521, DMA Port number is 2
    //   The data is evenly divided into four PermuteDMAs
    // - The resulting input shapes for the PermuteDMAs are:
    //   Three segments of 130x16 and one segment of 131x16
    auto baseSize = totalPlaneCount / optimizedDMACount;
    auto extraSizeCount = totalPlaneCount % optimizedDMACount;
    SmallVector<Shape> subOutputShapes;
    auto subShape = Shape(shape.raw());
    if (baseSize != 0) {
        subShape[numPlaneDim] = baseSize;
        subOutputShapes.insert(subOutputShapes.end(), optimizedDMACount - extraSizeCount, subShape);
    }

    if (extraSizeCount != 0) {
        subShape[numPlaneDim] = baseSize + 1;
        subOutputShapes.insert(subOutputShapes.end(), extraSizeCount, subShape);
    }

    return subOutputShapes;
}
}  // namespace

bool vpux::VPUIP::satisfiesOptimizedMemPermute(config::ArchKind arch, NDTypeInterface inType, NDTypeInterface outType) {
    // MemPermute kernel is specially optimized for the conditions below
    if (arch == config::ArchKind::NPU37XX) {
        return false;
    }
    const auto inBits = inType.getElemTypeSize().count();
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();

    if ((inOrder == DimsOrder::NHWC) && (outOrder == DimsOrder::NCHW)) {
        const auto inC = inType.getShape()[Dims4D::Act::C];
        const auto inW = inType.getShape()[Dims4D::Act::W];
        if ((inC == 4) && (inW >= 256) && (inBits == 8)) {
            return true;
        }
    }
    return false;
}

bool isDMASupportedMemPermuteDistribution(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);
    // If both input/output is not distributed type we can convert to DMA.
    if (inDistributedType == nullptr && outDistributedType == nullptr) {
        return true;
    }

    if (inDistributedType != nullptr) {
        const auto inMode = inDistributedType.getDistribution().getMode().getValue();
        if (outDistributedType != nullptr) {
            const auto outMode = outDistributedType.getDistribution().getMode().getValue();
            auto areBothDuplicated =
                    inMode == VPU::DistributionMode::DUPLICATED && outMode == VPU::DistributionMode::DUPLICATED;
            return areBothDuplicated;
        }
        auto duplicatedInput = inMode == VPU::DistributionMode::DUPLICATED;
        return duplicatedInput;
    }

    // Input is not distributed, Output is distributed
    const auto outMode = outDistributedType.getDistribution().getMode().getValue();
    const auto supportedOutputMode =
            outMode == VPU::DistributionMode::SEGMENTED || outMode == VPU::DistributionMode::OVERLAPPED ||
            outMode == VPU::DistributionMode::DUPLICATED || outMode == VPU::DistributionMode::MULTICASTED;
    return supportedOutputMode;
}

bool vpux::VPUIP::isBeneficialForUsingPermuteDMA(config::ArchKind arch, NDTypeInterface inType, NDTypeInterface outType,
                                                 mlir::AffineMap memPerm, int64_t dmaPortCount, vpux::Logger log) {
    // Check if the memory permutation satisfies optimized conditions for Shave optimizations
    // If it does, using PermuteDMA is not beneficial
    if (satisfiesOptimizedMemPermute(arch, inType, outType)) {
        return false;
    }

    if (!isDMASupportedMemPermuteDistribution(inType, outType)) {
        return false;
    }

    auto subShapes = VPUIP::getPermuteDMASubInputShapes(arch, inType, outType, memPerm, dmaPortCount, log);
    return subShapes.has_value();
}

bool vpux::VPUIP::isBeneficialForUsingSWDepthToSpace(VPU::DepthToSpaceOp d2sOp) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType());
    const auto mode = d2sOp.getMode();
    const auto blockSize = d2sOp.getBlockSize();

    return ::isBeneficialForUsingSWDepthToSpace(inType, mode, blockSize);
}

// In order to simplify the difference cases about input layout and mem perm, the merged input shape need to be
// calculated. For example,
// [1, 4, 16, 16] #NCHW -> [1, 4, 16, 16] #NHWC, memPerm=[d0, d2, d3, d1] can be merged into
// [4, 256] #NC -> [256, 4] #NC, memPerm=[d1, d0]
std::optional<Shape> vpux::VPUIP::getPermuteDMAInputShape(NDTypeInterface inType, NDTypeInterface outType,
                                                          mlir::AffineMap perm, vpux::Logger log) {
    if (!perm.isPermutation()) {
        log.trace("Permute op with input {0}, output {1} doesn't support DMA with memPerm {2}", inType, outType, perm);
        return std::nullopt;
    }

    auto inputMemShape = inType.getMemShape().raw();
    auto memPerm = DimsOrder::fromAffineMap(perm);
    SmallVector<int64_t> memPermIdx;
    for (size_t idx = 0; idx < inputMemShape.size(); idx++) {
        if (inputMemShape[memPerm.dimAt(idx).ind()] != 1) {
            memPermIdx.push_back(memPerm.dimAt(idx).ind());
        }
    }

    Shape newInputShape;
    int64_t shapeSize = 1;
    // Consolidates adjacent dimensions into a single
    // Example: inputMemShape [2, 3, 4, 5], memPermIdx [d1, d2, d3, d0], outputMemShape [3, 4, 5, 2]
    // Dimensions [d1, d2, d3] form a continuous data block, allowing reshaping to [24, 5]
    for (size_t idx = 0; idx < memPermIdx.size(); idx++) {
        shapeSize *= inputMemShape[memPermIdx[idx]];

        bool isLastElementOrDiscontinuous =
                (idx + 1 == memPermIdx.size()) || (memPermIdx[idx] + 1 != memPermIdx[idx + 1]);
        if (isLastElementOrDiscontinuous) {
            newInputShape.push_back(shapeSize);
            shapeSize = 1;
        }
    }

    auto inShape = inType.getShape();
    auto outShape = outType.getShape();

    if (newInputShape.size() == 1) {
        auto permDim = Dim(memPermIdx.front());
        return checked_cast<size_t>(permDim.ind()) < memPerm.dimPos(permDim) ? Shape{newInputShape.front(), 1}
                                                                             : Shape{1, newInputShape.front()};
    } else if (newInputShape.size() == 2) {
        return Shape{newInputShape.back(), newInputShape.front()};
    } else if (newInputShape.size() == 3) {
        auto mergedMemPerm = getPermuteDMAMergedMemPerm(inType, perm);
        auto ctx = inType.getContext();
        // The data order of newInputshape has appiled permutation. So we need reverse it according the permute map.

        if (mergedMemPerm == DimsOrder::HCW.toAffineMap(ctx)) {
            // Check for permute pattern: HWC->WHC
            return Shape{newInputShape[Dim(1)], newInputShape[Dim(0)], newInputShape[Dim(2)]};
        } else if (mergedMemPerm == DimsOrder::CWH.toAffineMap(ctx)) {
            // Check for permute pattern: HWC->HCW
            return Shape{newInputShape[Dim(0)], newInputShape[Dim(2)], newInputShape[Dim(1)]};
        } else if (mergedMemPerm == DimsOrder::WHC.toAffineMap(ctx)) {
            // Check for permute pattern: HWC->CWH
            // Special case if one of the merged input dim is 1, it can not be split
            // Special case if d0 is not merged in mergedPerm, it can not be split
            // Special case NCHW-> WCHN is supported now
            // When more cases are supported, the restriction would be removed.
            auto anyDimIsOne = std::any_of(inShape.begin() + 1, inShape.end(), [](auto dim) {
                return dim == 1;
            });
            auto notSupportedPermute = [&](mlir::AffineMap perm) {
                return (perm == DimsOrder::WHNC.toAffineMap(ctx) || perm == DimsOrder::HWCN.toAffineMap(ctx));
            };
            if ((anyDimIsOne && perm != DimsOrder::WCHN.toAffineMap(ctx)) || notSupportedPermute(perm)) {
                return std::nullopt;
            }

            return Shape{newInputShape[Dim(2)], newInputShape[Dim(1)], newInputShape[Dim(0)]};
        } else {
            return std::nullopt;
        }
    } else if (newInputShape.size() == 4) {
        auto mergedMemPerm = getPermuteDMAMergedMemPerm(inType, perm);
        if (mergedMemPerm == DimsOrder::NHCW.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
            return Shape{newInputShape[Dim(0)], newInputShape[Dim(2)], newInputShape[Dim(1)], newInputShape[Dim(3)]};
        } else if (mergedMemPerm == DimsOrder::HCNW.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d2, d1, d0, d3]
            return Shape{newInputShape[Dim(2)], newInputShape[Dim(1)], newInputShape[Dim(0)], newInputShape[Dim(3)]};
        } else if (mergedMemPerm == DimsOrder::NWHC.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d0, d3, d2, d1]
            return Shape{newInputShape[Dim(0)], newInputShape[Dim(3)], newInputShape[Dim(2)], newInputShape[Dim(1)]};
        } else if (mergedMemPerm == DimsOrder::CWNH.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d1, d3, d0, d2]
            return Shape{newInputShape[Dim(1)], newInputShape[Dim(3)], newInputShape[Dim(0)], newInputShape[Dim(2)]};
        } else if (mergedMemPerm == DimsOrder::HNWC.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d2, d0, d3, d1]
            return Shape{newInputShape[Dim(2)], newInputShape[Dim(0)], newInputShape[Dim(3)], newInputShape[Dim(1)]};
        } else if (mergedMemPerm == DimsOrder::HCWN.toAffineMap(inType.getContext())) {
            // Check for permute pattern: [d0, d1, d2, d3] -> [d2, d1, d3, d0]
            return Shape{newInputShape[Dim(2)], newInputShape[Dim(1)], newInputShape[Dim(3)], newInputShape[Dim(0)]};
        } else {
            return std::nullopt;
        }
    }

    log.trace("Can't convert Permute to DMA with inshape {0}, outshape {1}, memPerm {2}.", inShape, outShape, memPerm);
    return std::nullopt;
}

std::optional<Shape> vpux::VPUIP::getPermuteDMAOutputShape(NDTypeInterface inType, NDTypeInterface outType,
                                                           mlir::AffineMap perm, vpux::Logger log) {
    auto mergedInputShape = getPermuteDMAInputShape(inType, outType, perm, log);
    if (!mergedInputShape.has_value()) {
        return std::nullopt;
    }
    auto mergedOutputShape = getPermuteDMASubOutputShapes({mergedInputShape.value()}, inType, outType, perm).front();
    return mergedOutputShape;
}

std::optional<SmallVector<Shape>> vpux::VPUIP::getPermuteDMASubInputShapes(config::ArchKind arch,
                                                                           NDTypeInterface inType,
                                                                           NDTypeInterface outType,
                                                                           mlir::AffineMap perm, int64_t dmaPortCount,
                                                                           vpux::Logger log) {
    if (!perm.isPermutation()) {
        log.trace("PermuteOp doesn't support permutation.");
        return std::nullopt;
    }

    auto newInputShape = getPermuteDMAInputShape(inType, outType, perm, log);
    if (!newInputShape.has_value()) {
        return std::nullopt;
    }

    auto numPlaneDim = getPermuteDMANumPlaneDim(inType, perm);
    return computeDMASubShape(arch, newInputShape.value(), numPlaneDim, dmaPortCount);
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
            if (mergedMemPerm == DimsOrder::HCW.toAffineMap(ctx)) {
                subOutputShapes.push_back(Shape(SmallVector<int64_t>{inShape[1], inShape[0], inShape[2]}));
            } else if (mergedMemPerm == DimsOrder::CWH.toAffineMap(ctx)) {
                subOutputShapes.push_back(Shape(SmallVector<int64_t>{inShape[0], inShape[2], inShape[1]}));
            } else {
                VPUX_THROW("unsupported inShape {0} with memPerm {1}, mergedPerm {2}", inShape, memPerm, mergedMemPerm);
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
    for (unsigned int idx = 0; idx < memPerm.getNumResults(); idx++) {
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

    if (mergedPerm == DimsOrder::CWH.toAffineMap(ctx)) {
        return Dim(1);
    }
    return Dim(0);
}

bool vpux::VPUIP::isSplitNeededForPermuteDMA(vpux::NDTypeInterface inType, mlir::AffineMap memPerm) {
    auto ctx = inType.getContext();
    auto mergedPerm = getPermuteDMAMergedMemPerm(inType, memPerm);

    return mergedPerm == DimsOrder::WHC.toAffineMap(ctx) || mergedPerm == DimsOrder::NHCW.toAffineMap(ctx) ||
           mergedPerm == DimsOrder::HCNW.toAffineMap(ctx) || mergedPerm == DimsOrder::NWHC.toAffineMap(ctx) ||
           mergedPerm == DimsOrder::CWNH.toAffineMap(ctx) || mergedPerm == DimsOrder::HNWC.toAffineMap(ctx) ||
           mergedPerm == DimsOrder::HCWN.toAffineMap(ctx);
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
std::optional<Dim> vpux::VPUIP::getTileDimForPermuteDMA(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                        mlir::AffineMap memPerm,
                                                        VPUIP::DistributedBufferType distributedOutputType,
                                                        vpux::Logger log) {
    const auto distributionAttr = distributedOutputType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unexpected distributed mode {0}", VPU::stringifyEnum(mode));
    const auto outputShape = outType.getShape();
    const auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, log).value();
    const auto mergedOutputDimList = VPUIP::getPermuteDMAOutputMergedDimList(outType, mergedOutputShape);

    VPUX_THROW_UNLESS(mergedOutputDimList.size() == 2 || mergedOutputDimList.size() == 3,
                      "Unexpected merged dim list {0}", mergedOutputDimList);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());

    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
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
    return std::nullopt;
}

bool vpux::VPUIP::doesPermuteDMATileDimSupportWrapInCluster(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                            mlir::AffineMap memPerm,
                                                            VPUIP::DistributedBufferType distributedOutputType,
                                                            vpux::Logger log) {
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, log).value();
    auto mergedDims = VPUIP::getPermuteDMAOutputMergedDimList(outType, mergedOutputShape);
    VPUX_THROW_UNLESS(mergedDims.size() == 2 || mergedDims.size() == 3, "Invalid dims size, get {0}",
                      mergedDims.size());

    const auto distributionAttr = distributedOutputType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    // Disable duplicate permute since there is performance regression for some models. Need set up a cost function in
    // future to evaluate the dma cost and decide to fuse permute into cluster or not
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return false;
    }

    if (mergedDims.size() == 3) {
        return false;
    }

    auto tileDim = getTileDimForPermuteDMA(inType, outType, memPerm, distributedOutputType, log);
    return tileDim.has_value();
}

bool vpux::VPUIP::doesSWLayerFitIntoCMX(mlir::Operation* op, vpux::Logger log) {
    if (!mlir::isa<IE::DepthToSpaceOp, IE::SpaceToDepthOp, VPUIP::SwKernelOp>(op)) {
        log.trace("unsupported op type at '{0}'", op->getLoc());
        return false;
    }
    if (mlir::isa<VPUIP::SwKernelOp>(op)) {
        // SwKernelOp should be tiled to fit CMX
        return true;
    }
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

    Byte requiredCMX(0);
    requiredCMX += inputType.getTotalAllocSize();
    requiredCMX += outputType.getTotalAllocSize();
    if (requiredCMX > VPU::getTotalCMXSize(op)) {
        log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(op), requiredCMX);
        return false;
    }
    return true;
}

bool vpux::VPUIP::isLegalConvertToDMA(mlir::Operation* op, vpux::Logger log, bool checkCMXSize) {
    if (config::getCompilationMode(op) == config::CompilationMode::ReferenceSW) {
        return false;
    }
    const auto arch = config::getArch(op);
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(arch);
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<VPU::MemPermuteOp>([&](mlir::Operation* op) {
                log.trace("Got Permute Op at {0}.", op->getLoc());

                if (IE::hasDynamicTensors(op)) {
                    // TODO(E#105847): MemPermute with the dynamic shape ops cannot be converted to DMA
                    return false;
                }
                const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
                const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

                mlir::AffineMap memPerm;
                if (auto memPermuteOp = mlir::dyn_cast<VPU::MemPermuteOp>(op)) {
                    memPerm = memPermuteOp.getMemPerm();
                } else {
                    return false;
                }

                auto module = op->getParentOfType<mlir::ModuleOp>();
                const auto dmaPortNum = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

                if (!VPUIP::getPermuteDMASubInputShapes(arch, inputType, outputType, memPerm, dmaPortNum, log)
                             .has_value()) {
                    log.trace("MemPermute Op at {0} doesn't support DMA implementation.", op->getLoc());
                    return false;
                }

                if (checkCMXSize && !VPUIP::doesSWLayerFitIntoCMX(op, log)) {
                    log.trace("Memory size of SW Op at {0} is larger than CMX, can not move to CMX.", op->getLoc());
                    return false;
                }

                log.trace("PermuteOp at {0} can convert to PermuteDMAOp.", op->getLoc());
                return true;
            })
            .Case<IE::DepthToSpaceOp, VPU::DepthToSpaceOp>([&](mlir::Operation* op) {
                log.trace("Got DepthToSpaceOp Op at {0}.", op->getLoc());

                log.trace("DepthToSpaceOp at {0} can convert to DepthToSpaceDMA.", op->getLoc());
                return true;
            })
            .Case<IE::SpaceToDepthOp>([&](mlir::Operation* op) {
                log.trace("Got SpaceToDepthOp at {0}.", op->getLoc());

                if (op->hasAttr("mode") && op->hasAttr("block_size")) {
                    const auto blockSize = mlir::cast<mlir::IntegerAttr>(op->getAttr("block_size")).getInt();
                    return blockSize <= dmaMaxNumPlanes;
                }

                log.trace("SpaceToDepthOp at {0} can convert to SpaceToDepthDMA.", op->getLoc());
                return true;
            })
            .Case<VPU::SpaceToDepthOp>([&](mlir::Operation* op) {
                log.trace("Got SpaceToDepthOp at {0}.", op->getLoc());

                if (op->hasAttr("mode") && op->hasAttr("block_size")) {
                    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
                    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
                    const auto mode = mlir::cast<vpux::IE::SpaceToDepthModeAttr>(op->getAttr("mode")).getValue();
                    const auto blockSize = mlir::cast<mlir::IntegerAttr>(op->getAttr("block_size")).getInt();
                    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(op->getContext(), log);
                    auto dmaDescriptor = dmaDescriptorGenerator.generate(inputType, outputType, mode, blockSize);
                    auto numPlanes = dmaDescriptor.getNumPlanes().getInt();
                    return numPlanes <= dmaMaxNumPlanes;
                }

                log.trace("SpaceToDepthOp at {0} can convert to SpaceToDepthDMA.", op->getLoc());
                return true;
            })
            .Case<VPUIP::SwKernelOp>([&](VPUIP::SwKernelOp swKernelOp) {
                // TODO(E#105847): dynamic shape ops cannot be converted to DMA
                if (VPUIP::hasBoundedBuffers(swKernelOp) || VPUIP::hasUngroupedBoundedBuffers(swKernelOp)) {
                    return false;
                }
                if (auto memPerm = getMemPermFromSwKernel(swKernelOp)) {
                    // At VPUX37XX: VPU::MemPermute -> VPUIP::SwKernelOp -> VPUIP::PermuteDMA
                    if (swKernelOp->getNumOperands() != 2) {
                        log.trace("MemPermute {0} is more efficient on SW", op->getLoc());
                        return false;
                    }
                    const auto inputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getOperand(0).getType());
                    const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getResult(0).getType());
                    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
                    const auto dmaPortNum = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

                    if (!VPUIP::getPermuteDMASubInputShapes(config::getArch(op), inputType, outputType, memPerm.value(),
                                                            dmaPortNum, log)
                                 .has_value()) {
                        log.trace("SwKernelOp at {0} doesn't support DMA implementation.", op->getLoc());
                        return false;
                    }
                    log.trace("SwKernelOp at {0} can convert to PermuteDMAOp.", op->getLoc());
                    return true;
                } else if (getDepthToSpaceSwKernelAttr(swKernelOp).has_value()) {
                    // At VPUX37XX: VPU::DepthToSpace -> VPUIP::SwKernelOp -> VPUIP::DepthToSpaceDMA

                    // In general, DepthToSpace has 2 operands - inputs and outputs
                    // But if a DepthToSpace SW Kernel Op has been tiled into N tiles by tile-act-shave-kernel-task
                    // pass, it will have N swKernelRun had and (N * 2) operands
                    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
                    auto swKernelRunNum = std::distance(swKernelRun.begin(), swKernelRun.end());
                    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == swKernelRunNum * 2,
                                      "Unexpected operand number for VPUIP.SwKernelOp at '{0}'", swKernelOp);
                    log.trace("SwKernelOp at {0} can convert to DepthToSpaceDMA.", op->getLoc());
                    return true;
                } else if (getSpaceToDepthSwKernelAttr(swKernelOp).has_value()) {
                    // At VPUX37XX: VPU::DepthToSpace -> VPUIP::SwKernelOp -> VPUIP::DepthToSpaceDMA
                    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2,
                                      "Unexpected operand number for VPUIP.SwKernelOp at '{0}'", swKernelOp);

                    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
                    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

                    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
                    VPUX_THROW_UNLESS(spaceToDepthAttrs.has_value(),
                                      "Cannot extract spaceToDepth attribute from spaceToDepth SwKernel '{0}'.",
                                      swKernelOp.getLoc());
                    auto mode = spaceToDepthAttrs.value().first.getValue();
                    auto blockSize = spaceToDepthAttrs.value().second.getInt();

                    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(op->getContext(), log);
                    auto dmaDescriptor = dmaDescriptorGenerator.generate(inputType, outputType, mode, blockSize);
                    auto numPlanes = dmaDescriptor.getNumPlanes().getInt();

                    if (numPlanes > dmaMaxNumPlanes) {
                        log.trace("{0} at {1} cannot convert to DMA due to numPlanes exceeds limit.", op->getName(),
                                  op->getLoc());
                        return false;
                    }

                    log.trace("SwKernelOp at {0} can convert to DMA.", op->getLoc());
                    return true;
                } else if (isTileSwKernel(swKernelOp)) {
                    // At VPUX37XX: VPU::Tile -> VPUIP::SwKernelOp -> VPUIP::PerAxisTileDMA
                    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
                    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

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
            .Case<VPUIP::UpsamplingOp>([&](VPUIP::UpsamplingOp op) {
                const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                if (inType.getDimsOrder() != DimsOrder::NCHW && inType.getDimsOrder() != DimsOrder::NHWC) {
                    return false;
                }

                const auto inputShape = getShape(op.getInput());
                const auto upsamplingFactorVector = parseIntArrayAttr<int64_t>(op.getUpsamplingFactor());

                // UpsamplingDMA only supports 4D Input shape
                // UpsamplingDMA supports pads only for 3 axes
                // UpsamplingDMA supports factors only for 3 axes
                return (inputShape.size() != 4 || upsamplingFactorVector.size() != 3);
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
    const auto arch = config::getArch(op);
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto dmaPortNum = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    VPUX_THROW_WHEN(dmaPortNum <= 0, "Number of ports should be a positive integer, while it is {0}", dmaPortNum);
    if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        if (VPUIP::isDepthToSpaceSwKernel(swKernelOp)) {
            return !::isBeneficialForUsingSWDepthToSpace(swKernelOp, arch);
        } else if (VPUIP::isSpaceToDepthSwKernel(swKernelOp) || VPUIP::isTileSwKernel(swKernelOp)) {
            return true;
        } else if (VPUIP::isMemPermSwKernel(swKernelOp)) {
            auto memPerm = getMemPermFromSwKernel(swKernelOp);
            VPUX_THROW_UNLESS(memPerm.has_value(), "Cannot extract mem_perm attribute from permute SwKernel '{0}'.",
                              swKernelOp.getLoc());

            const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
            return isBeneficialForUsingPermuteDMA(config::getArch(op), inputType, outputType, memPerm.value(),
                                                  dmaPortNum, log);
        }

        return false;
    }
    return true;
}

VPUIP::DMADescriptorAttr vpux::VPUIP::updateNumPlanes(VPUIP::DMADescriptorAttr dmaDescriptor, int64_t numPlane) {
    auto ctx = dmaDescriptor.getContext();
    auto numPlaneAttr = vpux::getIntAttr(ctx, numPlane);
    return VPUIP::DMADescriptorAttr::get(ctx, numPlaneAttr, dmaDescriptor.getLen(), dmaDescriptor.getSrcWidth(),
                                         dmaDescriptor.getSrcStride(), dmaDescriptor.getSrcPlaneStride(),
                                         dmaDescriptor.getDstWidth(), dmaDescriptor.getDstStride(),
                                         dmaDescriptor.getDstPlaneStride());
}

bool vpux::VPUIP::isMemPermSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelTask.getKernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "reorder";
}

std::optional<mlir::AffineMap> vpux::VPUIP::getMemPermFromSwKernel(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isMemPermSwKernel(swKernelTask)) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().empty(),
                    "Cannot get VPUIP.SwKernelRun at '{0}'", swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.getAttrs().has_value(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get reversed permute value
    const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
    VPUX_THROW_WHEN(arrayAttrs.empty(), "Empty attribute at '{0}'", kernelRun->getLoc());
    auto reversedPermute = parseIntArrayAttr<unsigned>(mlir::dyn_cast<mlir::ArrayAttr>(arrayAttrs.getValue()[0]));
    auto permute = correctPermutation(reversedPermute);
    return mlir::AffineMap::getPermutationMap(permute, swKernelTask->getContext());
}

bool vpux::VPUIP::isDepthToSpaceSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelTask.getKernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "depth_to_space";
}

bool isDistributedOutputTypeIncompatibleOverlapped(VPU::ClusteredOpInterface prevOp, VPU::ClusteredOpInterface curOp,
                                                   int64_t numClusters) {
    if (prevOp == nullptr || curOp == nullptr) {
        return false;
    }
    auto resultType = mlir::cast<vpux::NDTypeInterface>(prevOp->getResult(0).getType());
    auto outputDistributedType = getDistributedOutputTypeFromOp(prevOp, resultType, numClusters);
    auto distributedTensor = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outputDistributedType);

    if (distributedTensor == nullptr) {
        return false;
    }

    auto mode = distributedTensor.getDistribution().getMode().getValue();
    if (mode != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }

    // check if the overlapped mode is compatible with the curOp
    auto curOpInputDistributedType =
            mlir::dyn_cast<vpux::VPU::DistributedTensorType>(getDistributedActivationTypeFromOp(
                    curOp, curOp->getOperand(0), curOp->getOperand(0).getType(), numClusters));
    if (curOpInputDistributedType == nullptr) {
        return false;
    }
    return mlir::failed(VPU::areDistributionAttrsCompatible(distributedTensor, curOpInputDistributedType, false));
}

bool vpux::VPUIP::isCompatibleWithMultiClusterNNDMA(VPU::DepthToSpaceOp op, vpux::ShapeRef nTilesOnDim) {
    if (op.getMode() != IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return false;
    }
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    const auto inOrder = inputType.getDimsOrder();
    const auto outOrder = outputType.getDimsOrder();
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC) {
        return false;
    }
    const auto inputShape = inputType.getShape();

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(op));
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

    if (inputShape[Dims4D::Act::H] > dmaMaxNumPlanes) {
        // TODO: split more DMAs when the numPlanes is larger than 256 [Track number: E#57027]
        return false;
    }

    // Check previous op
    auto parentOp = op->getOperand(0).getDefiningOp();
    if (parentOp == nullptr) {
        return false;
    }

    // check if VPU.Slice on the way
    // ClusteredNCEOp -> SliceOp -> D2S
    bool intermediateSliceOp = false;
    if (auto sliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(parentOp)) {
        parentOp = sliceOp.getSource().getDefiningOp();
        intermediateSliceOp = true;
    }

    auto prevOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(parentOp);
    if (prevOp == nullptr) {
        return false;
    }

    auto prevOpStrategyAttr = prevOp.getMultiClusterStrategy();
    if (!prevOpStrategyAttr.has_value() || prevOpStrategyAttr.value() != VPU::MultiClusterStrategy::SplitOverHeight) {
        return false;
    }
    auto module = prevOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(module);
    auto numClusters = tileOp.getCount();
    VPUX_THROW_WHEN(numClusters <= 0, "Number of clusters should be a positive integer, while it is {0}", numClusters);

    // For VPUX40XX all SOH are SOH-overlapped tile them now
    // Support for overlapped buffers will be added with E#86818
    // With intermediate sliceOp there will be a spill
    if (!intermediateSliceOp &&
        isDistributedOutputTypeIncompatibleOverlapped(
                prevOp, mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation()), numClusters)) {
        return false;
    }

    auto hasSingleUser = [](mlir::Operation* op) -> bool {
        if (op->hasOneUse()) {
            return true;
        }
        auto user = *op->getUsers().begin();
        for (auto otherUser : llvm::drop_begin(op->getUsers())) {
            if (otherUser != user) {
                return false;
            }
        }
        return true;
    };

    // Check next ops
    for (auto nextOp : op->getUsers()) {
        while (VPU::isPureViewOp(nextOp)) {
            if (!hasSingleUser(nextOp)) {
                return false;
            }
            nextOp = *nextOp->getUsers().begin();
        }
        auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nextOp);
        if (nceOp == nullptr) {
            return false;
        }

        auto strategyAttr = nceOp.getMultiClusterStrategy();
        if (!strategyAttr.has_value()) {
            return false;
        }
        auto strategy = strategyAttr.value();
        if (strategy != VPU::MultiClusterStrategy::SplitOverHeight && strategy != VPU::MultiClusterStrategy::HKSwitch) {
            return false;
        }
        if (isDistributedOutputTypeIncompatibleOverlapped(mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation()),
                                                          nceOp, numClusters)) {
            return false;
        }
    }

    // Only support SOH and when numTiles is smaller than numClusters
    if (nTilesOnDim[Dims4D::Act::H] > numClusters) {
        return false;
    }
    // No tile on other axis
    if (nTilesOnDim[Dims4D::Act::C] != 1 || nTilesOnDim[Dims4D::Act::W] != 1) {
        return false;
    }
    return true;
}

bool vpux::VPUIP::isSpaceToDepthSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelTask.getKernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "space_to_depth";
}

bool vpux::VPUIP::isTileSwKernel(VPUIP::SwKernelOp swKernelTask) {
    auto module = swKernelTask->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelTask.getKernelFunctionAttr());
    if (!kernelFunc) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (!kernelEntryPoint) {
        return false;
    }

    return kernelEntryPoint.getValue() == "tile";
}

bool vpux::VPUIP::isPerAxisTileSwKernel(VPUIP::SwKernelOp swKernelTask) {
    if (!isTileSwKernel(swKernelTask)) {
        return false;
    }

    // Only support TileOp with one Axis expansion
    const auto inType = mlir::cast<vpux::NDTypeInterface>(swKernelTask->getOperand(0).getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(swKernelTask->getOperand(1).getType());
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

std::optional<VPUIP::DepthToSpaceReturnType> vpux::VPUIP::getDepthToSpaceSwKernelAttr(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isDepthToSpaceSwKernel(swKernelTask)) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().empty(),
                    "Cannot get VPUIP.SwKernelRun at '{0}'", swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.getAttrs().has_value(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get DepthToSpace attrs
    const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
    VPUX_THROW_WHEN(arrayAttrs.empty(), "Empty attribute at '{0}'", kernelRun->getLoc());

    auto blockSizeAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[0]);
    auto mode = IE::DepthToSpaceMode(mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[1]).getInt());
    auto modeAttr = IE::DepthToSpaceModeAttr::get(swKernelTask.getContext(), mode);
    auto paddedInChannels =
            arrayAttrs.getValue().size() > 2 ? mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[2]) : nullptr;
    auto paddedOutChannels =
            arrayAttrs.getValue().size() > 3 ? mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[3]) : nullptr;
    VPUX_THROW_WHEN(blockSizeAttr == nullptr || modeAttr == nullptr, "Empty DepthToSpace attribute at '{0}'",
                    kernelRun->getLoc());

    auto paddedChannels =
            (paddedInChannels != nullptr && paddedOutChannels != nullptr)
                    ? IE::ChannelPaddingAttr::get(swKernelTask.getContext(), paddedInChannels, paddedOutChannels)
                    : nullptr;

    return VPUIP::DepthToSpaceReturnType(modeAttr, blockSizeAttr, paddedChannels);
}

std::optional<std::pair<IE::SpaceToDepthModeAttr, mlir::IntegerAttr>> vpux::VPUIP::getSpaceToDepthSwKernelAttr(
        VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelTask)) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().empty(),
                    "Cannot get VPUIP.SwKernelRun at '{0}'", swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.getAttrs().has_value(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    // get SpaceToDepth attrs
    const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(arrayAttrs.size() == 2, "Wrong numbers of attribute at '{0}', expected 2 but got '{1}'",
                      kernelRun->getLoc(), arrayAttrs.size());

    auto blockSizeAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[0]);
    auto modeIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrayAttrs.getValue()[1]);
    VPUX_THROW_UNLESS(blockSizeAttr != nullptr && modeIntAttr != nullptr,
                      "Failed to extract block size and mode at '{0}'", kernelRun->getLoc());

    auto modeAttr =
            IE::SpaceToDepthModeAttr::get(swKernelTask.getContext(), IE::SpaceToDepthMode(modeIntAttr.getInt()));
    VPUX_THROW_WHEN(blockSizeAttr == nullptr || modeAttr == nullptr, "Empty SpaceToDepth attribute at '{0}'",
                    kernelRun->getLoc());

    return std::pair<IE::SpaceToDepthModeAttr, mlir::IntegerAttr>(modeAttr, blockSizeAttr);
}

std::optional<VPUIP::PerAxisTileAttr> vpux::VPUIP::getPerAxisTileSwKernelAttr(VPUIP::SwKernelOp swKernelTask) {
    if (!VPUIP::isPerAxisTileSwKernel(swKernelTask)) {
        return std::nullopt;
    }

    // get PerAxisTile attrs
    VPUX_THROW_WHEN(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().empty(),
                    "Cannot get VPUIP.SwKernelRun at '{0}'", swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.getAttrs().has_value(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(arrayAttrs.size() == 2, "Wrong numbers of attribute at '{0}', expected 2 but got '{1}'",
                      kernelRun->getLoc(), arrayAttrs.size());

    auto repeatsAttr = mlir::dyn_cast<mlir::ArrayAttr>(arrayAttrs.getValue()[1]);
    VPUX_THROW_UNLESS(repeatsAttr != nullptr, "Failed to extract repeatsAttr at '{0}'", kernelRun->getLoc());

    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };

    auto repeats = parseIntArrayAttr<int64_t>(repeatsAttr);
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

    const auto getMergedShape = [](const MemShape& shape, int64_t axis) -> Shape {
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

SmallVector<vpux::Shape> vpux::VPUIP::getPerAxisTileDMASubShapes(config::ArchKind arch, vpux::ShapeRef shape) {
    const auto shapeSize = shape.size();
    VPUX_THROW_UNLESS(shapeSize == 3, "PerAxisTile merged Shape size should be 3, but got {0}", shapeSize);

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(arch);
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

    const auto totalNumPlane = shape[Dim(0)];
    auto numberDMAs = divUp(totalNumPlane, dmaMaxNumPlanes);
    if (numberDMAs > 1) {
        auto subShape = Shape(shape.raw());
        subShape[Dim(0)] = dmaMaxNumPlanes;
        SmallVector<Shape> subOutputShapes(numberDMAs - 1, subShape);
        subShape[Dim(0)] = totalNumPlane - dmaMaxNumPlanes * (numberDMAs - 1);
        subOutputShapes.push_back(subShape);
        return subOutputShapes;
    }
    return SmallVector<Shape>{Shape(shape.raw())};
}

VPURT::DeclareBufferOp vpux::VPUIP::createNewDeclareBuffer(mlir::PatternRewriter& rewriter,
                                                           mlir::Operation* insertionPoint,
                                                           VPURT::DeclareBufferOp declBuff,
                                                           vpux::NDTypeInterface newType, int64_t offset) {
    auto ctx = declBuff->getContext();
    auto section = declBuff.getSection();
    int64_t sectionIndex = declBuff.getSectionIndex().has_value()
                                   ? parseIntArrayAttr<int64_t>(declBuff.getSectionIndex().value())[0]
                                   : -1;
    const auto symbolAttr =
            sectionIndex == -1
                    ? vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)))
                    : vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)), sectionIndex);

    if (mlir::isa<VPUIP::DistributedBufferType>(newType)) {
        VPUX_THROW_WHEN(declBuff.getSection() != VPURT::BufferSection::CMX_NN,
                        "Section of DistributedBufferType should point to CMX_NN!");
    } else {
        newType = newType.changeMemSpace(symbolAttr);
    }

    return sectionIndex == -1
                   ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, declBuff->getLoc(), newType,
                                                             section, nullptr, offset, declBuff.getSwizzlingKeyAttr())
                   : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, declBuff->getLoc(), newType,
                                                             section, declBuff.getSectionIndex().value(), offset,
                                                             declBuff.getSwizzlingKeyAttr());
}

vpux::NDTypeInterface vpux::VPUIP::changeShapeWithMemShape(vpux::NDTypeInterface* type, vpux::ShapeRef newMemShape,
                                                           DimsOrder order) {
    auto newShape = order.toLogicalOrder(DimsOrder::NCHW.toMemoryOrder(newMemShape));
    return type->changeShape(ShapeRef(newShape));
}
