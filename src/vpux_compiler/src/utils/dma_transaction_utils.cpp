//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include <cstdint>
#include <vpux/compiler/core/attributes/strides.hpp>
#include <vpux/compiler/dialect/VPUIP/utils/utils.hpp>
#include <vpux/compiler/utils/dma_transaction_utils.hpp>
#include <vpux/compiler/utils/types.hpp>
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
// Expand DMA helper, to keep the original strides but change the shapes for output type (from which we subtract the
// padding info).
namespace {
NDTypeInterface changeShapeAndKeepStrides(vpux::NDTypeInterface type, vpux::ShapeRef newShape) {
    auto originalStrides = type.getStrides();
    auto newType = type;
    auto distributedCopy = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(type);
    if (distributedCopy == nullptr || !distributedCopy.containsDistributedTypes()) {
        newType = newType.changeShape(newShape);
        return newType.changeStrides(originalStrides);
    }

    auto outputType = mlir::cast<vpux::VPUIP::DistributedBufferType>(distributedCopy.getDistributedTypes().front());
    const auto oldDistribution = outputType.getDistribution();
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(oldDistribution)) {
        newType = newType.changeShape(newShape);
        return newType.changeStrides(originalStrides);
    }

    const auto newDistribution = VPU::getNonOverlappedDistributedAttr(
            newShape, oldDistribution.getMode(), oldDistribution.getNumTiles(), oldDistribution.getNumClusters(),
            oldDistribution.getAlignment(), oldDistribution.getUniformDistributedSegments(), type.getContext());

    newType = outputType.changeShapeForExplicitDistribution(newShape, newDistribution);
    return newType.changeStrides(originalStrides);
}
}  // namespace

//
// Below function reduceDimsForDma, tries to collapse contiguous dimensions in memory
// depending on strides being compact; such that we'll end up with the most reduced
// transfer in terms of dimensionality;
// This is beneficial since the NPU DMA engines are limited in terms of the rank of the transfers
// 3D for NPU37XX; 6D for NPU40XX
//
// The general relation between sizes and strides for memrefs is
// Size[X] is the amount of elements at the index of X;
// Stride[X] is the amount of elements to jump in memory IN BETWEEN different elements of Size[X]
// Size:   [ 1,  2, 3, 4]
// Stride: [24, 12, 4, 1]
//
// When dealing with memory shapes and memory strides; the highest index represent the
// innermost data which is contiguous in memory, to which strides can also be applied;
// Plus memStrides also include the element type size in bits factored in.
//
// When factoring in layout, memory shapes and strides will result from the layout permutation
// applied to the logical shapes and strides.
//
// Example:
// Element type FP16
// Size:   [ 1,  2, 3, 4]
// Stride: [24,  1, 8, 2]
// Layout #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// MemSize   [1, 3, 4, 2]
// MemStride [24 * 16, 8 * 16, 2 * 16, 1 * 16]
//
// Then for the task of reducing the transfer dimensionality; we're doing the following steps:
// For example memref<1x2x3x4xf16, {order = #NCHW, strides = [64, 32, 8, 2]}
// MemSize:   [ 1,       2,      3,      4]
// MemStride: [64 * 16, 32 * 16, 8 * 16, 2 * 16]
// Here we have a two non contiguous levels of stride at indexes [1, 3]
//
// While when dealing with DMA transactions and concepts of sizes and strides;
//   Size[X] is the amount of data to transfer at the index of X, in terms of bytes
// Stride[X] is the amount of bytes to jump in memory AFTER all Size[X] were transferred
// Because we have a discrepancy with regards to what strides mean for memrefs and what
//  they mean for DMA transactions, we need to arrive at an aligned representation.
//
// So if we'd take the previous example, we'd extend them as such for DMA specific logic;
// Size:   [ 1,       2,       3,      4,     16] // by adding a new dim at end representing
//                                                type size in bits
// Stride: [64 * 16, 64 * 16, 32 * 16, 8 * 16, 2 * 16] // by adding a new dim at start
//                                  for the stride over all elements of size[0]
//
// Further taking this into consideration; if we'd like to reduce the array of size/strides
// based on the indexes which are contiguous, we'd be left with the following:
// Size:   [ 2,      12,      16]
// Stride: [64 * 16, 32 * 16, 2 * 16]
//
// At the same dimension reduction stage, we align the dimensions on byte boundary; such that
// it can be transferred by DMA; the engine does not support sub byte access
// Size:   [ 2,  12, 2] // representing dimensions of DMA transfer in bytes
// Stride: [128, 64, 4] // representing strides in bytes
//
// This is generic enough and can be used for all HW platforms to define the DMA transaction
//
// A pseudo code logic for how DMA engine executes the transfer would be:
//
// offset0 = 0;
// for(i0 = 0; i0 < SIZE[0]; i0++) {
//     offset1 = 0
//     for(i1 = 0; i1 < SIZE[1]; i1++) {
//         offset2 = 0
//         for(i2 = 0; i2 < SIZE[2]; i2++) {
//             Do_memory_access(SRC_ADDR + sum(offset<2,1,0>));
//             offset2 ++;
//         }
//         offset1 += STRIDE[2]
//     }
//     offset0 += STRIDE[1]
// }
//
// This fits well with our representation of DMA transaction dims where the innermost dim
// represents the amount of bytes to transfer and the outermost dims are just plane dimensions
// to further iterate over;
//
// Having only the individual reduced dimensions defined as above fits perfectly with the logic
// of how NPU40XX DMA works, since it does nesting for loops over the dimensions to transfer
// all data.
//
// Some extra adaptations need to be done for NPU37XX see function patchDimsForNPU37XX
//
// For NPU37XX style DMA it's a bit more peculiar, for which reason we need to product
// accumulate the sizes from innermost to outermost dimension.
// So we'd have:
// Size:   [2, 12, 2]
//   V
// Size:   [2 * 12 * 2, 12 * 2, 2]
//   V
// Size:   [48, 24, 2]
//
// Then specifically for generation of NPU37XX descriptors which are driven by a total length and
// distributing that length in memory with up to 2 levels of strides
// We'd further reduce the above case to
// Total length: 48
// Size:   [24, 2]
// Stride: [64, 4]
//

DMAPattern reduceDimsForDma(SmallVector<int64_t> memShape, SmallVector<vpux::MemSize<vpux::MemType::Bit>> memStrides,
                            int64_t elemSize) {
    // Get memory view of shape and strides
    // Store them all as bits so to handle sub byte types

    // Extend shape and strides to accommodate for element type size and batch stride
    memShape.push_back(elemSize);
    memStrides.insert(memStrides.begin(), memStrides.front() * memShape.front());
    auto innerMostIndex = memShape.size() - 1;

    llvm::SmallVector<Bit> reducedBitDims;
    llvm::SmallVector<Bit> reducedBitStrides;

    const auto alignToByteBoundary = [&](Bit val) {
        return alignMemSize(Bit(val), Byte(1));
    };

    // Iterate over dim/stride pairs and push cases of non compact strides
    int64_t accumulatedSize = 1;
    auto previousStrideInBits = Bit(1);
    for (int64_t dim = innerMostIndex; dim >= 0; --dim) {
        auto currentSize = memShape[dim];
        auto currentStrideInBits = memStrides[dim];

        accumulatedSize *= currentSize;
        // Found non-compact stride
        if (checked_cast<int64_t>(currentSize) * previousStrideInBits != currentStrideInBits) {
            reducedBitDims.push_back(Bit(accumulatedSize));
            reducedBitStrides.push_back(currentStrideInBits);
            accumulatedSize = elemSize;
        }

        previousStrideInBits = currentStrideInBits;
    }

    // Flush out remaining accumulated sizes.
    // Also handle scalar cases of 1 byte transfers.
    if (accumulatedSize > elemSize || reducedBitDims.empty()) {
        reducedBitDims.emplace_back(accumulatedSize);
        reducedBitStrides.push_back(memStrides.front());
    }

    // Align dim[0] to byte size and divide by elemSize
    // Only src/dst dim[0] represents data movement in terms of bytes, all subsequent dims represent a loop over that
    // dimension and don't have the same restrictions
    SmallVector<Bit> reducedBitDimsByteAligned;
    reducedBitDimsByteAligned.reserve(reducedBitDims.size());
    if (!reducedBitDims.empty()) {
        reducedBitDimsByteAligned.push_back(alignToByteBoundary(reducedBitDims[0]));
        std::copy(reducedBitDims.begin() + 1, reducedBitDims.end(), std::back_inserter(reducedBitDimsByteAligned));
    }

    SmallVector<int64_t> reducedDims;
    reducedDims.reserve(reducedBitDimsByteAligned.size());
    std::transform(reducedBitDimsByteAligned.begin(), reducedBitDimsByteAligned.end(), std::back_inserter(reducedDims),
                   [elemSize](const Bit& bitDim) {
                       return checked_cast<int64_t>(bitDim.count() / elemSize);
                   });

    // Innermost dim is special
    reducedDims.front() = Bit(reducedDims.front() * elemSize).to<Byte>().count();

    VPUX_THROW_WHEN(std::any_of(std::begin(reducedBitStrides), std::prev(std::end(reducedBitStrides)),
                                [](auto bitStride) {
                                    return bitStride.count() % CHAR_BIT != 0;
                                }),
                    "Non byte aligned stride value");

    // Align all strides to byte size
    auto reducedStrides = to_small_vector(reducedBitStrides | vpux::transformed(alignToByteBoundary) |
                                          vpux::transformed([](auto bitStride) {
                                              return checked_cast<int64_t>(bitStride.template to<Byte>().count());
                                          }));

    // Validate byte alignment for strides
    for (auto idx : irange(reducedBitStrides.size() - 1)) {
        VPUX_THROW_WHEN(reducedBitStrides[idx].count() % CHAR_BIT,
                        "Non byte aligned stride value '{0}' at index '{1}'.", reducedBitStrides[idx].count(), idx);
    }

    // Reverse the arrays such that we keep the same logical order of dimensions as memrefs
    std::reverse(reducedDims.begin(), reducedDims.end());
    std::reverse(reducedStrides.begin(), reducedStrides.end());

    return DMAPattern(std::move(reducedDims), std::move(reducedStrides));
}

void patchDimsForNPU37XX(DMAPattern& dmaPattern) {
    // NPU37XX hacks from here on

    // Re - accumulate the reduced shape dims, for NPU37XX purpose;
    int64_t accum = dmaPattern.dims.back();
    for (auto rIt = dmaPattern.dims.rbegin() + 1; rIt != dmaPattern.dims.rend(); ++rIt) {
        *rIt *= accum;
        accum = *rIt;
    }

    // Drop first pair of dims strides, those will be deduced based on totalLength
    if (dmaPattern.dims.size() > 1) {
        dmaPattern.dims.erase(dmaPattern.dims.begin());
        dmaPattern.strides.erase(dmaPattern.strides.begin());
    }
}

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::AffineMap loopOrder) {
    return getDMATransactionFromPermutation(inType, outType, mappingOrder,
                                            VPUIP::getSmallVectorFromAffineMap(loopOrder));
}

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::SmallVector<int64_t> loopOrder) {
    DMATransaction dmaTransaction;

    auto ctx = inType.getContext();

    VPUX_THROW_WHEN(inType.getRank() != outType.getRank(), "Rank mismatch between input and output types");
    VPUX_THROW_WHEN(inType.getRank() != mappingOrder.getNumDims(),
                    "Rank mismatch between input type and mapping order");
    VPUX_THROW_WHEN(inType.getRank() != static_cast<int64_t>(loopOrder.size()),
                    "Rank mismatch between input type and loop order");

    // Mapping order maps out logical dims to in logical dims
    // This mapping allows to find the in logical dim corresponding to a given out logical dim

    // Define explicit vars to avoid confusion
    // Mapping to find out logical dim based on given in logical dim
    auto mappingLogicalInToOut = VPUIP::getSmallVectorFromAffineMap(mlir::inversePermutation(mappingOrder));
    // Mapping to find in logical dim based on given out logical dim
    auto mappingLogicalOutToIn = VPUIP::getSmallVectorFromAffineMap(mappingOrder);

    // Dims order maps memory dims to logical dims

    // Mapping to find logical dim corresponding to a given mem dim
    auto inDimsOrderAffineMap = inType.getDimsOrder().toAffineMap(ctx);
    auto outDimsOrderAffineMap = outType.getDimsOrder().toAffineMap(ctx);

    // Define explicit vars to avoid confusion
    // Mapping to find in mem dim based on given in logical dim
    auto mappingInLogicalToMem = VPUIP::getSmallVectorFromAffineMap(mlir::inversePermutation(inDimsOrderAffineMap));
    // Mapping to find in logical dim based on given in mem dim
    auto mappingInMemToLogical = VPUIP::getSmallVectorFromAffineMap(inDimsOrderAffineMap);

    // Mapping to find out mem dim based on given out logical dim
    auto mappingOutLogicalToMem = VPUIP::getSmallVectorFromAffineMap(mlir::inversePermutation(outDimsOrderAffineMap));
    // Mapping to find out logical dim based on given out mem dim
    auto mappingOutMemToLogical = VPUIP::getSmallVectorFromAffineMap(outDimsOrderAffineMap);

    // Original in and out mem dims and mem strides
    auto inDims = to_small_vector(inType.getShape());
    auto inStrides = to_small_vector(inType.getStrides());
    auto inMemDims = to_small_vector(inType.getMemShape());
    auto inMemStrides = to_small_vector(inType.getMemStrides());
    auto outDims = to_small_vector(outType.getShape());
    auto outStrides = to_small_vector(outType.getStrides());
    auto outMemDims = to_small_vector(outType.getMemShape());
    auto outMemStrides = to_small_vector(outType.getMemStrides());

    // Permuted (i.e. after applying in to out permutation and loop order) mem dims and mem strides
    auto inPermutedMemDims = inMemDims;
    auto inPermutedMemStrides = inMemStrides;
    auto outPermutedMemDims = outMemDims;
    auto outPermutedMemStrides = outMemStrides;

    auto loopOrderVec(std::move(loopOrder));

    // All sizes here should be equal, but for now just check against inMemDims size
    VPUX_THROW_WHEN(loopOrderVec.size() != inMemDims.size(), "Partial iteration over input is not supported");

    for (auto index : irange(inType.getRank())) {
        // Get in logical dim to process
        auto inLogicalDimIndex = loopOrderVec[index];
        // Get in memory dim to process
        auto inMemDimIndex = mappingInLogicalToMem[inLogicalDimIndex];

        // Paranoia checks
        VPUX_THROW_WHEN(inDims[inLogicalDimIndex] != inMemDims[inMemDimIndex],
                        "Mismatch between logical dim size and mem dim size");
        VPUX_THROW_WHEN(inStrides[inLogicalDimIndex] != inMemStrides[inMemDimIndex],
                        "Mismatch between logical dim stride and mem dim stride");

        // Get mem dim and stride for current in logical dim
        inPermutedMemDims[index] = inMemDims[inMemDimIndex];
        inPermutedMemStrides[index] = inMemStrides[inMemDimIndex];

        // Get corresponding out logical dim
        auto outLogicalDimIndex = mappingLogicalInToOut[inLogicalDimIndex];

        // Map out logical dim to out memory dim
        auto outMemDimIndex = mappingOutLogicalToMem[outLogicalDimIndex];

        // Paranoia checks
        VPUX_THROW_WHEN(outDims[outLogicalDimIndex] != outMemDims[outMemDimIndex],
                        "Mismatch between logical dim size and mem dim size");
        VPUX_THROW_WHEN(outStrides[outLogicalDimIndex] != outMemStrides[outMemDimIndex],
                        "Mismatch between logical dim stride and mem dim stride");

        outPermutedMemDims[index] = outMemDims[outMemDimIndex];
        outPermutedMemStrides[index] = outMemStrides[outMemDimIndex];

        VPUX_THROW_WHEN(inPermutedMemDims[index] != outPermutedMemDims[index], "Dim size mismatch");
    }

    dmaTransaction.inputs.push_back(reduceDimsForDma(std::move(inPermutedMemDims), std::move(inPermutedMemStrides),
                                                     inType.getElemTypeSize().count()));

    dmaTransaction.outputs.push_back(reduceDimsForDma(std::move(outPermutedMemDims), std::move(outPermutedMemStrides),
                                                      outType.getElemTypeSize().count()));

    return dmaTransaction;
}

DMATransaction getDMATransactionFromExpand(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                           mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd) {
    // Only support ExpandDMA padding at end
    // TODO: support padding at begin E65670
    VPUX_THROW_WHEN(llvm::any_of(parseIntArrayAttr<int64_t>(padsBegin),
                                 [](auto padValue) {
                                     return padValue != 0;
                                 }),
                    "ExpandDMA doesn't support padding at begin!");

    DMATransaction dmaTransaction;
    auto inMemDims = to_small_vector(inType.getMemShape());
    auto inMemStrides = to_small_vector(inType.getMemStrides());

    dmaTransaction.inputs.push_back(
            reduceDimsForDma(std::move(inMemDims), std::move(inMemStrides), inType.getElemTypeSize().count()));

    auto padEnd = parseIntArrayAttr<int64_t>(padsEnd);
    const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };
    const auto padEndAxisIter = std::find_if(padEnd.begin(), padEnd.end(), nonZeroAxisPredicate);
    VPUX_THROW_WHEN(padEndAxisIter == padEnd.end(), "Can not find padding axis");

    const auto padEndAxis = std::distance(padEnd.begin(), padEndAxisIter);
    auto outShapes = outType.getShape().toValues();
    VPUX_THROW_UNLESS(outShapes[Dim(padEndAxis)] > padEnd[padEndAxis], "Can't subtract padding from shape!");
    outShapes[Dim(padEndAxis)] = outShapes[Dim(padEndAxis)] - padEnd[padEndAxis];

    auto newOutType = changeShapeAndKeepStrides(outType, outShapes);

    auto outMemDims = to_small_vector(newOutType.getMemShape());
    auto outMemStrides = to_small_vector(newOutType.getMemStrides());

    dmaTransaction.outputs.push_back(
            reduceDimsForDma(std::move(outMemDims), std::move(outMemStrides), newOutType.getElemTypeSize().count()));

    return dmaTransaction;
}

std::tuple<SmallVector<int64_t>, SmallVector<Bit>, int64_t> getTypeInfo(NDTypeInterface type) {
    auto elemSize = type.getElemTypeSize().count();
    auto memShape = to_small_vector(type.getMemShape());
    if (memShape.empty()) {
        memShape.push_back(1);
    }
    auto memStrides = to_small_vector(type.getMemStrides());
    if (memStrides.empty()) {
        memStrides.push_back(Bit(elemSize));
    }
    return {memShape, memStrides, elemSize};
}

}  // namespace vpux
