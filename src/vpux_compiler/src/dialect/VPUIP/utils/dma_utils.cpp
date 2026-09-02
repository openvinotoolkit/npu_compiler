//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
// Expand DMA helper, to keep the original strides but change the shapes for output type (from which we subtract the
// padding info).
vpux::NDTypeInterface changeShapeAndKeepStrides(vpux::NDTypeInterface type, vpux::ShapeRef newShape) {
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
}  // namespace vpux

namespace vpux::VPUIP {

int64_t getDMAPortValue(mlir::Operation* wrappedTaskOp) {
    if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp)) {
        auto portAttr = dmaOp.getPortAttribute();
        if (portAttr == nullptr) {
            return 0;
        }
        return portAttr.getInt();
    }

    VPUX_THROW("Could not cast to DMA task '{0}'", *wrappedTaskOp);
}

std::string getDMAChannelTypeAsString(VPUIP::DmaChannelType channelType, config::ArchKind arch) {
    if (arch <= config::ArchKind::NPU37XX) {
        return "";
    }

    return stringifyEnum(channelType).str();
}

std::string getDMAChannelTypeAsString(int64_t dmaQueueIdEncoding, config::ArchKind arch) {
    if (arch <= config::ArchKind::NPU37XX) {
        return "";
    }

    return stringifyEnum(getDMAChannelTypeFromEncodedId(dmaQueueIdEncoding, arch)).str();
}

DMATransaction getDMATransactionFromExpand(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                           mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd, bool stridedInput,
                                           bool stridedOutput) {
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

    dmaTransaction.inputs.push_back(reduceDimsForDma(std::move(inMemDims), std::move(inMemStrides),
                                                     inType.getElemTypeSize().count(), stridedInput));

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

    dmaTransaction.outputs.push_back(reduceDimsForDma(std::move(outMemDims), std::move(outMemStrides),
                                                      newOutType.getElemTypeSize().count(), stridedOutput));

    return dmaTransaction;
}

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::AffineMap loopOrder,
                                                bool stridedInput, bool stridedOutput) {
    return getDMATransactionFromPermutation(inType, outType, mappingOrder,
                                            VPUIP::getSmallVectorFromAffineMap(loopOrder), stridedInput, stridedOutput);
}

DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::SmallVector<int64_t> loopOrder,
                                                bool stridedInput, bool stridedOutput) {
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
                                                     inType.getElemTypeSize().count(), stridedInput));

    dmaTransaction.outputs.push_back(reduceDimsForDma(std::move(outPermutedMemDims), std::move(outPermutedMemStrides),
                                                      outType.getElemTypeSize().count(), stridedOutput));

    return dmaTransaction;
}

}  // namespace vpux::VPUIP
