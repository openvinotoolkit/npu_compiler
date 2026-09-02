//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/distributed_buffer_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <numeric>

using namespace vpux;

namespace {
//
// Private helpers for reshape memory-dimension matching
//

SmallVector<MemDimArr> getOutMemDimsCandidates(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim) {
    const size_t targetDimSize = checked_cast<size_t>(inMemShape[inMemDim]);
    SmallVector<MemDimArr> outMemDims;
    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: 'H'
    // The 'targetDimSize' will have two candidates: [[0, 1],[2]]
    // Candidate 1: [0, 1] input 'H' split into output 'N' and 'C'
    // Candidate 2: [2] input 'H' split into output 'H'
    for (size_t dimIdx = 0; dimIdx < outMemShape.size(); dimIdx++) {
        size_t accumulateSize = 1;
        size_t beginIdx = dimIdx;
        MemDimArr currMemDims;
        while (beginIdx < outMemShape.size()) {
            accumulateSize = accumulateSize * outMemShape[MemDim(beginIdx)];
            currMemDims.push_back(MemDim(beginIdx));
            if (accumulateSize == targetDimSize) {
                outMemDims.push_back(currMemDims);
            } else if (accumulateSize > targetDimSize) {
                break;
            }
            beginIdx++;
        }
    }
    return outMemDims;
}
}  // namespace

std::optional<MemDimArr> VPUIP::deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape,
                                                         MemDim inMemDim) {
    const auto outMemDimsCandidates = getOutMemDimsCandidates(inMemShape, outMemShape, inMemDim);
    if (outMemDimsCandidates.empty()) {
        return std::nullopt;
    }

    auto getAccumulateSize = [](MemShapeRef memShape, auto beginIdx, auto endIdx) {
        VPUX_THROW_UNLESS(checked_cast<int32_t>(beginIdx) <= checked_cast<int32_t>(endIdx) &&
                                  memShape.begin() + endIdx <= memShape.end(),
                          "Got unexpect memShape");
        return std::accumulate(memShape.begin() + beginIdx, memShape.begin() + endIdx, int64_t(1),
                               std::multiplies<int64_t>());
    };

    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: H
    // The 'outMemDims' will have two candidates: [[0, 1],[2]]
    // Candidate 1: outMemDims is [0, 1]
    // inTotalLeftSize(1x512) != outTotalLeftSize(1) && inTotalRightSize(16) != outTotalRightSize(512x16)
    // Candidate 2: outMemDims is [2]
    // inTotalLeftSize(1x512) == outTotalLeftSize(16x32) && inTotalRightSize(16) == outTotalRightSize(16)
    // The candidate 2 is legal candidate
    for (auto& outMemDims : outMemDimsCandidates) {
        const auto inTotalLeftShapeSize = getAccumulateSize(inMemShape, 0, inMemDim.ind());
        const auto inTotalRightShapeSize = getAccumulateSize(inMemShape, inMemDim.ind() + 1, inMemShape.size());
        const auto outTotalLeftShapeSize = getAccumulateSize(outMemShape, 0, outMemDims.front().ind());
        const auto outTotalRightShapeSize =
                getAccumulateSize(outMemShape, outMemDims.back().ind() + 1, outMemShape.size());
        if (inTotalLeftShapeSize == outTotalLeftShapeSize && inTotalRightShapeSize == outTotalRightShapeSize) {
            return outMemDims;
        }
    }
    return std::nullopt;
}

namespace {
//
// Private helpers for reshape stride propagation
//

// If inputType has strides, infer the corresponding output type according to the output shape.
// If return 'std::nullopt', the output type cannot be inferred.
// This function only infer input stride only exist on one axis.
// Assume that there is a input MemShape [a, b, c, d] and MemStrides [2bcd, 2cd, 2d, 1], stridesDim: d2
// If output shape splits d2 into d3, cannot infer the output MemStride for d3 because it's not contiguous.
// If output shape splits d2 into d1, can infer the output MemStride for d1 because it's contiguous.
// This can be extended to the general case that the left product and right product split by stridesDim of input and
// output should be equal. The algorithm is as follows:
// 1. Find the 'stridesDim' in input type.
// 2. Split by 'stridesDim', input and output memory shape can be divided into two parts:
//    stridesMemDimLeftProduct: the product from highest dim to strided memDim
//    [inStridesMemDimLeftProduct,  inStridesMemDimRightProduct]
//    [outStridesMemDimLeftProduct, outStridesMemDimRightProduct]
//    inStridesMemDimLeftProduct should equal to outStridesMemDimLeftProduct
// 3. If stridesMemDimLeftProduct is equal to 1:
//    1x256x32x16, strides = [262144, 512, 16, 1] -> 256x32x16, strides = [512, 16, 1]
//    Although it seems that the corresponding strides info is lost, in NNDMA it will change its offset.
std::optional<Strides> inferReshapeOutputStrides(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    if (inType.getShape().totalSize() != outType.getShape().totalSize()) {
        return outType.getStrides();
    }

    const auto inStridesMemDims = VPUIP::getStridesMemDims(inType);
    if (inStridesMemDims.size() > 1) {
        return std::nullopt;
    }

    const auto outMemShape = outType.getMemShape();
    const auto outOrder = outType.getDimsOrder();
    const auto outElemSize = outType.getElemTypeSize();

    auto outMemStrides = StrideReqs::compact(outOrder.numDims()).calcStrides(outElemSize, outMemShape);

    if (inStridesMemDims.empty()) {
        return outOrder.toLogicalOrder(outMemStrides);
    }

    const auto inMemShape = inType.getMemShape();
    const auto inMemStrides = inType.getMemStrides();
    const auto inStridesMemDim = inStridesMemDims.front().ind();

    int64_t inStridesMemDimProduct = 1;
    for (size_t i = 0; i <= static_cast<size_t>(inStridesMemDim); i++) {
        inStridesMemDimProduct *= inMemShape.raw()[i];
    }

    int64_t outStridesMemDimProduct = 1;
    size_t outStridesDimRightBoundary = 0;
    const size_t outMemShapeSize = outMemShape.size();
    while (outStridesDimRightBoundary < outMemShapeSize) {
        outStridesMemDimProduct *= outMemShape.raw()[outStridesDimRightBoundary];
        if (outStridesMemDimProduct >= inStridesMemDimProduct) {
            break;
        }
        ++outStridesDimRightBoundary;
    }

    if (inStridesMemDimProduct != 1 && inStridesMemDimProduct != outStridesMemDimProduct) {
        return std::nullopt;
    }

    if (inStridesMemDimProduct == outStridesMemDimProduct) {
        outMemStrides.raw()[outStridesDimRightBoundary] = inMemStrides.raw()[inStridesMemDim];
        for (int64_t ind = static_cast<int64_t>(outStridesDimRightBoundary) - 1; ind >= 0; --ind) {
            const size_t currentMemDim = static_cast<size_t>(ind);
            const size_t prevMemDim = currentMemDim + 1;
            outMemStrides.raw()[currentMemDim] = outMemStrides.raw()[prevMemDim] * outMemShape.raw()[prevMemDim];
        }
    }

    return outOrder.toLogicalOrder(outMemStrides);
}
}  // namespace

mlir::FailureOr<vpux::NDTypeInterface> VPUIP::updateStridesForReshape(const vpux::NDTypeInterface& inType,
                                                                      const vpux::NDTypeInterface& outType) {
    const auto outputStrides = inferReshapeOutputStrides(inType, outType);
    if (!outputStrides.has_value()) {
        return mlir::failure();
    }
    const auto outputStridesVal = outputStrides.value();
    return outType.getStrides() != outputStridesVal ? outType.changeStrides(outputStridesVal) : outType;
}

// This function checks if inType and outType strides are compatible for multi stride dims.
// If they have compatible strides, they should meet:
// Both memory strides [memStride0, memStrideForDim1, memStride1, ...]
// and memory shape [leftMemShapeProductForMemStride0, 1, leftMemShapeProductForMemStride1, ...]
// should be equal without dim size 1.
// The algorithm checks if the corresponding memStrides and leftMemShapeProduct are equal for stride dimension.
bool VPUIP::isInAndOutStridesCompatible(const vpux::NDTypeInterface& inType, const vpux::NDTypeInterface& outType) {
    const auto inStridesMemDims = VPUIP::getStridesMemDims(inType);
    const auto outStridesMemDims = VPUIP::getStridesMemDims(outType);
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();
    const auto inMemStrides = inType.getMemStrides();
    const auto outMemStrides = outType.getMemStrides();

    // Store leftProduct and corresponding stride dim memStride
    struct ProductStrideInfo {
        int64_t product;
        Bit stride;

        bool operator==(const ProductStrideInfo& other) const {
            return product == other.product && stride == other.stride;
        }
    };

    SmallVector<ProductStrideInfo> leftProductsIn;
    SmallVector<ProductStrideInfo> leftProductsOut;

    const auto getLeftProducts = [](const auto& memShape, const auto& memStrides, const auto& stridesMemDims,
                                    auto& leftProducts) {
        int64_t leftProduct = 1;
        if (stridesMemDims.empty()) {
            return;
        }
        size_t cnt = 0;
        for (size_t i = 0; i < memShape.size(); ++i) {
            leftProduct *= memShape[MemDim(i)];
            if (cnt < stridesMemDims.size() && llvm::is_contained(stridesMemDims, MemDim(i))) {
                leftProducts.push_back({leftProduct, memStrides[MemDim(i)]});
                leftProduct = 1;
                ++cnt;
                continue;
            }
            if (cnt == stridesMemDims.size()) {
                break;
            }
        }
    };

    getLeftProducts(inMemShape, inMemStrides, inStridesMemDims, leftProductsIn);
    getLeftProducts(outMemShape, outMemStrides, outStridesMemDims, leftProductsOut);

    // Remove all elements with product equal to 1
    llvm::erase_if(leftProductsIn, [](const ProductStrideInfo& info) {
        return info.product == 1;
    });
    llvm::erase_if(leftProductsOut, [](const ProductStrideInfo& info) {
        return info.product == 1;
    });

    return leftProductsIn == leftProductsOut;
}

//
// Back type inference for reshape/shape-change view ops
//

// Propagate the element type across a shape-only view.
//
// Example 1: original op was f16 -> f16. If the input was rewritten to u8,
// infer the other side as u8, provided u8 is valid for the inferred shape.
// Example 2: original op was per-axis-quantized -> storage type. If the caller
// changes the known-side element type, there is no generic rule to reconstruct the
// opposite side, so the candidate is rejected.
std::optional<mlir::Type> VPUIP::inferReshapeElementType(mlir::Type newKnownElemType, mlir::Type origKnownElemType,
                                                         mlir::Type origOtherElemType, ShapeRef otherShape) {
    auto inferredElemType = newKnownElemType;

    if (origKnownElemType == newKnownElemType) {
        inferredElemType = origOtherElemType;
    } else if (origKnownElemType != origOtherElemType) {
        return std::nullopt;
    }

    if (!vpux::isSupportedElemTypeQuantization(inferredElemType, otherShape)) {
        return std::nullopt;
    }
    return inferredElemType;
}

std::optional<mlir::Type> VPUIP::inferReshapeOutputType(vpux::NDTypeInterface newInputNDType,
                                                        vpux::NDTypeInterface origInputNDType,
                                                        vpux::NDTypeInterface origOutputNDType, ShapeRef outShape) {
    const auto outElemType = inferReshapeElementType(newInputNDType.getElementType(), origInputNDType.getElementType(),
                                                     origOutputNDType.getElementType(), outShape);
    if (!outElemType.has_value()) {
        return std::nullopt;
    }

    if (auto distBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(newInputNDType)) {
        // For distributed reshape, the result type is not just the new logical
        // shape. The tiling axis has to be remapped to an output axis that
        // represents the same contiguous memory extent.
        // Example: 1x16x64x1 SEGMENTED over axis 2 -> 1x16x1x8x8 moves tiling
        // to the first non-unit dimension in the 1x8x8 output range.
        auto distribution = distBufType.getDistribution();
        const auto axesMapping =
                VPUIP::inferReshapeDistributedAxesMapping(newInputNDType, origOutputNDType, distribution);
        if (!axesMapping.has_value()) {
            return std::nullopt;
        }

        auto newDistAttr = VPUIP::changeDistributedAxisOnDistributionInfoAttr(distribution, axesMapping->first,
                                                                              axesMapping->second, outShape);
        auto ctx = distBufType.getContext();
        auto orderAttr = mlir::AffineMapAttr::get(origOutputNDType.getDimsOrder().toAffineMap(ctx));
        auto distributedOutType = VPUIP::createDistributedBufferTypeOrNull(
                ctx, outShape, outElemType.value(), orderAttr, distBufType.getMemSpace(), newDistAttr);
        if (!distributedOutType.has_value()) {
            return std::nullopt;
        }
        if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                    distBufType, distributedOutType.value())) {
            return std::nullopt;
        }
        return mlir::cast<mlir::Type>(distributedOutType.value());
    }

    return mlir::cast<mlir::Type>(newInputNDType.changeShapeElemType(outShape, outElemType.value()));
}

namespace {
//
// Private helpers for distributed reshape axis mapping
//

// Return the single source axis that carries SEGMENTED tiling. Alignment, when
// present, must describe the same axis. Alignment without num_tiles is rejected:
// the distribution verifier should not produce a segmented layout without a
// tiling axis, and back-inference should not guess one from alignment alone.
std::optional<int64_t> getReshapeDistributedSourceAxis(VPU::DistributionInfoAttr distribution) {
    const auto sourceAxis = VPUIP::getSpecificAxisFromAttr(distribution.getNumTiles());
    if (sourceAxis == -1) {
        return std::nullopt;
    }

    const auto alignmentAxis = VPUIP::getSpecificAxisFromAttr(distribution.getAlignment());
    if (alignmentAxis != -1 && sourceAxis != alignmentAxis) {
        return std::nullopt;
    }

    return sourceAxis;
}

std::optional<std::pair<int64_t, int64_t>> inferReshapeDistributedAxesMappingByEqualDimSize(
        vpux::NDTypeInterface sourceType, vpux::NDTypeInterface targetType, VPU::DistributionInfoAttr distribution) {
    // Example: source shape [1, 16, 64, 8] is segmented on axis 2 and target shape is [1, 64, 16, 8].
    // The tiled extent 64 appears once in the target shape, so the distribution can move from axis 2 to axis 1.
    const auto sourceAxis = getReshapeDistributedSourceAxis(distribution);
    if (!sourceAxis.has_value()) {
        return std::nullopt;
    }
    const auto sourceAxisVal = sourceAxis.value();

    const auto sourceShape = sourceType.getShape().raw();
    const auto targetShape = targetType.getShape().raw();
    if (sourceAxisVal >= checked_cast<int64_t>(sourceShape.size())) {
        return std::nullopt;
    }

    const auto sourceDimSize = sourceShape[checked_cast<size_t>(sourceAxisVal)];
    int64_t targetAxis = -1;
    for (auto ind : irange(targetShape.size())) {
        if (targetShape[ind] != sourceDimSize) {
            continue;
        }
        if (targetAxis != -1) {
            return std::nullopt;
        }
        targetAxis = checked_cast<int64_t>(ind);
    }

    if (targetAxis == -1) {
        return std::nullopt;
    }
    return std::make_pair(sourceAxisVal, targetAxis);
}

std::optional<std::pair<int64_t, int64_t>> inferReshapeDistributedAxesMappingByTileBoundary(
        vpux::NDTypeInterface sourceType, vpux::NDTypeInterface targetType, VPU::DistributionInfoAttr distribution) {
    // Example: source mem shape [1, 16, 64, 1] is segmented on the 64-dim and target mem shape is [1, 16, 1, 8, 8].
    // The tiled extent 64 becomes the contiguous target range 1x8x8; after skipping the leading 1, tiling can move to
    // the first 8-dim if it is divisible by the tiling factor.
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        return std::nullopt;
    }
    if (distribution.getMode().getValue() != VPU::DistributionMode::SEGMENTED ||
        distribution.getNumTiles() == nullptr) {
        return std::nullopt;
    }

    const auto sourceAxis = getReshapeDistributedSourceAxis(distribution);
    if (!sourceAxis.has_value()) {
        return std::nullopt;
    }
    const auto sourceAxisVal = sourceAxis.value();

    const auto tilingFactor = parseIntArrayAttr<int64_t>(distribution.getNumTiles())[sourceAxisVal];
    if (tilingFactor <= 1) {
        return std::nullopt;
    }

    const auto srcOrder = sourceType.getDimsOrder();
    const auto tgtOrder = targetType.getDimsOrder();
    const auto srcMemShape = srcOrder.toMemoryOrder(sourceType.getShape());
    const auto tgtMemShape = tgtOrder.toMemoryOrder(targetType.getShape());
    const auto srcMemDim = srcOrder.toMemDim(Dim(sourceAxisVal));

    const auto leftProduct = [](MemShapeRef memShape, int64_t endExclusive) {
        int64_t product = 1;
        for (int64_t idx = 0; idx < endExclusive; ++idx) {
            product *= memShape[MemDim(idx)];
        }
        return product;
    };

    const auto srcLeft = leftProduct(srcMemShape, srcMemDim.ind());
    const auto srcTiledExtent = srcMemShape[srcMemDim];

    // Match the source tiled memory extent to a contiguous target memory range. For 64 -> 1x8x8, use the first 8-dim as
    // the carrier instead of rejecting the range because it starts with a unit dim.
    int64_t targetTiledMemDim = -1;
    int64_t runningLeft = 1;
    for (int64_t idx = 0; idx < checked_cast<int64_t>(tgtMemShape.size()); ++idx) {
        if (runningLeft == srcLeft) {
            int64_t rangeProduct = 1;
            for (int64_t end = idx; end < checked_cast<int64_t>(tgtMemShape.size()); ++end) {
                rangeProduct *= tgtMemShape[MemDim(end)];
                if (rangeProduct == srcTiledExtent) {
                    auto candidateMemDim = idx;
                    while (candidateMemDim <= end && tgtMemShape[MemDim(candidateMemDim)] == 1) {
                        ++candidateMemDim;
                    }
                    if (candidateMemDim <= end && tgtMemShape[MemDim(candidateMemDim)] % tilingFactor == 0) {
                        targetTiledMemDim = candidateMemDim;
                    }
                    break;
                }
                if (rangeProduct > srcTiledExtent) {
                    break;
                }
            }
            break;
        }
        runningLeft *= tgtMemShape[MemDim(idx)];
        if (runningLeft > srcLeft) {
            break;
        }
    }

    if (targetTiledMemDim == -1) {
        return std::nullopt;
    }

    const auto targetAxis = tgtOrder.toDim(MemDim(targetTiledMemDim)).ind();
    return std::make_pair(sourceAxisVal, targetAxis);
}
}  // namespace

std::optional<std::pair<int64_t, int64_t>> VPUIP::inferReshapeDistributedAxesMapping(
        vpux::NDTypeInterface sourceType, vpux::NDTypeInterface targetType, VPU::DistributionInfoAttr distribution) {
    auto axesMappingOrFailure = VPUIP::getDistributedAxesMappingAfterShapeChanged(
            sourceType, targetType.getShape(), targetType.getDimsOrder(), distribution, Logger::global());
    auto axesMapping = mlir::succeeded(axesMappingOrFailure)
                               ? std::optional<std::pair<int64_t, int64_t>>(axesMappingOrFailure.value())
                               : std::nullopt;
    if (axesMapping.has_value() && axesMapping->first != -1 && axesMapping->second != -1) {
        return axesMapping;
    }

    const auto equalDimMapping = inferReshapeDistributedAxesMappingByEqualDimSize(sourceType, targetType, distribution);
    if (equalDimMapping.has_value() && (!axesMapping.has_value() || equalDimMapping.value() != axesMapping.value())) {
        return equalDimMapping;
    }

    return inferReshapeDistributedAxesMappingByTileBoundary(sourceType, targetType, distribution);
}

std::optional<mlir::Type> VPUIP::inferReshapeInputType(mlir::Operation* op, vpux::NDTypeInterface desiredOutputNDType,
                                                       vpux::NDTypeInterface origInputNDType,
                                                       vpux::NDTypeInterface origOutputNDType) {
    if (origInputNDType.getNumElements() != desiredOutputNDType.getNumElements()) {
        return std::nullopt;
    }

    const auto inputElemType =
            inferReshapeElementType(desiredOutputNDType.getElementType(), origOutputNDType.getElementType(),
                                    origInputNDType.getElementType(), origInputNDType.getShape());
    if (!inputElemType.has_value()) {
        return std::nullopt;
    }

    if (auto desiredOutputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(desiredOutputNDType)) {
        // For distributed outputs, move the target distribution axis back to the
        // original input shape before building the input buffer type. The final
        // compatibility checks ensure the reconstructed type has computable
        // per-cluster memory views and remains legal for a shape-change view.
        auto distribution = desiredOutputDistType.getDistribution();
        auto ctx = op->getContext();
        auto orderAttr = mlir::AffineMapAttr::get(origInputNDType.getDimsOrder().toAffineMap(ctx));

        auto buildInputType =
                [&](std::pair<int64_t, int64_t> axesMapping) -> std::optional<VPUIP::DistributedBufferType> {
            if (axesMapping.first == -1 || axesMapping.second == -1) {
                return std::nullopt;
            }

            auto newDistAttr = VPUIP::changeDistributedAxisOnDistributionInfoAttr(
                    distribution, axesMapping.first, axesMapping.second, origInputNDType.getShape());
            return VPUIP::createDistributedBufferTypeOrNull(ctx, origInputNDType.getShape(), inputElemType.value(),
                                                            orderAttr, desiredOutputDistType.getMemSpace(),
                                                            newDistAttr);
        };

        auto axesMapping = inferReshapeDistributedAxesMapping(desiredOutputNDType, origInputNDType, distribution);
        auto inType = axesMapping.has_value() ? buildInputType(axesMapping.value())
                                              : std::optional<VPUIP::DistributedBufferType>();
        if (!inType.has_value()) {
            return std::nullopt;
        }
        if (!VPUIP::isSupportedPerClusterMemoryShapesAndOffsets(inType.value()) ||
            !VPUIP::isSupportedPerClusterMemoryShapesAndOffsets(desiredOutputDistType)) {
            return std::nullopt;
        }
        if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                    inType.value(), desiredOutputDistType)) {
            return std::nullopt;
        }
        return mlir::cast<mlir::Type>(inType.value());
    }

    auto inType = desiredOutputNDType.changeShapeElemType(origInputNDType.getShape(), inputElemType.value());
    inType = inType.changeDimsOrder(origInputNDType.getDimsOrder());
    return mlir::cast<mlir::Type>(inType);
}
