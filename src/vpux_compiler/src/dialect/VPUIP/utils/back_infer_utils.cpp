//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/back_infer_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"

namespace vpux {
namespace VPUIP {

namespace {

bool areTypesEquivalentForBackInfer(mlir::Type inferredType, mlir::Type requestedType) {
    if (inferredType == requestedType) {
        return true;
    }

    // DistributedBufferType equality can be stricter than the semantic facets
    // needed by back-inference because the same layout can be represented either
    // by an AffineMapAttr or a MemRefAttr. Recreated types are accepted only
    // when all DistributedBufferType parameters are equivalent: shape, element
    // type, layout, memory space, distribution and sparsity compression. For
    // layout, order/strides are compared together with MemRefAttr-only metadata
    // such as allocSize, bounds and HW-specific fields.
    auto inferredDistType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(inferredType);
    auto requestedDistType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(requestedType);
    if (inferredDistType == nullptr || requestedDistType == nullptr) {
        return false;
    }

    const auto hasSameLayout = [&]() {
        if (inferredDistType.getDimsOrder() != requestedDistType.getDimsOrder() ||
            inferredDistType.getStrides() != requestedDistType.getStrides()) {
            return false;
        }

        auto inferredMemRefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(inferredDistType.getLayout());
        auto requestedMemRefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(requestedDistType.getLayout());
        const auto hasMetadata = [](vpux::MemRefAttr memRefAttr) {
            return memRefAttr != nullptr && (memRefAttr.allocSize() != nullptr || !memRefAttr.bounds().empty() ||
                                             !memRefAttr.hwSpecificFields().empty());
        };

        if (inferredMemRefAttr == nullptr || requestedMemRefAttr == nullptr) {
            return !hasMetadata(inferredMemRefAttr) && !hasMetadata(requestedMemRefAttr);
        }

        return inferredMemRefAttr.allocSize() == requestedMemRefAttr.allocSize() &&
               inferredMemRefAttr.bounds().raw() == requestedMemRefAttr.bounds().raw() &&
               inferredMemRefAttr.hwSpecificFields() == requestedMemRefAttr.hwSpecificFields();
    };

    return inferredDistType.getShape() == requestedDistType.getShape() &&
           inferredDistType.getElementType() == requestedDistType.getElementType() && hasSameLayout() &&
           inferredDistType.getMemSpace() == requestedDistType.getMemSpace() &&
           inferredDistType.getDistribution() == requestedDistType.getDistribution() &&
           inferredDistType.getSparsityCompression() == requestedDistType.getSparsityCompression();
}

}  // namespace

std::optional<mlir::Type> BackInferUtils::inferOutputType(mlir::Operation* viewOp, mlir::Type newInputType) {
    if (viewOp == nullptr) {
        return std::nullopt;
    }

    auto backInferView = mlir::dyn_cast<VPUIP::BackInferViewTypeOpInterface>(viewOp);
    if (backInferView == nullptr) {
        return std::nullopt;
    }

    return backInferView.inferOutputTypeFromInput(newInputType);
}

std::optional<mlir::Type> BackInferUtils::reverseInferInputType(mlir::Operation* viewOp, mlir::Type desiredOutputType) {
    if (viewOp == nullptr) {
        return std::nullopt;
    }

    auto backInferView = mlir::dyn_cast<VPUIP::BackInferViewTypeOpInterface>(viewOp);
    if (backInferView == nullptr) {
        return std::nullopt;
    }

    const auto inputType = backInferView.inferInputTypeFromOutput(desiredOutputType);
    if (!inputType.has_value()) {
        return std::nullopt;
    }

    // Reverse inference is intentionally guarded by a forward round-trip.
    // Example: a ViewOp must not infer a CMX input for a requested DDR output
    // if running the ViewOp forward would still produce the original CMX result.
    const auto roundTripOutputType = backInferView.inferOutputTypeFromInput(inputType.value());
    if (!roundTripOutputType.has_value() ||
        !areTypesEquivalentForBackInfer(roundTripOutputType.value(), desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

}  // namespace VPUIP
}  // namespace vpux
