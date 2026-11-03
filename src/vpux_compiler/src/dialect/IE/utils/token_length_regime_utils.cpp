//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/token_length_regime_utils.hpp"

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"

using namespace vpux;

std::optional<bool> IE::isDataBoundTokenDim(mlir::Operation* matmulOp) {
    if (matmulOp == nullptr || matmulOp->getNumResults() < 1) {
        return std::nullopt;
    }

    auto outShape = getShape(matmulOp->getResult(0));
    auto tokenDim = [&]() -> std::optional<vpux::Dim> {
        if (mlir::isa<IE::FullyConnectedOp>(matmulOp)) {
            return Dim(0);
        }

        if (mlir::isa<IE::MatMulOp>(matmulOp)) {
            const auto outputRank = mlir::cast<NDTypeInterface>(matmulOp->getResult(0).getType()).getRank();
            if (outputRank < 2) {
                return std::nullopt;
            }
            return Dim(outputRank - 2);
        }

        return std::nullopt;
    }();

    if (!tokenDim.has_value()) {
        return std::nullopt;
    }

    const auto leadingDim = outShape[tokenDim.value()];
    if (leadingDim == mlir::ShapedType::kDynamic) {
        return false;
    }
    return leadingDim <= DECODE_MAX_TOKEN_LENGTH;
}
