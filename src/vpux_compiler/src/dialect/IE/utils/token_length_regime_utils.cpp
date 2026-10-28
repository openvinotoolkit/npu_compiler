//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/token_length_regime_utils.hpp"

#include <mlir/IR/BuiltinTypeInterfaces.h>

using namespace vpux;

bool IE::isDataBoundTokenDim(vpux::ShapeRef outShape) {
    if (outShape.empty()) {
        return false;
    }
    const auto leadingDim = outShape[Dim(0)];
    if (leadingDim == mlir::ShapedType::kDynamic) {
        return false;
    }
    return leadingDim <= DECODE_MAX_TOKEN_LENGTH;
}
