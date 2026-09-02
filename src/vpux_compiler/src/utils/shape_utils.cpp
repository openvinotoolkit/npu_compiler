//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/shape_utils.hpp"

namespace vpux {

bool isNotDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape) {
    auto getNonOneDims = [](ShapeRef shape) {
        Shape resultShape;
        llvm::copy_if(shape, std::back_inserter(resultShape), [](int64_t elem) {
            return elem != 1;
        });
        return resultShape;
    };
    return getNonOneDims(origShape) != getNonOneDims(reshapeShape);
}

}  // namespace vpux
