//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_expand_dma.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux::VPUIP::ExpandDMA {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef outShape, ShapeRef offset) {
    auto inShape = to_small_vector(outShape);
    // After Expand fuse into Permute and got one PermuteDMA Op
    // The channel size of input and output are not same
    // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
    // The channel size need align with the input
    inShape[Dims4D::Act::C.ind()] = originType.getShape()[Dims4D::Act::C];
    const auto elemType = originType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, ShapeRef(inShape), offset);
        return originType.changeShapeElemType(ShapeRef(inShape), newQType);
    }

    return originType.changeShape(ShapeRef(inShape));
}

}  // namespace vpux::VPUIP::ExpandDMA
