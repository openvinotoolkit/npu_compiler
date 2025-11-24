//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

using namespace vpux;
using namespace VPU;

mlir::RankedTensorType vpux::VPU::inferNCEMatmulOutputType(vpux::NDTypeInterface input1Type,
                                                           vpux::NDTypeInterface input2Type,
                                                           vpux::NDTypeInterface origOutputType) {
    const auto input1Shape = input1Type.getShape();
    const auto input2Shape = input2Type.getShape();
    SmallVector<int64_t> outputShape{input1Shape[Dim(0)], input1Shape[Dim(1)], input2Shape[Dim(1)], input1Shape[Dim(3)],
                                     input1Shape[Dim(4)]};

    return mlir::RankedTensorType::get(outputShape, origOutputType.getElementType(),
                                       VPU::createTensorAttrFromType(input1Type));
}
