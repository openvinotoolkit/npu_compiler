//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/compressed_convolution_utils.hpp"

using namespace vpux;

bool VPU::hasFP16CompressedConv(mlir::Operation* op) {
    return VPU::getConstraint<bool>(op, FP16_COMPRESSED_CONV);
}
