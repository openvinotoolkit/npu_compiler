//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

using namespace vpux;

bool VPU::isSprLUTEnabled(mlir::Operation* op) {
    return VPU::getConstraint<bool>(op, VPU::SPRLUT_ENABLED);
}
