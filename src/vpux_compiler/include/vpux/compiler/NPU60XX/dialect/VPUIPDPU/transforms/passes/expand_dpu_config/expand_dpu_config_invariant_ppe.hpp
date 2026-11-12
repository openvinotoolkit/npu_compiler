//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_ppe.hpp"

#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

namespace vpux::VPUIPDPU::arch60xx::PPE {

mlir::LogicalResult configurePPE(arch50xx::PPE::PPEConfig& config, mlir::Type outDataType,
                                 VPUIP::NCETaskType dpuTaskType, const arch50xx::PPE::PPETask& ppeTask,
                                 bool isWeightTableProvided, bool isSprLookUpTableProvided);

}  // namespace vpux::VPUIPDPU::arch60xx::PPE
