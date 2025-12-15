//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux::VPUIPDPU::arch50xx {

mlir::LogicalResult buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder, const Logger& log,
                                       ELF::SymbolReferenceMap& symRefMap);

mlir::LogicalResult buildDPUVariantPPE(
        VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder,
        vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode);

}  // namespace vpux::VPUIPDPU::arch50xx
