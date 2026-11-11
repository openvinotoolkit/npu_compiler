//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_general.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch60xx {

mlir::LogicalResult buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder, const Logger& log,
                                       ELF::SymbolReferenceMap& symRefMap, bool dpuSwizzlingEnabled = true);

mlir::LogicalResult buildDPUVariantPPE(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder);

mlir::LogicalResult buildDPUVariantGeneral(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder, const Logger& log);

}  // namespace vpux::VPUIPDPU::arch60xx
