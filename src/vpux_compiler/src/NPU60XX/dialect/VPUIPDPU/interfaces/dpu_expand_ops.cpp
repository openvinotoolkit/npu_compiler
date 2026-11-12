//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
using namespace vpux::VPUIPDPU;

namespace {

using namespace vpux;

class DPUInvariantExpandOpInterfaceModel final :
        public VPUASM::DPUInvariantExpandOpInterface::ExternalModel<DPUInvariantExpandOpInterfaceModel,
                                                                    VPUASM::DPUInvariantOp> {
public:
    mlir::LogicalResult expandIDUConfig(
            mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
            const std::unordered_map<BlockArg, size_t>& invBlockArgsPos, ELF::SymbolReferenceMap&,
            vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode /*npu5PPEBackwardsCompatibilityMode*/) const {
        return arch60xx::buildDPUInvariantIDU(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos);
    }

    mlir::LogicalResult expandMPEConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap&) const {
        return arch60xx::buildDPUInvariantMPE(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos);
    }

    mlir::LogicalResult expandPPEConfig(
            mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
            const std::unordered_map<BlockArg, size_t>& invBlockArgsPos, ELF::SymbolReferenceMap& symRefMap,
            vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode /*npu5PPEBackwardsCompatibilityMode*/) const {
        return arch60xx::buildDPUInvariantPPE(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos, symRefMap);
    }

    mlir::LogicalResult expandODUConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, Logger log,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap& symRefMap) const {
        return arch60xx::buildDPUInvariantODU(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos, symRefMap);
    }

    mlir::LogicalResult expandGeneralConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder) const {
        return arch60xx::buildDPUInvariantGeneral(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder);
    }
};

class DPUVariantExpandOpInterfaceModel final :
        public VPUASM::DPUVariantExpandOpInterface::ExternalModel<DPUVariantExpandOpInterfaceModel,
                                                                  VPUASM::DPUVariantOp> {
public:
    mlir::LogicalResult expandGeneralConfig(mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, Logger log) const {
        return arch60xx::buildDPUVariantGeneral(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log);
    }

    mlir::LogicalResult expandIDUConfig(
            mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, Logger log, ELF::SymbolReferenceMap& symRefMap,
            vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode /*npu5PPEBackwardsCompatibilityMode*/) const {
        return arch60xx::buildDPUVariantIDU(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log, symRefMap);
    }

    mlir::LogicalResult expandPPEConfig(
            mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, Logger, ELF::SymbolReferenceMap&,
            vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode /*npu5PPEBackwardsCompatibilityMode*/) const {
        return arch60xx::buildDPUVariantPPE(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder);
    }

    mlir::LogicalResult expandODUConfig(mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, Logger log,
                                        mlir::Block* varBlock, ELF::SymbolReferenceMap& symRefMap) const {
        return arch40xx::buildDPUVariantODU(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log, varBlock,
                                            symRefMap);
    }
};
}  // namespace

void vpux::VPUIPDPU::arch60xx::registerDPUExpandOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIPDPU::VPUIPDPUDialect*) {
        VPUASM::DPUInvariantOp::attachInterface<DPUInvariantExpandOpInterfaceModel>(*ctx);
        VPUASM::DPUVariantOp::attachInterface<DPUVariantExpandOpInterfaceModel>(*ctx);
    });
}
