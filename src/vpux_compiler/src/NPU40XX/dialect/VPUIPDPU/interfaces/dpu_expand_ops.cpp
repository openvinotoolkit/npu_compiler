//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux::VPUIPDPU;

namespace {

class DPUInvariantExpandOpInterfaceModel final :
        public VPUASM::DPUInvariantExpandOpInterface::ExternalModel<DPUInvariantExpandOpInterfaceModel,
                                                                    VPUASM::DPUInvariantOp> {
public:
    mlir::LogicalResult expandIDUConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap&) const {
        return arch40xx::buildDPUInvariantIDU(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos);
    }
    mlir::LogicalResult expandMPEConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger&,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap&) const {
        return arch40xx::buildDPUInvariantMPE(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, invBlock,
                                              invBlockArgsPos);
    }
    mlir::LogicalResult expandPPEConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap&) const {
        return arch40xx::buildDPUInvariantPPE(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos);
    }
    mlir::LogicalResult expandODUConfig(mlir::Operation* dpuInvariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        mlir::Block* invBlock,
                                        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                        ELF::SymbolReferenceMap& symRefMap) const {
        return arch40xx::buildDPUInvariantODU(mlir::cast<VPUASM::DPUInvariantOp>(dpuInvariantOp), builder, log,
                                              invBlock, invBlockArgsPos, symRefMap);
    }
};

class DPUVariantExpandOpInterfaceModel final :
        public VPUASM::DPUVariantExpandOpInterface::ExternalModel<DPUVariantExpandOpInterfaceModel,
                                                                  VPUASM::DPUVariantOp> {
public:
    mlir::LogicalResult expandGeneralConfig(mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, Logger log) const {
        return arch40xx::buildDPUVariantGeneral(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log);
    }
    mlir::LogicalResult expandIDUConfig(mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        ELF::SymbolReferenceMap& symRefMap) const {
        return arch40xx::buildDPUVariantIDU(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log, symRefMap);
    }

    mlir::LogicalResult expandPPEConfig(mlir::Operation*, mlir::OpBuilder&, const Logger&,
                                        ELF::SymbolReferenceMap&) const {
        return mlir::success();
    }
    mlir::LogicalResult expandODUConfig(mlir::Operation* dpuVariantOp, mlir::OpBuilder& builder, const Logger& log,
                                        mlir::Block* varBlock, ELF::SymbolReferenceMap& symRefMap) const {
        return arch40xx::buildDPUVariantODU(mlir::cast<VPUASM::DPUVariantOp>(dpuVariantOp), builder, log, varBlock,
                                            symRefMap);
    }
};

}  // namespace

void vpux::VPUIPDPU::arch40xx::registerDPUExpandOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIPDPU::VPUIPDPUDialect*) {
        VPUASM::DPUInvariantOp::attachInterface<DPUInvariantExpandOpInterfaceModel>(*ctx);
        VPUASM::DPUVariantOp::attachInterface<DPUVariantExpandOpInterfaceModel>(*ctx);
    });
}
