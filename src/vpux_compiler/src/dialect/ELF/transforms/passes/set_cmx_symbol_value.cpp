//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/utils/core/error.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace vpux::VPURegMapped {
#define GEN_PASS_DECL_SETCMXSYMBOLVALUE
#define GEN_PASS_DEF_SETCMXSYMBOLVALUE
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::VPURegMapped

using namespace vpux;

namespace {
class SetCMXSymbolValue : public VPURegMapped::impl::SetCMXSymbolValueBase<SetCMXSymbolValue> {
public:
    explicit SetCMXSymbolValue(Logger log, std::optional<uint32_t> workspaceAddr, std::optional<uint32_t> workspaceSize,
                               std::optional<uint32_t> metadataAddr, std::optional<uint32_t> metadataSize)
            : _log(log),
              _workspaceAddr(workspaceAddr),
              _workspaceSize(workspaceSize),
              _metadataAddr(metadataAddr),
              _metadataSize(metadataSize) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }

        if (workspaceAddr.hasValue()) {
            _workspaceAddr = workspaceAddr.getValue();
        }
        if (workspaceSize.hasValue()) {
            _workspaceSize = workspaceSize.getValue();
        }
        if (metadataAddr.hasValue()) {
            _metadataAddr = metadataAddr.getValue();
        }
        if (metadataSize.hasValue()) {
            _metadataSize = metadataSize.getValue();
        }

        return mlir::success();
    }

private:
    void safeRunOnModule() final;

    Logger _log;
    std::optional<uint32_t> _workspaceAddr;
    std::optional<uint32_t> _workspaceSize;

    std::optional<uint32_t> _metadataAddr;
    std::optional<uint32_t> _metadataSize;
};

void SetCMXSymbolValue::safeRunOnModule() {
    VPUX_THROW_UNLESS(_workspaceAddr.has_value() && _workspaceSize.has_value() && _metadataAddr.has_value() &&
                              _metadataSize.has_value(),
                      "Expected values are not present!");
    auto moduleOp = getOperation();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());

    auto elfMain = mainOps[0];
    ELF::SymbolReferenceMap symRefMap(elfMain);
    auto sTabOps = elfMain.getOps<ELF::CreateSymbolTableSectionOp>();

    for (auto symTab : sTabOps) {
        auto elfSymbols = symTab.getOps<ELF::SymbolOp>();
        for (auto elfSymbol : elfSymbols) {
            auto reference = symRefMap.lookupSymbol(elfSymbol.getReference());
            if (auto secInterface = mlir::dyn_cast<ELF::ElfSectionInterface>(reference)) {
                auto secType = secInterface.getSectionType();
                if (secType == ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA) {
                    elfSymbol.setValue(_metadataAddr.value());
                    elfSymbol.setSize(_metadataSize.value());
                } else if (secType == ELF::SectionTypeAttr::VPU_SHT_CMX_WORKSPACE) {
                    elfSymbol.setValue(_workspaceAddr.value());
                    elfSymbol.setSize(_workspaceSize.value());
                }
            }
        }
    }
}

}  // namespace

//
// createSetCMXSymbolValue
//

std::unique_ptr<mlir::Pass> ELF::createSetCMXSymbolValuePass(Logger log, std::optional<uint32_t> workspaceAddr,
                                                             std::optional<uint32_t> workspaceSize,
                                                             std::optional<uint32_t> metadataAddr,
                                                             std::optional<uint32_t> metadataSize) {
    return std::make_unique<SetCMXSymbolValue>(log, workspaceAddr, workspaceSize, metadataAddr, metadataSize);
}
