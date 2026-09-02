//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
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
    explicit SetCMXSymbolValue(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void SetCMXSymbolValue::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto netFunc = net::getMainFunc(moduleOp);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());

    auto elfMain = mainOps[0];
    ELF::SymbolReferenceMap symRefMap(elfMain);
    auto sTabOps = elfMain.getOps<ELF::CreateSymbolTableSectionOp>();

    static constexpr uint32_t workspaceAddr = CMX_BASE_ADDR + 0x00200000;  // CMX0 base address

    const auto cmxSpaceAttr = mlir::SymbolRefAttr::get(moduleOp.getContext(), "CMX_NN");
    const auto availableCMXMemory = config::getAvailableMemory(moduleOp, cmxSpaceAttr).size();
    const auto workspaceSize = static_cast<uint32_t>(availableCMXMemory.count());

    auto metadataMem = config::getCMXMetadataReservedMemory(moduleOp);
    VPUX_THROW_UNLESS(metadataMem, "Missing reserved CMX memory for metadata");
    const auto metadataSize = static_cast<uint32_t>(metadataMem.getByteSize());

    auto metadataMemOffset = metadataMem.getOffset();
    VPUX_THROW_UNLESS(metadataMemOffset.has_value(), "No address allocated for metadata in CMX");
    const auto metadataAddr = workspaceAddr + static_cast<uint32_t>(metadataMemOffset.value());

    for (auto symTab : sTabOps) {
        auto elfSymbols = symTab.getOps<ELF::SymbolOp>();
        for (auto elfSymbol : elfSymbols) {
            auto reference = symRefMap.lookupSymbol(elfSymbol.getReference());
            if (auto secInterface = mlir::dyn_cast<ELF::ElfSectionInterface>(reference)) {
                auto secType = secInterface.getSectionType();
                if (secType == ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA) {
                    elfSymbol.setValue(metadataAddr);
                    elfSymbol.setSize(metadataSize);
                } else if (secType == ELF::SectionTypeAttr::VPU_SHT_CMX_WORKSPACE) {
                    elfSymbol.setValue(workspaceAddr);
                    elfSymbol.setSize(workspaceSize);
                }
            }
        }
    }
}

}  // namespace

//
// createSetCMXSymbolValue
//

std::unique_ptr<mlir::Pass> ELF::createSetCMXSymbolValuePass(Logger log) {
    return std::make_unique<SetCMXSymbolValue>(log);
}
