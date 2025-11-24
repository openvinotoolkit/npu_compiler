//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDELFSYMBOLTABLE
#define GEN_PASS_DEF_ADDELFSYMBOLTABLE
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddElfSymbolTablePass : public ELF::impl::AddELFSymbolTableBase<AddElfSymbolTablePass> {
public:
    explicit AddElfSymbolTablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddElfSymbolTablePass::safeRunOnFunc() {
    mlir::MLIRContext* ctx = &getContext();
    auto netFunc = getOperation();

    std::unordered_map<ELF::SectionSignature, ELF::CreateSymbolTableSectionOp> symTabMap;

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto sectionBuilder = mlir::OpBuilder::atBlockEnd(&elfMain.getContent().front());

    mlir::OpBuilder symbolBuilder(ctx);

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto symbolicallyRepresentedSection =
                mlir::dyn_cast<ELF::SymbolicallyRepresentedOpInterface>(section.getOperation());
        if (!symbolicallyRepresentedSection) {
            continue;
        }

        auto symTabSignature = section.getSymbolTableSectionSignature();
        ELF::CreateSymbolTableSectionOp symTab;

        if (auto symTabEntry = symTabMap.find(symTabSignature); symTabEntry != symTabMap.end()) {
            symTab = symTabEntry->second;
        } else {
            symTab = sectionBuilder.create<ELF::CreateSymbolTableSectionOp>(elfMain.getLoc(), symTabSignature.getName(),
                                                                            symTabSignature.getFlags());
            symTabMap[symTabSignature] = symTab;
        }

        auto symbolSignature = symbolicallyRepresentedSection.getSymbolSignature();

        symbolBuilder.setInsertionPointToEnd(symTab.getBlock());
        symbolBuilder.create<ELF::SymbolOp>(section.getLoc(), symbolSignature);
    }
}
}  // namespace

//
// createAddELFSymbolTablePass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddELFSymbolTablePass(Logger log) {
    return std::make_unique<AddElfSymbolTablePass>(log);
}
