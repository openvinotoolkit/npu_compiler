//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_SETENTRYPOINT
#define GEN_PASS_DEF_SETENTRYPOINT
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {
class SetEntryPointPass : public ELF::arch40xx::impl::SetEntryPointBase<SetEntryPointPass> {
public:
    explicit SetEntryPointPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SetEntryPointPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto getMPI = [](ELF::MainOp main) -> mlir::SymbolRefAttr {
        for (auto dataSection : main.getOps<ELF::DataSectionOp>()) {
            auto mpiOps = dataSection.getBlock()->getOps<VPUASM::MappedInferenceOp>();
            if (mpiOps.empty()) {
                continue;
            }
            const auto mpiCount = std::distance(mpiOps.begin(), mpiOps.end());
            VPUX_THROW_UNLESS(mpiCount == 1, "Expected single MappedInferenceOp, found {0}", mpiCount);
            auto mpi = *mpiOps.begin();
            auto mpiRef = mlir::FlatSymbolRefAttr::get(mpi.getNameAttr());
            return mlir::SymbolRefAttr::get(dataSection.getNameAttr(), {mpiRef});
        }
        VPUX_THROW("Could not find MappedInferenceOp");
        return nullptr;
    };

    auto getSymTab = [](ELF::MainOp main) -> ELF::CreateSymbolTableSectionOp {
        for (auto symTab : main.getOps<ELF::CreateSymbolTableSectionOp>()) {
            if (symTab.getName() == "symtab")
                return symTab;
        }
        VPUX_THROW("Coult not find default symtab");
        return nullptr;
    };

    // should have a better way to get the MPI
    mlir::SymbolRefAttr mpiRef = getMPI(elfMain);
    ELF::CreateSymbolTableSectionOp symTab = getSymTab(elfMain);

    auto builder = mlir::OpBuilder::atBlockEnd(symTab.getBlock());

    builder.create<ELF::SymbolOp>(elfMain.getLoc(),                 // location
                                  "entry",                          // sym_name
                                  mpiRef,                           // reference
                                  ELF::SymbolType::VPU_STT_ENTRY);  // type
}
}  // namespace

//
// createSetEntryPointPass
//

std::unique_ptr<mlir::Pass> ELF::createSetEntryPointPass(Logger log) {
    return std::make_unique<SetEntryPointPass>(log);
}
