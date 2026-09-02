//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUASM {
#define GEN_PASS_DECL_ADDHOSTPARSEDINFERENCE
#define GEN_PASS_DEF_ADDHOSTPARSEDINFERENCE
#include "vpux/compiler/dialect/VPUASM/passes.hpp.inc"
}  // namespace vpux::VPUASM

using namespace vpux;

namespace {

//
// AddHostParsedInferencePass
//

class AddHostParsedInferencePass : public VPUASM::impl::AddHostParsedInferenceBase<AddHostParsedInferencePass> {
public:
    explicit AddHostParsedInferencePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddHostParsedInferencePass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();

    VPUX_THROW_UNLESS(llvm::hasSingleElement(funcOp.getOps<ELF::MainOp>()), "Expected exactly one ELF mainOp. Got {0}",
                      llvm::range_size(funcOp.getOps<ELF::MainOp>()));
    auto elfMain = *funcOp.getOps<ELF::MainOp>().begin();

    // At this stage the ManagedMappedInference is already wrapped in its ELF data section, so look it
    // up inside the sections (a shallow getOps on ELF.Main would not reach it).
    VPUASM::ManagedMappedInferenceOp managedOp;
    ELF::DataSectionOp managedSection;
    for (auto dataSection : elfMain.getOps<ELF::DataSectionOp>()) {
        auto managedOps = dataSection.getBlock()->getOps<VPUASM::ManagedMappedInferenceOp>();
        if (managedOps.empty()) {
            continue;
        }
        VPUX_THROW_UNLESS(llvm::hasSingleElement(managedOps), "Expected exactly one ManagedMappedInferenceOp, got {0}",
                          llvm::range_size(managedOps));
        managedOp = *managedOps.begin();
        managedSection = dataSection;
        break;
    }
    VPUX_THROW_UNLESS(managedOp != nullptr, "Could not find ManagedMappedInferenceOp");
    VPUX_THROW_UNLESS(managedSection != nullptr, "Could not find DataSectionOp for ManagedMappedInferenceOp");

    auto mappedRef = mlir::SymbolRefAttr::get(managedSection.getSymNameAttr(),
                                              {mlir::FlatSymbolRefAttr::get(managedOp.getSymNameAttr())});

    const auto nnSliceCount = VPUIP::getNumTilesUsed(moduleOp);
    const auto nnSliceLength =
            checked_cast<int64_t>(config::getAvailableMemory(moduleOp, vpux::VPU::MemoryKind::CMX_NN).getByteSize());
    // nn_barriers_ is unused by the runtime.
    const auto nnBarriers = 0;

    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto hpiOp = builder.create<VPUASM::HostParsedInferenceOp>(
            elfMain.getLoc(), builder.getStringAttr("HostParsedInference"), mappedRef,
            builder.getI64IntegerAttr(nnSliceLength), builder.getI64IntegerAttr(nnSliceCount),
            builder.getI64IntegerAttr(nnBarriers));

    ELF::moveOpToSection(hpiOp.getOperation(), builder);
}

}  // namespace

//
// createAddHostParsedInferencePass
//

std::unique_ptr<mlir::Pass> VPUASM::createAddHostParsedInferencePass(Logger log) {
    return std::make_unique<AddHostParsedInferencePass>(log);
}
