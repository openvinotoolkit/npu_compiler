//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/reloc_manager.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDELFRELOCATIONS
#define GEN_PASS_DEF_ADDELFRELOCATIONS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddELFRelocationsPass : public ELF::impl::AddELFRelocationsBase<AddELFRelocationsPass> {
public:
    explicit AddELFRelocationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddELFRelocationsPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::RelocManager relocManager(elfMain);

    for (auto sectionInterface : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto targetSection = sectionInterface.getOperation();

        // TODO:  E#59169 consider to add interface dedicated for sections that are intended to hold program data, aka
        // should be relocateable?
        if (!mlir::isa<ELF::DataSectionOp, ELF::LogicalSectionOp>(targetSection)) {
            continue;
        }

        auto block = sectionInterface.getBlock();

        for (auto relocatableOp : block->getOps<ELF::RelocatableOpInterface>()) {
            relocManager.createRelocations(relocatableOp);
        }
    }
}

}  // namespace

//
// createRemoveEmptyELFSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddELFRelocationsPass(Logger log) {
    return std::make_unique<AddELFRelocationsPass>(log);
}
