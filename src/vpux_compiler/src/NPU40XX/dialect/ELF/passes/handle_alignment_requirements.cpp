//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_HANDLEALIGNMENTREQUIREMENTS
#define GEN_PASS_DEF_HANDLEALIGNMENTREQUIREMENTS
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {

class HandleAlignmentRequirementsPass :
        public ELF::arch40xx::impl::HandleAlignmentRequirementsBase<HandleAlignmentRequirementsPass> {
public:
    explicit HandleAlignmentRequirementsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    Logger _log;
};

void HandleAlignmentRequirementsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto& ctx = getContext();
    auto moduleOp = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "The top-level module is missing");
    const auto arch = VPU::getArch(moduleOp);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);
    auto u64Type = vpux::getUInt64Type(&ctx);

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto block = section.getBlock();
        if (block->empty()) {
            continue;
        }
        const auto binaryOps = block->getOps<ELF::BinaryOpInterface>();
        if (binaryOps.empty()) {
            continue;
        }

        // Update the section alignment information based on the first binary op
        auto firstBodyOp = *binaryOps.begin();
        const auto alignmentRequirement = firstBodyOp.getAlignmentRequirements(arch);
        // Only enforce LCM alignment in case of sections that get allocated
        auto secAlignReq = ELF::bitEnumContainsAll(section.getSectionFlags(), ELF::SectionFlagsAttr::SHF_ALLOC)
                                   ? ELF::math::lcm(elf::VPU_SH_ADDR_ALIGN_FOR_VPU, alignmentRequirement)
                                   : alignmentRequirement;
        auto alignmentAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, secAlignReq, false));
        section.updateSectionAddressAlignment(alignmentAttr);

        // Add inner section padding ops
        auto builder = mlir::OpBuilder::atBlockEnd(block);
        size_t offsetTracker = 0;
        for (auto binaryOp : binaryOps) {
            const auto binaryOperation = binaryOp.getOperation();
            const auto alignmentRequirement = binaryOp.getAlignmentRequirements(arch);
            size_t paddingRequired = offsetTracker % alignmentRequirement;
            if (paddingRequired) {
                builder.setInsertionPoint(binaryOperation);
                auto paddingSize = alignmentRequirement - paddingRequired;
                builder.template create<ELF::PadOp>(builder.getUnknownLoc(), paddingSize, nullptr);
                offsetTracker += paddingSize;
            }
            offsetTracker += binaryOp.getBinarySizeCached(symRefMap, arch);
        }
    }
}
}  // namespace

//
// createHandleAlignmentRequirements
//

std::unique_ptr<mlir::Pass> vpux::ELF::createHandleAlignmentRequirementsPass(Logger log) {
    return std::make_unique<HandleAlignmentRequirementsPass>(log);
}
