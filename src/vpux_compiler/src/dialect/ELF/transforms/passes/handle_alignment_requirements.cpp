//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

namespace vpux::ELF {
#define GEN_PASS_DECL_HANDLEALIGNMENTREQUIREMENTS
#define GEN_PASS_DEF_HANDLEALIGNMENTREQUIREMENTS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {
constexpr auto VPUX_SHAVE_KERNEL_PREFETCH_PAD = static_cast<size_t>((128_Byte).count());

class HandleAlignmentRequirementsPass :
        public ELF::impl::HandleAlignmentRequirementsBase<HandleAlignmentRequirementsPass> {
public:
    explicit HandleAlignmentRequirementsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleAlignmentRequirementsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto& ctx = getContext();
    auto moduleOp = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "The top-level module is missing");
    const auto arch = config::getArch(moduleOp);

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
        const auto binarySizeOps = block->getOps<ELF::BinarySizeOpInterface>();
        if (binarySizeOps.empty()) {
            continue;
        }

        // Update the section alignment information based on the first binary op
        auto firstBodyOp = *binarySizeOps.begin();
        const auto alignmentRequirement = firstBodyOp.getAlignmentRequirements(arch);
        // Only enforce LCM alignment in case of sections that get allocated
        auto secAlignReq = ELF::bitEnumContainsAll(section.getSectionFlags(), ELF::SectionFlagsAttr::SHF_ALLOC)
                                   ? ELF::math::lcm(elf::VPU_SH_ADDR_ALIGN_FOR_VPU, alignmentRequirement)
                                   : alignmentRequirement;
        auto alignmentAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, secAlignReq, false));
        section.updateSectionAddressAlignment(alignmentAttr);

        auto requiresEndPadding =
                ELF::bitEnumContainsAll(section.getSectionFlags(), ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);

        // Add inner section padding ops
        auto builder = mlir::OpBuilder::atBlockEnd(block);
        size_t offsetTracker = 0;
        for (auto binarySizeOp : binarySizeOps) {
            const auto binarySizeOperation = binarySizeOp.getOperation();
            const auto alignmentRequirement = binarySizeOp.getAlignmentRequirements(arch);
            size_t paddingRequired = offsetTracker % alignmentRequirement;
            if (paddingRequired) {
                builder.setInsertionPoint(binarySizeOperation);
                auto paddingSize = alignmentRequirement - paddingRequired;
                builder.template create<ELF::PadOp>(builder.getUnknownLoc(), paddingSize, nullptr);
                offsetTracker += paddingSize;
            }
            offsetTracker += binarySizeOp.getBinarySizeCached(symRefMap, arch);
        }

        if (requiresEndPadding && section.getSectionType() != ELF::SectionTypeAttr::SHT_NOBITS) {
            builder.setInsertionPointToEnd(block);
            builder.template create<ELF::PadOp>(builder.getUnknownLoc(), VPUX_SHAVE_KERNEL_PREFETCH_PAD, nullptr);
            offsetTracker += VPUX_SHAVE_KERNEL_PREFETCH_PAD;
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
