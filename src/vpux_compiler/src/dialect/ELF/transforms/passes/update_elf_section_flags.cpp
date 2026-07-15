//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include <llvm/ADT/StringRef.h>
#include "vpux/compiler/dialect/ELF/IR/attributes.hpp"
#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_UPDATEELFSECTIONFLAGS
#define GEN_PASS_DEF_UPDATEELFSECTIONFLAGS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class UpdateELFSectionFlagsPass final : public ELF::impl::UpdateELFSectionFlagsBase<UpdateELFSectionFlagsPass> {
public:
    explicit UpdateELFSectionFlagsPass(Logger log, bool isShaveDDRAccessEnabled) {
        Base::initLogger(log, Base::getArgumentName());
        enableShaveDDRAccess = isShaveDDRAccessEnabled;
    }

private:
    void registerDDRSections(ELF::MainOp& elfMain,
                             llvm::SmallDenseMap<llvm::StringRef, ELF::ElfSectionInterface>& sectionMap,
                             std::vector<ELF::DDRMemoryAccessingOpInterface>& ddrAccessingOps) {
        for (auto sectionOp : elfMain.getOps<ELF::ElfSectionInterface>()) {
            auto secFlags = sectionOp.getSectionFlags();

            // only register Sections that will be allocated
            if (!ELF::bitEnumContainsAll(secFlags, ELF::SectionFlagsAttr::SHF_ALLOC)) {
                continue;
            }
            sectionMap.insert(std::make_pair(sectionOp.getSectionName(), sectionOp));
            for (auto wrappableOp : sectionOp.getBlock()->getOps<ELF::WrappableOpInterface>()) {
                // Some ops inherently determine memory access to themselves by some processor
                // If op fits in this typology, gather its flags
                if (auto knownPurposeMemOp =
                            mlir::dyn_cast<ELF::PredefinedPurposeMemoryOpInterface>(wrappableOp.getOperation())) {
                    secFlags = secFlags | knownPurposeMemOp.getPredefinedMemoryAccessors();
                }

                // Collect all ops that determine access to some other DDR data (maybe as I/O)
                if (auto ddrAccessingOp =
                            mlir::dyn_cast<ELF::DDRMemoryAccessingOpInterface>(wrappableOp.getOperation())) {
                    ddrAccessingOps.push_back(ddrAccessingOp);
                }
            }
            sectionOp.updateSectionFlags(secFlags);
        }
    }

    void safeRunOnModule() final {
        mlir::ModuleOp moduleOp = getOperation();

        auto funcOp = net::getMainFunc(moduleOp);

        auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
        VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
        auto elfMain = mainOps[0];

        const auto sufficientAccessFlags = enableShaveDDRAccess ? ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE |
                                                                          ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA
                                                                : ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;

        llvm::SmallDenseMap<llvm::StringRef, ELF::ElfSectionInterface> sectionMap;
        std::vector<ELF::DDRMemoryAccessingOpInterface> ddrAccessingOps;

        registerDDRSections(elfMain, sectionMap, ddrAccessingOps);

        for (auto op : ddrAccessingOps) {
            // Iterate through the sections that the op determines access to and update their flags
            for (auto accessedSection : op.getAccessedSections()) {
                auto sectionOpIt = sectionMap.find(accessedSection.getValue());
                if (sectionOpIt != sectionMap.end()) {
                    auto memAccessFlags = op.getMemoryAccessingProcForSection(accessedSection);
                    auto updatedFlags = sectionOpIt->getSecond().updateSectionFlags(memAccessFlags);
                    // Early-exit condition for the specific section if all needed flags have already been set
                    if (ELF::bitEnumContainsAll(updatedFlags, sufficientAccessFlags)) {
                        sectionMap.erase(sectionOpIt);
                    }
                }
            }
        }
    };
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createUpdateELFSectionFlagsPass(Logger log, bool enableShaveDDRAccess) {
    return std::make_unique<UpdateELFSectionFlagsPass>(log, enableShaveDDRAccess);
}
