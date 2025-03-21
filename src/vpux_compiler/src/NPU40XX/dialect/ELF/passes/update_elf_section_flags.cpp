//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_UPDATEELFSECTIONFLAGS
#define GEN_PASS_DEF_UPDATEELFSECTIONFLAGS
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {

class UpdateELFSectionFlagsPass final :
        public ELF::arch40xx::impl::UpdateELFSectionFlagsBase<UpdateELFSectionFlagsPass> {
public:
    explicit UpdateELFSectionFlagsPass(Logger log, std::string isShaveDDRAccessEnabled): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
        _isShaveDDRAccessEnabled = isShaveDDRAccessEnabled == "true";
        _sufficientAccessFlags = _isShaveDDRAccessEnabled ? ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE |
                                                                    ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA
                                                          : ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
    }

private:
    llvm::SmallDenseMap<llvm::StringRef, ELF::ElfSectionInterface> sectionMap;
    std::vector<ELF::DDRMemoryAccessingOpInterface> ddrAccessingOps;
    bool _isShaveDDRAccessEnabled;
    ELF::SectionFlagsAttr _sufficientAccessFlags;
    Logger _log;

private:
    void registerDDRSections(ELF::MainOp& elfMain) {
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
                    // If shave DDR access is disabled, ignore the ops that determine SHAVE access
                    if (!_isShaveDDRAccessEnabled &&
                        ddrAccessingOp.getMemoryAccessingProc() == ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE) {
                        continue;
                    }
                    ddrAccessingOps.push_back(ddrAccessingOp);
                }
            }
            sectionOp.updateSectionFlags(secFlags);
        }
    }

    void safeRunOnModule() final {
        mlir::ModuleOp moduleOp = getOperation();

        IE::CNNNetworkOp cnnOp;
        mlir::func::FuncOp funcOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

        auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
        VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
        auto elfMain = mainOps[0];

        registerDDRSections(elfMain);

        for (auto op : ddrAccessingOps) {
            auto memAccessFlags = op.getMemoryAccessingProc();
            // Iterate through the sections that the op determines access to and update their flags
            for (auto accessedSection : op.getAccessedSections()) {
                auto sectionOpIt = sectionMap.find(accessedSection.getValue());
                if (sectionOpIt != sectionMap.end()) {
                    auto updatedFlags = sectionOpIt->getSecond().updateSectionFlags(memAccessFlags);
                    // Early-exit condition for the specific section if all needed flags have already been set
                    if (ELF::bitEnumContainsAll(updatedFlags, _sufficientAccessFlags)) {
                        sectionMap.erase(sectionOpIt);
                    }
                }
            }
        }
    };
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createUpdateELFSectionFlagsPass(Logger log,
                                                                       std::string isShaveDDRAccessEnabled) {
    return std::make_unique<UpdateELFSectionFlagsPass>(log, isShaveDDRAccessEnabled);
}
