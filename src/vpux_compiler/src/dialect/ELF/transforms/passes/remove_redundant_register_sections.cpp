//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Verifier.h>

namespace vpux::ELF {
#define GEN_PASS_DECL_REMOVEREDUNDANTREGISTERSECTIONS
#define GEN_PASS_DEF_REMOVEREDUNDANTREGISTERSECTIONS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

using SymbolUses = mlir::SmallVector<mlir::SymbolTable::SymbolUse>;

// Checks whether a SymbolRefAttr rooted at removedRootSymbol appears anywhere inside attr, at any nesting depth
// (mirrors what mlir::SymbolTable's internal walkSymbolRefs does via Attribute::walk).
bool referencesSymbol(mlir::Attribute attr, mlir::StringRef removedRootSymbol) {
    return attr
            .walk<mlir::WalkOrder::PreOrder>([&](mlir::SymbolRefAttr symbolRef) {
                return symbolRef.getRootReference().getValue() == removedRootSymbol ? mlir::WalkResult::interrupt()
                                                                                    : mlir::WalkResult::advance();
            })
            .wasInterrupted();
}

// Rebuilds attr with every element/entry referencing removedRootSymbol dropped; sets changed to true if anything
// was removed.
mlir::Attribute filterSymbolReferences(mlir::Attribute attr, mlir::StringRef removedRootSymbol, bool& changed) {
    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
        SmallVector<mlir::Attribute> filteredAttrs;
        for (auto element : arrayAttr) {
            if (referencesSymbol(element, removedRootSymbol)) {
                changed = true;
                continue;
            }
            filteredAttrs.push_back(filterSymbolReferences(element, removedRootSymbol, changed));
        }
        return mlir::ArrayAttr::get(attr.getContext(), filteredAttrs);
    }
    if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
        SmallVector<mlir::NamedAttribute> filteredAttrs;
        for (auto namedAttr : dictAttr) {
            if (referencesSymbol(namedAttr.getValue(), removedRootSymbol)) {
                changed = true;
                continue;
            }
            filteredAttrs.emplace_back(namedAttr.getName(),
                                       filterSymbolReferences(namedAttr.getValue(), removedRootSymbol, changed));
        }
        return mlir::DictionaryAttr::get(attr.getContext(), filteredAttrs);
    }
    return attr;
}

void updateSymbolReferences(mlir::Operation* op, mlir::StringRef removedRootSymbol) {
    // example of attributes to update:
    // output_buffs = [@reg.Register.0::@DeclareBuffer_40] becomes output_buffs = []
    // input = @reg.Register.0::@DeclareBuffer_40 (scalar reference) gets dropped entirely
    SmallVector<mlir::NamedAttribute> attrsToUpdate;
    SmallVector<mlir::StringAttr> attrsToRemove;
    for (auto& namedAttr : op->getAttrs()) {
        if (!referencesSymbol(namedAttr.getValue(), removedRootSymbol)) {
            continue;
        }
        // a bare SymbolRefAttr can't be "filtered", so the whole attribute must be dropped
        if (mlir::isa<mlir::SymbolRefAttr>(namedAttr.getValue())) {
            attrsToRemove.push_back(namedAttr.getName());
            continue;
        }
        bool changed = false;
        auto updatedValue = filterSymbolReferences(namedAttr.getValue(), removedRootSymbol, changed);
        if (changed) {
            attrsToUpdate.emplace_back(namedAttr.getName(), updatedValue);
        }
    }

    for (const auto& updatedAttr : attrsToUpdate) {
        op->setAttr(updatedAttr.getName(), updatedAttr.getValue());
    }
    for (auto attrName : attrsToRemove) {
        op->removeAttr(attrName);
        VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(op)),
                          "Removing attribute '{0}' makes DMA operation '{1}' invalid; only optional attributes "
                          "may be removed",
                          attrName, op->getName());
    }
}

bool isDmaOp(mlir::Operation* op) {
    return op->hasTrait<vpux::DMAOpTrait>();
}

bool hasNonDmaUser(llvm::ArrayRef<mlir::SymbolTable::SymbolUse> users) {
    return llvm::find_if(users, [](auto user) {
               auto* userOp = user.getUser();
               return !mlir::isa<ELF::SymbolOp>(userOp) && !isDmaOp(userOp);
           }) != users.end();
}

void cleanupRegisterSection(mlir::Operation* symbol, llvm::ArrayRef<mlir::SymbolTable::SymbolUse> users,
                            SmallVector<mlir::Operation*, 16>& toErase) {
    auto symbolInterface = mlir::dyn_cast<mlir::SymbolOpInterface>(symbol);
    if (!symbolInterface) {
        return;
    }

    auto symName = symbolInterface.getName();
    for (auto user : users) {
        auto* userOp = user.getUser();
        if (mlir::isa<ELF::SymbolOp>(userOp)) {
            // remove symbol from symbol table (example ELF.Symbol @elfsym.reg.Register.0)
            toErase.push_back(userOp);
            userOp->remove();
        } else if (isDmaOp(userOp)) {
            updateSymbolReferences(userOp, symName);
        }
    }

    // elf section to remove (example: ELF.CreateLogicalSection @reg.Register.0)
    toErase.push_back(symbol);
    symbol->remove();
}

bool isRegisterBufferSection(ELF::LogicalSectionOp section) {
    return llvm::any_of(
            section.getBody()->getOps<VPUASM::DeclareBufferOp>(), [](VPUASM::DeclareBufferOp declareBuffer) {
                return declareBuffer.getBufferType().getLocation().getSection() == VPURT::BufferSection::Register;
            });
}

struct SectionToErase {
    ELF::LogicalSectionOp section;
    SymbolUses users;
};

class RemoveRedundantRegisterSectionsPass final :
        public ELF::impl::RemoveRedundantRegisterSectionsBase<RemoveRedundantRegisterSectionsPass> {
public:
    explicit RemoveRedundantRegisterSectionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveRedundantRegisterSectionsPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain);

    llvm::DenseSet<mlir::Operation*> relocationSources;
    for (auto relocSection : elfMain.getOps<ELF::CreateRelocationSectionOp>()) {
        for (auto reloc : relocSection.getOps<ELF::RelocOp>()) {
            auto sourceSymbolOp =
                    mlir::dyn_cast_or_null<ELF::SymbolOp>(symRefMap.lookupSymbol(reloc.getSourceSymbol()));
            if (sourceSymbolOp == nullptr) {
                continue;
            }

            auto referencedOp = symRefMap.lookupSymbol(sourceSymbolOp.getReference());
            if (referencedOp != nullptr) {
                relocationSources.insert(referencedOp);
            }
        }
    }

    SmallVector<SectionToErase> sectionsToErase;
    for (auto section : elfMain.getOps<ELF::LogicalSectionOp>()) {
        if (!isRegisterBufferSection(section)) {
            continue;
        }

        if (relocationSources.contains(section.getOperation())) {
            continue;
        }

        auto users = vpux::ELF::getSymbolUses(section.getOperation(), elfMain);
        if (hasNonDmaUser(users)) {
            continue;
        }

        sectionsToErase.push_back({section, std::move(users)});
    }

    SmallVector<mlir::Operation*, 16> toErase;
    for (auto& sectionToErase : sectionsToErase) {
        cleanupRegisterSection(sectionToErase.section.getOperation(), sectionToErase.users, toErase);
    }

    for (auto op : toErase) {
        op->destroy();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createRemoveRedundantRegisterSectionsPass(Logger log) {
    return std::make_unique<RemoveRedundantRegisterSectionsPass>(log);
}
