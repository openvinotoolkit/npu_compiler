//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_REMOVEEMPTYELFSECTIONS
#define GEN_PASS_DEF_REMOVEEMPTYELFSECTIONS
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {

// An ELF Section OP is basically a container for other OPS. An empty one represents no real usage.
// Furthermore, by current ELF design, symbolic relationships are direct usage-based.
// AKA: if a section can be removed, all it's users are safe to remove (aka a section's symbol or relocation sections
// that target the particular section)

void recursivelyRemove(mlir::Operation* symbol, ELF::MainOp main, llvm::SmallVector<mlir::Operation*, 16>& toErase) {
    if (mlir::isa<mlir::SymbolOpInterface>(symbol)) {
        auto users = mlir::SymbolTable::getSymbolUses(symbol, main);
        if (users.has_value()) {
            for (auto user : llvm::make_early_inc_range(users.value())) {
                recursivelyRemove(user.getUser(), main, toErase);
            }
        }
    } else {
        // it may not be the case, but for safety erase also all SSA users
        auto users = symbol->getUsers();
        for (auto user : llvm::make_early_inc_range(users)) {
            recursivelyRemove(user, main, toErase);
        }
    }

    toErase.push_back(symbol);
    symbol->remove();
}

class RemoveEmptyELFSectionsPass : public ELF::arch40xx::impl::RemoveEmptyELFSectionsBase<RemoveEmptyELFSectionsPass> {
public:
    explicit RemoveEmptyELFSectionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveEmptyELFSectionsPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto sections = elfMain.getOps<ELF::ElfSectionInterface>();

    llvm::SmallVector<mlir::Operation*, 16> toErase;

    for (auto section : llvm::make_early_inc_range(sections)) {
        if (section.getBlock()->empty()) {
            auto operation = section.getOperation();
            recursivelyRemove(operation, elfMain, toErase);
        };
    }

    for (auto op : toErase) {
        op->destroy();
    }
}

}  // namespace

//
// createRemoveEmptyELFSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createRemoveEmptyELFSectionsPass(Logger log) {
    return std::make_unique<RemoveEmptyELFSectionsPass>(log);
}
