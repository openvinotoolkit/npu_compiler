//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

namespace vpux::Core {
#define GEN_PASS_DECL_MOVEDECLARATIONSTOTOP
#define GEN_PASS_DEF_MOVEDECLARATIONSTOTOP
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

//
// MoveDeclarationsToTopPass
//

class MoveDeclarationsToTopPass final : public Core::impl::MoveDeclarationsToTopBase<MoveDeclarationsToTopPass> {
public:
    explicit MoveDeclarationsToTopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveDeclarationsToTopPass::safeRunOnFunc() {
    auto func = getOperation();
    VPUIP::moveDeclarationsToTop(func);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createMoveDeclarationsToTopPass(Logger log) {
    return std::make_unique<MoveDeclarationsToTopPass>(log);
}
