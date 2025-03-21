//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_NEXTSAMEIDASSIGNMENT
#define GEN_PASS_DEF_NEXTSAMEIDASSIGNMENT
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;
using namespace npu40xx;

namespace {
// TODO: E111344
class NextSameIdAssignmentPass : public VPUMI40XX::impl::NextSameIdAssignmentBase<NextSameIdAssignmentPass> {
public:
    explicit NextSameIdAssignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void NextSameIdAssignmentPass::safeRunOnFunc() {
    vpux::VPUMI40XX::setBarrierIDs(&(getContext()), getOperation());
}

}  // namespace

//
// createNextSameIdAssignmentPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createNextSameIdAssignmentPass(Logger log) {
    return std::make_unique<NextSameIdAssignmentPass>(log);
}
