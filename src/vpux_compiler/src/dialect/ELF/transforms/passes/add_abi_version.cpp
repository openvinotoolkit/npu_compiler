//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include <cstdint>

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDABIVERSION
#define GEN_PASS_DEF_ADDABIVERSION
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddABIVersionPass : public ELF::impl::AddABIVersionBase<AddABIVersionPass> {
public:
    AddABIVersionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto netFunc = getOperation();

        auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
        VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
        auto elfMain = mainOps[0];

        auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
        auto elfVersion = ELF::ABIVersionOp::create(builder, builder.getUnknownLoc());
        ELF::moveOpToSection(elfVersion, builder);
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createAddABIVersionPass(Logger log) {
    return std::make_unique<AddABIVersionPass>(log);
}
