// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/compiler_hash.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDCOMPILERHASH
#define GEN_PASS_DEF_ADDCOMPILERHASH
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {
//
// AddCompilerHashPass
//

class AddCompilerHashPass : public ELF::impl::AddCompilerHashBase<AddCompilerHashPass> {
public:
    AddCompilerHashPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddCompilerHashPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    mlir::OpBuilder builder(&(funcOp.getBody().front().back()));
    builder.create<ELF::CompilerHashOp>(builder.getUnknownLoc(), "CompilerHash",
                                        vpux::ELF::NPU_COMPILER_GIT_COMMIT_HASH);
}

}  // namespace

//
// createAddCompilerHashPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddCompilerHashPass(Logger log) {
    return std::make_unique<AddCompilerHashPass>(log);
}
