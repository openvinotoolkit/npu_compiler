//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/compiler_hash.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUASM {
#define GEN_PASS_DECL_ADDCOMPILERHASH
#define GEN_PASS_DEF_ADDCOMPILERHASH
#include "vpux/compiler/dialect/VPUASM/passes.hpp.inc"
}  // namespace vpux::VPUASM

using namespace vpux;

namespace {
//
// AddCompilerHashPass
//

class AddCompilerHashPass : public VPUASM::impl::AddCompilerHashBase<AddCompilerHashPass> {
public:
    AddCompilerHashPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddCompilerHashPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];
    std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface> sectionMap;
    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto compilerHashOp =
            builder.create<VPUASM::CompilerHashOp>(elfMain.getLoc(), builder.getStringAttr("CompilerHash"),
                                                   builder.getStringAttr(vpux::ELF::NPU_COMPILER_GIT_COMMIT_HASH));
    moveOpToSection(compilerHashOp.getOperation(), sectionMap, builder);
}

}  // namespace

//
// createAddCompilerHashPass
//

std::unique_ptr<mlir::Pass> VPUASM::createAddCompilerHashPass(Logger log) {
    return std::make_unique<AddCompilerHashPass>(log);
}
