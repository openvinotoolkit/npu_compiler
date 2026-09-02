//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/backend/export_utils.hpp"
#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDCOMPATIBILITYSTRING
#define GEN_PASS_DEF_ADDCOMPATIBILITYSTRING
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddCompatibilityStringPass : public ELF::impl::AddCompatibilityStringBase<AddCompatibilityStringPass> {
public:
    AddCompatibilityStringPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddCompatibilityStringPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    VPUX_THROW_UNLESS(llvm::hasSingleElement(funcOp.getOps<ELF::MainOp>()), "Expected exactly one ELF mainOp. Got {0}",
                      llvm::range_size(funcOp.getOps<ELF::MainOp>()));
    auto elfMain = *funcOp.getOps<ELF::MainOp>().begin();

    const auto elfVersion = config::getElfAbiVersion(elfMain);
    VPUX_THROW_WHEN(!elfVersion.has_value(), "ELF ABI version is required to be set for ELF export");

    const auto miVersion = backend::getMIVersionValue<ELF::DataSectionOp>(elfMain, [](ELF::DataSectionOp op) {
        return op.getContent().getOps();
    });
    const auto platformID = static_cast<uint64_t>(config::getPlatform(elfMain));
    const auto numOfTiles = config::getNumOfTiles(elfMain);

    const auto compatibilityString = backend::buildBlobCompatibilityString(backend::BlobCompatibilityInfo{
            platformID, numOfTiles, elf::Version{elfVersion->major, elfVersion->minor, elfVersion->patch}, miVersion});

    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto compatibilityStringOp =
            builder.create<ELF::CompatibilityStringOp>(elfMain.getLoc(), builder.getStringAttr(compatibilityString));
    ELF::moveOpToSection(compatibilityStringOp.getOperation(), builder);
}

}  // namespace

std::unique_ptr<mlir::Pass> ELF::createAddCompatibilityStringPass(Logger log) {
    return std::make_unique<AddCompatibilityStringPass>(log);
}
