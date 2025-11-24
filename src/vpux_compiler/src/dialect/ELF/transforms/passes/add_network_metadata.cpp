//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDNETWORKMETADATA
#define GEN_PASS_DEF_ADDNETWORKMETADATA
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {
class AddNetworkMetadata : public ELF::impl::AddNetworkMetadataBase<AddNetworkMetadata> {
public:
    explicit AddNetworkMetadata(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddNetworkMetadata::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto metadataSection = builder.create<ELF::CreateMetadataSectionOp>(elfMain.getLoc(), "MetadataSection", 1,
                                                                        ELF::SectionFlagsAttr::SHF_NONE);
    // add network metadata OPs

    builder.setInsertionPointToEnd(&metadataSection.getContent().emplaceBlock());
    auto metadataOp = builder.create<VPUASM::NetworkMetadataOp>(netFunc.getLoc(), "NetworkMetadata");

    auto actualAlignment = builder.getIntegerAttr(builder.getIntegerType(64, false),
                                                  metadataOp.getAlignmentRequirements(config::getArch(netFunc)));
    metadataSection.setSecAddrAlignAttr(actualAlignment);
}
}  // namespace

//
// createSetEntryPointPass
//

std::unique_ptr<mlir::Pass> ELF::createAddNetworkMetadataPass(Logger log) {
    return std::make_unique<AddNetworkMetadata>(log);
}
