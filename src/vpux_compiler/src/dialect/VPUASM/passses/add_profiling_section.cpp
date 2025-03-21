//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUASM {
#define GEN_PASS_DECL_ADDPROFILINGSECTION
#define GEN_PASS_DEF_ADDPROFILINGSECTION
#include "vpux/compiler/dialect/VPUASM/passes.hpp.inc"
}  // namespace vpux::VPUASM

using namespace vpux;

namespace {
class AddProfilingSection : public VPUASM::impl::AddProfilingSectionBase<AddProfilingSection> {
public:
    explicit AddProfilingSection(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AddProfilingSection::safeRunOnModule() {
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, netFunc);
    const auto arch = VPU::getArch(moduleOp);
    if (netOp.getProfilingOutputsInfo().empty()) {
        return;
    }

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());

    auto profilingMetadataOps = to_small_vector(elfMain.getOps<VPUASM::ProfilingMetadataOp>());
    VPUX_THROW_UNLESS(profilingMetadataOps.size() == 1, "Expected exactly one ProfilingMetadataOp. Got {0}",
                      profilingMetadataOps.size());
    auto metadataOp = profilingMetadataOps[0];

    auto profilingSection = builder.create<ELF::CreateProfilingSectionOp>(elfMain.getLoc(), ".profiling", 1,
                                                                          ELF::SectionFlagsAttr::SHF_ALLOC);

    builder.setInsertionPointToEnd(&profilingSection.getContent().emplaceBlock());
    auto op = builder.create<VPUASM::ProfilingMetadataOp>(metadataOp.getLoc(), metadataOp.getSymNameAttr(),
                                                          metadataOp.getMetadataAttr());

    auto actualAlignment = builder.getIntegerAttr(builder.getIntegerType(64, false), op.getAlignmentRequirements(arch));
    profilingSection.setSecAddrAlignAttr(actualAlignment);
}
}  // namespace

//
// createAddProfilingSectionPass
//

std::unique_ptr<mlir::Pass> VPUASM::createAddProfilingSectionPass(Logger log) {
    return std::make_unique<AddProfilingSection>(log);
}
