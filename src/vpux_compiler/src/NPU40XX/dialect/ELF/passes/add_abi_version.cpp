// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <cstdint>
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_ADDABIVERSION
#define GEN_PASS_DEF_ADDABIVERSION
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {
//
// AddABIVersionPass
//

class AddABIVersionPass : public ELF::arch40xx::impl::AddABIVersionBase<AddABIVersionPass> {
public:
    AddABIVersionPass(Logger log, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
            : _versionMajor(versionMajor), _versionMinor(versionMinor), _versionPatch(versionPatch) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    uint32_t _versionMajor;
    uint32_t _versionMinor;
    uint32_t _versionPatch;
};

void AddABIVersionPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    mlir::OpBuilder builder(&(funcOp.getBody().front().back()));
    builder.create<ELF::ABIVersionOp>(builder.getUnknownLoc(), _versionMajor, _versionMinor, _versionPatch);
}

}  // namespace

//
// createAddABIVersionPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddABIVersionPass(Logger log, uint32_t versionMajor, uint32_t versionMinor,
                                                               uint32_t versionPatch) {
    return std::make_unique<AddABIVersionPass>(log, versionMajor, versionMinor, versionPatch);
}
