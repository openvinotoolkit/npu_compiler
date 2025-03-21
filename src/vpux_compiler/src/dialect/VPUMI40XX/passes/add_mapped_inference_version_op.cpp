//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDMAPPEDINFERENCEVERSIONOP
#define GEN_PASS_DEF_ADDMAPPEDINFERENCEVERSIONOP
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX
namespace {
// TODO: E111344
class AddMappedInferenceVersionOpPass :
        public VPUMI40XX::impl::AddMappedInferenceVersionOpBase<AddMappedInferenceVersionOpPass> {
public:
    AddMappedInferenceVersionOpPass(Logger log, uint32_t versionMajor, uint32_t versionMinor, uint32_t versionPatch)
            : _versionMajor(versionMajor), _versionMinor(versionMinor), _versionPatch(versionPatch) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    uint32_t _versionMajor;
    uint32_t _versionMinor;
    uint32_t _versionPatch;
};

void AddMappedInferenceVersionOpPass::safeRunOnFunc() {
    auto ctx = &(getContext());
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    auto indexType = VPURegMapped::IndexType::get(ctx, 0);
    auto mappedInferenceVersion = builder.create<VPUMI40XX::MappedInferenceVersionOp>(
            builder.getUnknownLoc(), indexType, _versionMajor, _versionMinor, _versionPatch);
    mpi.getMappedInferenceVersionMutable().assign(mappedInferenceVersion);
}

}  // namespace

//
// createAddMappedInferenceVersionOp
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddMappedInferenceVersionOpPass(Logger log, uint32_t versionMajor,
                                                                                   uint32_t versionMinor,
                                                                                   uint32_t versionPatch) {
    return std::make_unique<AddMappedInferenceVersionOpPass>(log, versionMajor, versionMinor, versionPatch);
}
