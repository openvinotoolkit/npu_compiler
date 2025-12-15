// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux::NPUReg50XX {
#define GEN_PASS_DECL_SETUPIDUCMXMUXMODE
#define GEN_PASS_DEF_SETUPIDUCMXMUXMODE
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp.inc"
}  // namespace vpux::NPUReg50XX

using namespace vpux;
using namespace npu40xx;

namespace {
//
// SetupIduCmxMuxModePass
//

class SetupIduCmxMuxModePass : public NPUReg50XX::impl::SetupIduCmxMuxModeBase<SetupIduCmxMuxModePass> {
public:
    SetupIduCmxMuxModePass(Logger log, uint8_t iduCmxMuxMode): _iduCmxMuxMode(iduCmxMuxMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    uint8_t _iduCmxMuxMode;
};

mlir::LogicalResult SetupIduCmxMuxModePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (iduCmxMuxMode.hasValue()) {
        _log.trace("Overloading C++ createSetupIduCmxMuxModePass argument by MLIR variable");
        _iduCmxMuxMode = iduCmxMuxMode.getValue() == "MODE0"   ? 1u
                         : iduCmxMuxMode.getValue() == "MODE1" ? 2u
                         : iduCmxMuxMode.getValue() == "MODE2" ? 3u
                                                               : 0u;
    }

    return mlir::success();
}

void SetupIduCmxMuxModePass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto elfMain = ELF::getElfMainOp(funcOp);

    auto dataSectionOps = elfMain.getOps<ELF::DataSectionOp>();
    for (auto dataSectionOp : dataSectionOps) {
        dataSectionOp.walk([&](NPUReg50XX::DPUInvariantOp invariantOp) {
            auto invariantDescriptorRegMap = invariantOp.getProperties().getDescriptor();
            invariantDescriptorRegMap.write<vpux::NPUReg50XX::Fields::idu_cmx_mux_mode>(_iduCmxMuxMode);
            invariantOp.getProperties().setDescriptor(std::move(invariantDescriptorRegMap));
        });
    }
}

}  // namespace

//
// createSetupIduCmxMuxModePass
//

std::unique_ptr<mlir::Pass> vpux::NPUReg50XX::createSetupIduCmxMuxModePass(Logger log, uint8_t iduCmxMuxMode) {
    return std::make_unique<SetupIduCmxMuxModePass>(log, iduCmxMuxMode);
}
