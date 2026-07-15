//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/nnrt_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult NNRTConfigRewriter::matchAndRewrite(VPUASM::NNrtConfigOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    std::optional<uint64_t> stackSize;
    if (origOp.getActShaveStacks().has_value()) {
        auto stackRef =
                _symRefMap.lookupSymbol(mlir::dyn_cast<mlir::SymbolRefAttr>(*origOp.getActShaveStacks()->begin()));
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameBuffOp>(stackRef);
        stackSize = stackOp.getStackSize();
    }
    // NPU4 does not have stack frames provided by compiler
    // they are resolved by shave driver when initialized.

    npu40xx::nn_public::VpuNNRTConfig nnrtConfig = {};
    NPUReg40XX::fillNNrtConfig<NPUReg40XX::ActShaveRtOp>(nnrtConfig.shv_rt_configs, origOp, origOp.getActShaveRt(),
                                                         stackSize, isActShaveProfilingEnabled,
                                                         origOp.getIsActKernelInvocations(), /*stackFrames*/ {});

    VpuNNRTConfig nnrtDesc;
    VPUX_THROW_UNLESS(sizeof(npu40xx::nn_public::VpuNNRTConfig) == nnrtDesc.size(),
                      "HW VpuNNRTConfig size {0} != regMapped representation size {1}.",
                      sizeof(npu40xx::nn_public::VpuNNRTConfig), nnrtDesc.size());
    nnrtDesc.copyFrom(nnrtConfig);

    rewriter.create<NPUReg40XX::NNrtConfigOp>(origOp.getLoc(), origOp.getSymNameAttr(),
                                              origOp.getIsActKernelInvocations(), origOp.getActShaveRtAttr(),
                                              origOp.getActShaveStacksAttr(), origOp.getDmaHwpBaseAttr(),
                                              origOp.getHwpWorkpointCfgAttr(), std::move(nnrtDesc));
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
