//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/nnrt_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult NNRTConfigRewriter::matchAndRewrite(VPUASM::NNrtConfigOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();
    npu40xx::nn_public::VpuNNRTConfig nnrtConfig = {};

    std::optional<uint64_t> stackSize;
    std::optional<std::pair<uint32_t, uint32_t>> stackFrames;
    if (origOp.getActShaveStacks().has_value()) {
        auto stackRef =
                _symRefMap.lookupSymbol(mlir::dyn_cast<mlir::SymbolRefAttr>(*origOp.getActShaveStacks()->begin()));
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameOp>(stackRef);
        stackSize = stackOp.getStackSize();
    } else {
        stackFrames = std::make_pair(NPUReg50XX::STACK_FRAME1, NPUReg50XX::STACK_FRAME2);
        stackSize = NPUReg50XX::STACK_FRAME2 - NPUReg50XX::STACK_FRAME1;
    }

    NPUReg40XX::fillNNrtConfig<NPUReg50XX::ActShaveRtOp>(nnrtConfig.shv_rt_configs, origOp, origOp.getActShaveRt(),
                                                         stackSize, isActShaveProfilingEnabled,
                                                         origOp.getIsActKernelInvocations(), stackFrames);

    VpuNNRTConfig nnrtDesc;
    VPUX_THROW_UNLESS(sizeof(npu40xx::nn_public::VpuNNRTConfig) == nnrtDesc.size(),
                      "HW VpuNNRTConfig size {0} != regMapped representation size {1}.",
                      sizeof(npu40xx::nn_public::VpuNNRTConfig), nnrtDesc.size());
    nnrtDesc.copyFrom(nnrtConfig);

    rewriter.create<NPUReg50XX::NNrtConfigOp>(origOp.getLoc(), origOp.getSymNameAttr(),
                                              origOp.getIsActKernelInvocationsAttr(), origOp.getActShaveRtAttr(),
                                              origOp.getActShaveStacksAttr(), origOp.getDmaHwpBaseAttr(),
                                              origOp.getHwpWorkpointCfgAttr(), std::move(nnrtDesc));
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
