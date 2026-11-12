//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/nnrt_rewriter.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include <npu_60xx_nnrt.hpp>

using namespace vpux;
using namespace NPUReg60XX;
using namespace NPUReg60XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg60xx {

mlir::LogicalResult NNRTConfigRewriter::matchAndRewrite(VPUASM::NNrtConfigOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();
    npu60xx::nn_public::VpuNNRTConfig nnrtConfig = {};

    std::optional<uint64_t> stackSize;
    SmallVector<uint32_t> stackFrames;
    if (origOp.getActShaveStacks().has_value()) {
        auto stackRef =
                _symRefMap.lookupSymbol(mlir::dyn_cast<mlir::SymbolRefAttr>(*origOp.getActShaveStacks()->begin()));
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameBuffOp>(stackRef);
        stackSize = stackOp.getStackSize();
    } else {
        stackSize = CMX_SHAVE_STACK_SIZE.count();
        stackFrames = VPUASM::getCMXStackFrames(moduleOp);
    }

    NPUReg60XX::fillNNrtConfig<NPUReg60XX::ActShaveRtOp>(nnrtConfig.shv_rt_configs, origOp, origOp.getActShaveRt(),
                                                         stackSize, isActShaveProfilingEnabled,
                                                         origOp.getIsActKernelInvocations(), stackFrames);

    VpuNNRTConfig nnrtDesc;
    VPUX_THROW_UNLESS(sizeof(npu60xx::nn_public::VpuNNRTConfig) == nnrtDesc.size(),
                      "HW VpuNNRTConfig size {0} != regMapped representation size {1}.",
                      sizeof(npu60xx::nn_public::VpuNNRTConfig), nnrtDesc.size());
    nnrtDesc.copyFrom(nnrtConfig);

    NPUReg60XX::NNrtConfigOp::create(rewriter, origOp.getLoc(), origOp.getSymNameAttr(),
                                     origOp.getIsActKernelInvocations(), origOp.getActShaveRtAttr(),
                                     origOp.getActShaveStacksAttr(), origOp.getDmaHwpBaseAttr(),
                                     origOp.getHwpWorkpointCfgAttr(), std::move(nnrtDesc));

    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
