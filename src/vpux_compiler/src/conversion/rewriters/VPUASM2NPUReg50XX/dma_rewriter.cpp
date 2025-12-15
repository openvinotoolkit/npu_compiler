//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/dma_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/composers/dma_composer.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult NNDMARewriter::matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto descriptor = DMADescriptorComposer::compose(origOp, _symRefMap);

    auto dma = rewriter.create<NPUReg50XX::NNDMAOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), std::move(descriptor), origOp.getInputAttr(),
            origOp.getOutputBuffsAttr(), origOp.getNextLinkAttr(), origOp.getActCompressionSizeEntryAttr(),
            origOp.getActCompressionSparsityMapAttr(), origOp.getIndicesAttr(), origOp.getAddressingModeAttr());

    // TODO: (E#114625) Remove once proper refactoring happened
    if (!origOp.getTaskLocationAttr()) {
        dma.getOperation()->setAttr("directLink", rewriter.getUnitAttr());
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
