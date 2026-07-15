//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/dma_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/composers/dma_composer.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"

using namespace vpux;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult NNDMARewriter::matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto descriptor = DMADescriptorComposer::compose(origOp, _symRefMap);
    auto dma = rewriter.create<NPUReg40XX::NNDMAOp>(origOp->getLoc(), origOp.getSymNameAttr(), std::move(descriptor),
                                                    origOp.getInputAttr(), origOp.getOutputBuffsAttr(),
                                                    origOp.getNextLinkAttr(), origOp.getActCompressionSizeEntryAttr(),
                                                    origOp.getIndicesAttr(), origOp.getAddressingModeAttr());

    // TODO: (E#114625) Remove once proper refactoring happened
    if (!origOp.getTaskLocationAttr()) {
        dma.getOperation()->setAttr("directLink", rewriter.getUnitAttr());
    }

    if (auto strided = origOp->getAttr(vpux::stridedInputAttrName)) {
        dma->setAttr(vpux::stridedInputAttrName, strided);
    }

    if (auto strided = origOp->getAttr(vpux::stridedOutputAttrName)) {
        dma->setAttr(vpux::stridedOutputAttrName, strided);
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
