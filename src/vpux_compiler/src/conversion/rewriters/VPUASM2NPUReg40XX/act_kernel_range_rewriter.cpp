//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/act_kernel_range_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult ActKernelRangeRewriter::matchAndRewrite(VPUASM::ActKernelRangeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelEntry = NPUReg40XX::getKernelEntry(_symRefMap, origOp.getKernelEntry());
    auto kernelTextSize = NPUReg40XX::getKernelTextSize(_symRefMap, origOp.getKernelText());
    auto kernelTaskType = origOp.getKernelTaskType();
    auto kernelPath = NPUReg40XX::getKernelPath(_symRefMap, origOp.getKernelEntry(), kernelTaskType);
    auto actWLtype = static_cast<std::underlying_type<npu40xx::nn_public::VpuActWLType>::type>(
            NPUReg40XX::getActWLType(kernelTaskType));

    VpuActKernelRange descriptor;
    descriptor.write<Fields::type>(actWLtype);
    descriptor.write<Fields::kernel_entry>(kernelEntry);
    descriptor.write<Fields::code_size>(kernelTextSize);

    rewriter.create<NPUReg40XX::ActKernelRangeOp>(origOp->getLoc(), origOp.getSymNameAttr(), std::move(descriptor),
                                                  origOp.getTaskLocationAttr(), origOp.getKernelTextAttr(),
                                                  origOp.getKernelEntryAttr());

    _log.trace("[{0}] Got kernel '{1}' and cpu '{2}'", getDebugName(), kernelPath, config::getArch(origOp));

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
