//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/act_kernel_invocation_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult ActKernelInvocationRewriter::matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelRangeRef = _symRefMap.lookupSymbol(origOp.getKernelRange());
    auto kernelRangeTaskBufferOp = mlir::cast<VPUASM::DeclareTaskBufferOp>(kernelRangeRef);
    auto kernelRangeTileMask = VPUASM::getTileSelectMaskForBuffer(kernelRangeTaskBufferOp);
    auto kernelRangeIndex = origOp.getRangeIndex();

    uint64_t perfPacketTileMask = 0;
    if (auto profilingDataOpt = origOp.getProfilingData()) {
        auto perfPacketBufferRef = _symRefMap.lookupSymbol(*profilingDataOpt);
        auto perfPacketBufferOp = mlir::cast<VPUASM::DeclareBufferOp>(perfPacketBufferRef);
        perfPacketTileMask = VPUASM::getTileSelectMaskForBuffer(perfPacketBufferOp);
    }

    auto waitMaskHi = VPUMI40XX::computeMaskHi(origOp.getWaitBarriers());
    auto waitMaskLo = VPUMI40XX::computeMaskLo(origOp.getWaitBarriers());
    auto postMaskHi = VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers());
    auto postMaskLo = VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers());

    uint8_t barrier_group = 0;
    uint8_t barrier_mask = 0;

    std::tie(barrier_group, barrier_mask) = ELF::reduceWaitMaskTo8bit(waitMaskLo);

    auto nextAkiTileMask = origOp.getNextLink().has_value() ? kernelRangeTileMask : 0;

    VpuActKernelInvocation descriptor;
    descriptor.write<Fields::range>(kernelRangeTileMask);
    descriptor.write<Fields::barriers_wait_mask_hi_act>(waitMaskHi);
    descriptor.write<Fields::barriers_wait_mask_lo_act>(waitMaskLo);
    descriptor.write<Fields::barriers_post_mask_hi_act>(postMaskHi);
    descriptor.write<Fields::barriers_post_mask_lo_act>(postMaskLo);
    descriptor.write<Fields::group_act>(barrier_group);
    descriptor.write<Fields::mask_act>(barrier_mask);
    descriptor.write<Registers::act_invo_barriers_sched, Fields::start_after_>(origOp.getStartAfter());
    descriptor.write<Registers::act_invo_barriers_sched, Fields::clean_after_>(origOp.getCleanAfter());
    descriptor.write<Fields::invo_index>(origOp.getTaskIndex().getValue());
    descriptor.write<Fields::invo_tile>(origOp.getTile());
    descriptor.write<Fields::kernel_range_index>(kernelRangeIndex);
    descriptor.write<Fields::perf_packet_out>(perfPacketTileMask);
    descriptor.write<Fields::next_aki_wl_addr>(nextAkiTileMask);

    rewriter.create<NPUReg50XX::ActKernelInvocationOp>(origOp->getLoc(), origOp.getSymNameAttr(), std::move(descriptor),
                                                       origOp.getTaskLocationAttr(), origOp.getNextLinkAttr(),
                                                       origOp.getKernelRangeAttr(), origOp.getKernelDataAttr(),
                                                       origOp.getKernelParamsAttr(), origOp.getProfilingDataAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
