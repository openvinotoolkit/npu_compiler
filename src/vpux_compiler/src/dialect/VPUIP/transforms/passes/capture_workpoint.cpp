//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/profiling/common.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CAPTUREWORKPOINT
#define GEN_PASS_DEF_CAPTUREWORKPOINT
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

void insertCaptureDma(mlir::OpBuilder& builder, int64_t profOutputId, size_t dstDdrOffset) {
    auto* ctx = builder.getContext();

    const auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::Register));
    const vpux::NDTypeInterface hwTimerType =
            getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memKindAttr);

    const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, profiling::PROFILING_WORKPOINT_READ_ATTR));
    // Create declaration for source buffer which corresponds to HW register with free-running counter
    auto hwRegOp = builder.create<VPURT::DeclareBufferOp>(loc, hwTimerType, VPURT::BufferSection::Register,
                                                          VPUIP::HW_PLL_WORKPOINT_ABSOLUTE_ADDR);

    const auto profilingOutputType = vpux::getMemRefType(hwTimerType.getShape(), hwTimerType.getElementType());
    auto dstBufProfResultOp = builder.create<VPURT::DeclareBufferOp>(
            loc, profilingOutputType, VPURT::BufferSection::ProfilingOutput, profOutputId, dstDdrOffset);

    const auto port = 0;
    // Since the payload is copied into the final destination in DDR, no barriers are needed, so it may be inserted
    // anywhere in the network without barriers setup
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, /*waitBarriers=*/{}, /*updateBarriers=*/{}, loc, hwRegOp.getBuffer(),
                                          dstBufProfResultOp.getBuffer(), port, /*is_out_of_order=*/true,
                                          /*is_critical=*/false, /*spillId=*/nullptr);
}

//
//  CaptureWorkpointPass
//

class CaptureWorkpointPass final : public VPUIP::impl::CaptureWorkpointBase<CaptureWorkpointPass> {
public:
    explicit CaptureWorkpointPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CaptureWorkpointPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    const auto arch = config::getArch(module);

    auto [netInfo, func] = net::getFromModule(module);
    mlir::OpBuilder builder(&func.getBody().front().front());

    const auto profOutputId = static_cast<int64_t>(netInfo.getProfilingOutputsCount());
    const auto outputResult = mlir::MemRefType::get({profiling::WORKPOINT_BUFFER_SIZE / 4}, getUInt32Type(ctx));

    // Update network output information to have also new pll profiling result
    auto profilingResult = addNewProfilingOutput(ctx, func, netInfo, outputResult, profiling::ExecutorType::WORKPOINT);
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.getOperandsMutable().append(profilingResult);

    if (arch == config::ArchKind::NPU37XX) {
        // Set the insert point because workpoint register is accessible by DMA
        builder.setInsertionPoint(&func.getBody().front().front());
        // Capture setup in the begin of inference
        insertCaptureDma(builder, profOutputId, 0);
        // And in the end
        builder.setInsertionPoint(returnOp);
        insertCaptureDma(builder, profOutputId, VPUIP::HW_PLL_WORKPOINT_SIZE);
    }
}

}  // namespace

//
// createCaptureWorkpointPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCaptureWorkpointPass(Logger log) {
    return std::make_unique<CaptureWorkpointPass>(log);
}
