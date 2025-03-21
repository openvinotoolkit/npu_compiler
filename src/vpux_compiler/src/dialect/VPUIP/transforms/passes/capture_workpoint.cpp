//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/capture_workpoint_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/utils/profiling/common.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CAPTUREWORKPOINT
#define GEN_PASS_DEF_CAPTUREWORKPOINT
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

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
    const auto arch = VPU::getArch(module);
    auto archSpecificStrategy = VPUIP::createCaptureWorkpointStrategy(arch);

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netOp, func);
    mlir::OpBuilder builder(&func.getBody().front().front());

    const auto profOutputId = static_cast<int64_t>(netOp.getProfilingOutputsCount());
    const auto outputResult = mlir::MemRefType::get({profiling::WORKPOINT_BUFFER_SIZE / 4}, getUInt32Type(ctx));

    // Update network output information to have also new pll profiling result
    auto profilingResult = addNewProfilingOutput(ctx, func, netOp, outputResult, profiling::ExecutorType::WORKPOINT);
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.getOperandsMutable().append(profilingResult);

    archSpecificStrategy->prepareDMACapture(builder, func, profOutputId, returnOp);
}

}  // namespace

//
// createCaptureWorkpointPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCaptureWorkpointPass(Logger log) {
    return std::make_unique<CaptureWorkpointPass>(log);
}
