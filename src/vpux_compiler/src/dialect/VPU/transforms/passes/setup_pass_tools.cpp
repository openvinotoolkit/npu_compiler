//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/compiler/utils/pass_disabling_callback.hpp"
#include "vpux/compiler/utils/pass_usage_observer.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPPASSTOOLS
#define GEN_PASS_DEF_SETUPPASSTOOLS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupPassToolsPass
//

class SetupPassToolsPass final : public VPU::impl::SetupPassToolsBase<SetupPassToolsPass> {
public:
    SetupPassToolsPass() = default;
    SetupPassToolsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;
};

SetupPassToolsPass::SetupPassToolsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(initCompilerOptions);
}

mlir::LogicalResult SetupPassToolsPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    return Base::initializeOptions(options, errorHandler);
}

void SetupPassToolsPass::safeRunOnModule() {
    if (disabledPassesOpt.hasValue()) {
        auto& actionHandler = getActionHandler(getContext());
        actionHandler.setCallback(PassDisablingCallback(disabledPassesOpt.getValue()));
        actionHandler.addBreakpointManager(PassDisablingCallback::createBreakpointManager());
    }

    if (enableOutdatedPassDetectionOpt.getValue()) {
        auto& actionHandler = getActionHandler(getContext());
        actionHandler.registerObserver(std::make_unique<PassUsageObserver>(_log));
    }

    if (constantTracingOpt.getValue()) {
        auto& actionHandler = getActionHandler(getContext());
        actionHandler.registerObserver(std::make_unique<Const::CallStackObserver>());
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPassToolsPass(const InitCompilerOptions& initCompilerOptions,
                                                                Logger log) {
    return std::make_unique<SetupPassToolsPass>(initCompilerOptions, log);
}
