//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPCUSTOMCONSTRAINTS
#define GEN_PASS_DEF_SETUPCUSTOMCONSTRAINTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupCustomConstraintsPass
//

class SetupCustomConstraintsPass final : public VPU::impl::SetupCustomConstraintsBase<SetupCustomConstraintsPass> {
public:
    SetupCustomConstraintsPass() = default;
    SetupCustomConstraintsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final {
        if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
            return mlir::failure();
        }

        return mlir::success();
    }

    void safeRunOnModule() final {
        auto moduleOp = getModuleOp(getOperation());
        auto ctx = moduleOp->getContext();
        auto customConstraintsOp = moduleOp.lookupSymbol<config::CustomConstraintsOp>("Constraints");
        if (customConstraintsOp == nullptr) {
            return;
        }

        // Override max kernel size constraint in the context with the value from IR pipeline options.
        auto maxKernelSizeOption = customConstraintsOp.lookupSymbol<config::ConstraintOp>("constraint.MaxKernelSize");
        if (maxKernelSizeOption != nullptr) {
            const auto maxKernelSize = config::getNPUConstraints(ctx).maxKernelSize;
            auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(maxKernelSizeOption.getConstraintValue());
            const auto rawValue = intAttr.getSInt();
            VPUX_THROW_UNLESS(rawValue >= 0 && rawValue <= std::numeric_limits<uint32_t>::max(),
                              "Max kernel size value {0} is out of uint32_t range", rawValue);
            auto pipelineMaxKernelSize = static_cast<uint32_t>(rawValue);
            config::updateMaxKernelSize(ctx, pipelineMaxKernelSize);
            const auto updatedMaxKernelSize = config::getNPUConstraints(ctx).maxKernelSize;
            VPUX_THROW_WHEN(updatedMaxKernelSize != pipelineMaxKernelSize,
                            "Failed to update max kernel size constraint");
            _log.trace("Max kernel size is initialized with value {0}, and it is overridden to pipeline option {1}",
                       maxKernelSize, pipelineMaxKernelSize);
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupCustomConstraintsPass() {
    return std::make_unique<SetupCustomConstraintsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupCustomConstraintsPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupCustomConstraintsPass>(initCompilerOptions, log);
}
