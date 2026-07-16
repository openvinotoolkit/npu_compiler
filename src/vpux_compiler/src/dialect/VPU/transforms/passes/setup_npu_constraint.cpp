//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/wlm_register_config.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPNPUCONSTRAINT
#define GEN_PASS_DEF_SETUPNPUCONSTRAINT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SetupNpuConstraintPass
//

class SetupNpuConstraintPass final : public VPU::impl::SetupNpuConstraintBase<SetupNpuConstraintPass> {
public:
    SetupNpuConstraintPass() = default;
    SetupNpuConstraintPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);

        if (initCompilerOptions.workloadManagementMode.hasValue()) {
            _workloadManagementMode = initCompilerOptions.workloadManagementMode;
        }
        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;

    // Initialize fields from pass options
    void initializeFromOptions();

    bool _allowCustomValues = false;
    bool _enableSwFifoPerShave = false;

    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::FWLM_V1_PAGES;
};

template <typename T>
void addConstraint(mlir::OpBuilder optionsBuilder, config::PipelineOptionsOp pipelineOptionsOp,
                   mlir::StringRef constraintName, T constraintValue, bool allowCustomValues) {
    auto hasPipelineOption = pipelineOptionsOp.lookupSymbol<config::OptionOp>(constraintName) != nullptr;
    VPUX_THROW_WHEN(!allowCustomValues && hasPipelineOption,
                    "Constraint is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOption) {
        return;
    }

    auto* ctx = optionsBuilder.getContext();
    const auto constraintAttr = mlir::StringAttr::get(ctx, constraintName);

    if constexpr (std::is_same_v<T, llvm::SmallVector<uint32_t>>) {
        if (constraintValue.empty()) {
            return;
        }
        optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), constraintAttr,
                                                getIntArrayAttr(ctx, constraintValue));

    } else {
        mlir::IntegerType sizeType = mlir::IntegerType::get(ctx, sizeof(void*) * 8, mlir::IntegerType::Unsigned);
        optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), constraintAttr,
                                                mlir::IntegerAttr::get(sizeType, constraintValue));
    }
}

template <>
void addConstraint<bool>(mlir::OpBuilder optionsBuilder, config::PipelineOptionsOp pipelineOptionsOp,
                         mlir::StringRef constraintName, bool constraintValue, bool allowCustomValues) {
    auto hasPipelineOption = pipelineOptionsOp.lookupSymbol<config::OptionOp>(constraintName) != nullptr;
    VPUX_THROW_WHEN(!allowCustomValues && hasPipelineOption,
                    "Barrier constraint is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOption) {
        return;
    }

    auto* ctx = optionsBuilder.getContext();
    const auto constraintAttr = mlir::StringAttr::get(ctx, constraintName);
    optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), constraintAttr,
                                            mlir::BoolAttr::get(ctx, constraintValue));
}

mlir::LogicalResult SetupNpuConstraintPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupNpuConstraintPass::initializeFromOptions() {
    _enableSwFifoPerShave = enableSwKernelFifoPerShaveEngine;

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void SetupNpuConstraintPass::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp = config::getPipelineOptionsOp(getContext(), moduleOp);
    optionsBuilder =
            mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front(), optionsBuilder.getListener());

    auto arch = config::getArch(getOperation());
    if (_enableSwFifoPerShave && !config::hasSupportForFifoPerShaveEngine(arch)) {
        // if dedicated SHAVE FIFOs are not, or cannot be supported, the feature will be disabled. For convenience, this
        // will be reflected in the IR in pipeline options section, as the value will be accessed by multiple passes.
        _enableSwFifoPerShave = false;
        _log.info("Dedicated FIFOs per SHAVE engine were requested but are currently not supported by the architecture "
                  "({0}). The feature will not be enabled.",
                  arch);
    }

    if (!pipelineOptionsOp.lookupSymbol<config::OptionOp>(config::WORKLOAD_MANAGEMENT_MODE)) {
        config::setWorkloadManagementMode(moduleOp, _workloadManagementMode);
    }

    addConstraint(optionsBuilder, pipelineOptionsOp, config::USE_DEDICATED_FIFO_PER_SHAVE_ENGINE, _enableSwFifoPerShave,
                  _allowCustomValues);

    auto supportsSwFifoPerShave = config::getConstraint<bool>(moduleOp, config::USE_DEDICATED_FIFO_PER_SHAVE_ENGINE);
    _log.info("Support for FIFO per each SHAVE engine enabled: {0}", supportsSwFifoPerShave);

    auto perBarrierSlotConstraint = vpux::VPU::getPerBarrierSlotConstraint();
    auto barrSlotCount = static_cast<unsigned>(perBarrierSlotConstraint.getPerBarrierMaxSlotCount());

    addConstraint(optionsBuilder, pipelineOptionsOp, config::BARR_MAX_SLOT_COUNT, barrSlotCount, _allowCustomValues);

    auto regConfig = vpux::VPU::getRegisterConfig(arch, moduleOp);

    auto shvRegAddrs = regConfig.getSHVRegisterAddrs();
    auto dpuRegAddrs = regConfig.getDPURegisterAddrs();
    auto barrierFifoAddr = regConfig.getNCEBarrierFifoAddr();
    auto barrierFifoDepth = regConfig.getNCEBarrierFifoDepth();
    addConstraint(optionsBuilder, pipelineOptionsOp, config::DPU_FIFO_ADDRS, std::move(dpuRegAddrs),
                  _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::SHV_FIFO_ADDRS, std::move(shvRegAddrs),
                  _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::BARRIER_FIFO_ADDR, barrierFifoAddr, _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::BARRIER_FIFO_DEPTH, barrierFifoDepth, _allowCustomValues);

    // Get Maximum available space in CMX Metadata for various descriptor types
    auto maxVariants = vpux::VPU::getDefaultTaskListCount(VPU::TaskType::DPUVariant, arch);
    auto maxInvariants = vpux::VPU::getDefaultTaskListCount(VPU::TaskType::DPUInvariant, arch);
    auto maxDMAs = vpux::VPU::getDefaultTaskListCount(VPU::TaskType::DMA, arch);

    auto numShvExecutorsPerTile = [&] {
        auto tileOp = config::getTileExecutor(moduleOp);
        auto executorKind = config::ExecutorKind::SHAVE_ACT;
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query {0} executor.", executorKind);
        VPUX_THROW_UNLESS(tileOp.hasSubExecutor(executorKind), "Expected tileOp contain executor of type {0}.",
                          executorKind);
        return supportsSwFifoPerShave ? static_cast<size_t>(tileOp.getSubExecutor(executorKind).getCount())
                                      : static_cast<size_t>(1);
    }();

    VPUX_THROW_UNLESS(numShvExecutorsPerTile == 1 || numShvExecutorsPerTile == 2,
                      "Unsupported number of SHAVE executors '{0}'", numShvExecutorsPerTile);

    auto maxActKernelRange =
            vpux::VPU::getDefaultTaskListCount(VPU::TaskType::ActKernelRange, arch) / numShvExecutorsPerTile;
    auto maxActKernelInvocation =
            vpux::VPU::getDefaultTaskListCount(VPU::TaskType::ActKernelInvocation, arch) / numShvExecutorsPerTile;

    // Set CMX Metadata Constrains
    addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_VARIANT_COUNT, maxVariants,
                  _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_INVARIANT_COUNT, maxInvariants,
                  _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_KERNEL_INVOCATION_COUNT,
                  maxActKernelInvocation, _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_KERNEL_RANGE_COUNT, maxActKernelRange,
                  _allowCustomValues);
    addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_DMA_COUNT, maxDMAs, _allowCustomValues);

    if (!vpux::config::isArchVPUX3XXX(arch)) {
        auto maxMediaCount = vpux::VPU::getDefaultTaskListCount(VPU::TaskType::M2I, arch);
        addConstraint(optionsBuilder, pipelineOptionsOp, config::METADATA_MAX_MEDIA_COUNT, maxMediaCount,
                      _allowCustomValues);
    }
}

}  // namespace

//
// createSetupNpuConstraintPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupNpuConstraintPass() {
    return std::make_unique<SetupNpuConstraintPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupNpuConstraintPass(const VPU::InitCompilerOptions& initCompilerOptions,
                                                                    Logger log) {
    return std::make_unique<SetupNpuConstraintPass>(initCompilerOptions, log);
}
