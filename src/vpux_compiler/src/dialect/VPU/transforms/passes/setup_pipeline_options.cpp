//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETUPPIPELINEOPTIONS
#define GEN_PASS_DEF_SETUPPIPELINEOPTIONS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

static std::mutex& getPpeFactoryMutex() {
    static std::mutex mtx;
    return mtx;
}

template <typename ConcreteFactoryT>
static void registerPpeFactory() {
    // Note: Multi-threaded compilation tests will concurrently call SetupPipelineOptions in the same
    // compile_tool instance with alternating PPE versions. A mutex prevents data races, but also allows switches
    // between versions.
    std::lock_guard lock(getPpeFactoryMutex());
    VPU::PpeVersionConfig::setFactory<ConcreteFactoryT>();
}

//
// SetupPipelineOptionsPass
//

class SetupPipelineOptionsPass final : public VPU::impl::SetupPipelineOptionsBase<SetupPipelineOptionsPass> {
public:
    SetupPipelineOptionsPass() = default;
    SetupPipelineOptionsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);

        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    bool _allowCustomValues = false;
};

mlir::LogicalResult SetupPipelineOptionsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupPipelineOptionsPass::initializeFromOptions() {
    auto archStr = VPU::symbolizeEnum<VPU::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.has_value(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    const auto _arch = archStr.value();

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }

    // Register the default PPE factory singleton
    const auto& ppeVersion = ppeVersionOpt.getValue();
    if (ppeVersion == "Auto") {
        if (_arch == VPU::ArchKind::NPU37XX || _arch == VPU::ArchKind::NPU40XX) {
            registerPpeFactory<VPU::arch37xx::PpeFactory>();
            _log.info("Auto target PPE version set to: 'IntPPE'");
        }
    } else if (ppeVersion == "IntPPE") {
        registerPpeFactory<VPU::arch37xx::PpeFactory>();
    } else {
        _log.error("Unknown PPE version name: '{0}'", ppeVersion);
    }
}

void SetupPipelineOptionsPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getModuleOp(getOperation());

    const auto hasPipelineOptions = moduleOp.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS) != nullptr;
    VPUX_THROW_WHEN(!_allowCustomValues && hasPipelineOptions,
                    "PipelineOptions operation is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOptions) {
        return;
    }

    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp =
            optionsBuilder.create<IE::PipelineOptionsOp>(mlir::UnknownLoc::get(&ctx), VPU::PIPELINE_OPTIONS);
    pipelineOptionsOp.getOptions().emplaceBlock();
}

}  // namespace

//
// createSetupPipelineOptionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass() {
    return std::make_unique<SetupPipelineOptionsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupPipelineOptionsPass>(initCompilerOptions, log);
}
