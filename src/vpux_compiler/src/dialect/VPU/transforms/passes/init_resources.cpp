//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_INITRESOURCES
#define GEN_PASS_DEF_INITRESOURCES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// InitResourcesPass
//

class InitResourcesPass final : public VPU::impl::InitResourcesBase<InitResourcesPass> {
public:
    InitResourcesPass() = default;
    InitResourcesPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    config::ArchKind _arch = config::ArchKind::UNKNOWN;
    config::CompilationMode _compilationMode = config::CompilationMode::DefaultHW;
    std::optional<int> _revisionID;
    int _numOfDPUGroups = 1;
    std::optional<int> _numOfDMAPorts;
    std::optional<vpux::Byte> _availableCMXMemory;
    bool _allowCustomValues = false;
};

InitResourcesPass::InitResourcesPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(initCompilerOptions);

    initializeFromOptions();
}

mlir::LogicalResult InitResourcesPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void InitResourcesPass::initializeFromOptions() {
    auto archStr = config::symbolizeEnum<config::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.has_value(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    _arch = archStr.value();

    auto compilationModeStr = config::symbolizeEnum<config::CompilationMode>(compilationModeOpt.getValue());
    VPUX_THROW_UNLESS(compilationModeStr.has_value(), "Unknown compilation mode: '{0}'", compilationModeOpt.getValue());
    _compilationMode = compilationModeStr.value();

    if (revisionIDOpt.hasValue()) {
        _revisionID = revisionIDOpt.getValue();
    }

    _numOfDPUGroups = vpux::VPU::getMaxArchDPUClusterNum(_arch);
    if (numberOfDPUGroupsOpt.hasValue()) {
        _numOfDPUGroups = numberOfDPUGroupsOpt.getValue();
    }

    if (numberOfDMAPortsOpt.hasValue()) {
        _numOfDMAPorts = numberOfDMAPortsOpt.getValue();
    }

    if (availableCMXMemoryOpt.hasValue()) {
        _availableCMXMemory = Byte(static_cast<double>(availableCMXMemoryOpt.getValue()));
    }

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void InitResourcesPass::safeRunOnModule() {
    auto module = getOperation();

    _log.trace("Set VPU architecture to {0}", _arch);
    config::setArch(module, _arch, _numOfDPUGroups, _numOfDMAPorts, _availableCMXMemory, _allowCustomValues);

    VPUX_THROW_WHEN(!_allowCustomValues && config::hasCompilationMode(module),
                    "CompilationMode is already defined, probably you run '--init-compiler' twice");
    if (!config::hasCompilationMode(module)) {
        _log.trace("Set compilation mode to {0}", _compilationMode);
        config::setCompilationMode(module, _compilationMode);
    }

    VPUX_THROW_WHEN(!_allowCustomValues && config::hasRevisionID(module),
                    "RevisionID is already defined, probably you run '--init-compiler' twice");
    if (!config::hasRevisionID(module)) {
        if (_revisionID.has_value()) {
            int revisionIDValue = _revisionID.value();
            std::optional<config::RevisionID> revID = config::symbolizeRevisionID(revisionIDValue);
            if (revID.has_value()) {
                _log.trace("Set RevisionID to {0}", revisionIDValue);
                config::setRevisionID(module, revID.value());
            } else {
                _log.trace("Set RevisionID to REVISION_NONE");
                config::setRevisionID(module, config::RevisionID::REVISION_NONE);
            }
        } else {
            _log.trace("Set RevisionID to REVISION_NONE");
            config::setRevisionID(module, config::RevisionID::REVISION_NONE);
        }
    }

    auto nceCluster = config::getTileExecutor(module);
    if (!nceCluster.hasProcessorFrequency()) {
        auto revisionID = config::getRevisionID(module);
        auto freqMHz = vpux::VPU::getDpuFrequency(_arch, revisionID);
        _log.trace("Set DpuFrequency to {0}", freqMHz);
        nceCluster.setProcessorFrequency(getFPAttr(module.getContext(), freqMHz));
    }
}

}  // namespace

//
// createInitResourcesPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createInitResourcesPass() {
    return std::make_unique<InitResourcesPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createInitResourcesPass(const InitCompilerOptions& initCompilerOptions,
                                                               Logger log) {
    return std::make_unique<InitResourcesPass>(initCompilerOptions, log);
}
