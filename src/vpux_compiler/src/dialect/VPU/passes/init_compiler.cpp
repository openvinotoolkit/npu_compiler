//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"

using namespace vpux;

namespace {

//
// InitCompilerPass
//

class InitCompilerPass final : public VPU::InitCompilerBase<InitCompilerPass> {
public:
    InitCompilerPass() = default;
    InitCompilerPass(VPU::ArchKind arch, Optional<VPU::CompilationMode> compilationMode, Optional<int> numOfDPUGroups,
                     Optional<int> numOfDMAPorts, Optional<int> ddrHeapSize, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    VPU::ArchKind _arch = VPU::ArchKind::UNKNOWN;
    Optional<VPU::CompilationMode> _compilationMode;
    Optional<int> _numOfDPUGroups;
    Optional<int> _numOfDMAPorts;
    Optional<int> _ddrHeapSize;
};

InitCompilerPass::InitCompilerPass(VPU::ArchKind arch, Optional<VPU::CompilationMode> compilationMode,
                                   Optional<int> numOfDPUGroups, Optional<int> numOfDMAPorts, Optional<int> ddrHeapSize,
                                   Logger log)
        : _arch(arch),
          _compilationMode(compilationMode),
          _numOfDPUGroups(numOfDPUGroups),
          _numOfDMAPorts(numOfDMAPorts),
          _ddrHeapSize(ddrHeapSize) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult InitCompilerPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    auto archStr = VPU::symbolizeEnum<VPU::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.hasValue(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    _arch = archStr.getValue();

    if (compilationModeOpt.hasValue()) {
        auto compilationModeStr = VPU::symbolizeEnum<VPU::CompilationMode>(compilationModeOpt.getValue());
        VPUX_THROW_UNLESS(compilationModeStr.hasValue(), "Unknown compilation mode: '{0}'",
                          compilationModeOpt.getValue());
        _compilationMode = compilationModeStr.getValue();
    }

    if (numberOfDPUGroupsOpt.hasValue()) {
        _numOfDPUGroups = numberOfDPUGroupsOpt.getValue();
    }

    return mlir::success();
}

void InitCompilerPass::safeRunOnModule() {
    auto module = getOperation();

    _log.trace("Set VPU architecture to {0}", _arch);
    VPU::setArch(module, _arch, _numOfDPUGroups, _numOfDMAPorts, _ddrHeapSize);

    if (_compilationMode.hasValue()) {
        _log.trace("Set compilation mode to {0}", _compilationMode.getValue());
        VPU::setCompilationMode(module, _compilationMode.getValue());
    }
}

}  // namespace

//
// createInitCompilerPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createInitCompilerPass() {
    return std::make_unique<InitCompilerPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createInitCompilerPass(ArchKind arch, Optional<CompilationMode> compilationMode,
                                                              Optional<int> numOfDPUGroups, Optional<int> numOfDMAPorts,
                                                              Optional<int> ddrHeapSize, Logger log) {
    return std::make_unique<InitCompilerPass>(arch, compilationMode, numOfDPUGroups, numOfDMAPorts, ddrHeapSize, log);
}
