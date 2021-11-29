//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
    InitCompilerPass(VPU::ArchKind arch, VPU::CompilationMode compilationMode, Optional<int> numOfDPUGroups,
                     Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    VPU::ArchKind _arch = VPU::ArchKind::KMB;
    VPU::CompilationMode _compilationMode = VPU::CompilationMode::DefaultHW;
    Optional<int> _numOfDPUGroups;
};

InitCompilerPass::InitCompilerPass(VPU::ArchKind arch, VPU::CompilationMode compilationMode,
                                   Optional<int> numOfDPUGroups, Logger log)
        : _arch(arch), _compilationMode(compilationMode), _numOfDPUGroups(numOfDPUGroups) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult InitCompilerPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    auto archStr = VPU::symbolizeEnum<VPU::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.hasValue(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    _arch = archStr.getValue();

    auto compilationModeStr = VPU::symbolizeEnum<VPU::CompilationMode>(compilationModeOpt.getValue());
    VPUX_THROW_UNLESS(compilationModeStr.hasValue(), "Unknown compilation mode: '{0}'", compilationModeOpt.getValue());
    _compilationMode = compilationModeStr.getValue();

    if (numberOfDPUGroupsOpt.hasValue()) {
        _numOfDPUGroups = numberOfDPUGroupsOpt.getValue();
    }

    return mlir::success();
}

void InitCompilerPass::safeRunOnModule() {
    auto module = getOperation();

    _log.trace("Set VPU architecture to {0}", _arch);
    VPU::setArch(module, _arch, _numOfDPUGroups);

    _log.trace("Set compilation mode to {0}", _compilationMode);
    VPU::setCompilationMode(module, _compilationMode);
}

}  // namespace

//
// createInitCompilerPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createInitCompilerPass() {
    return std::make_unique<InitCompilerPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createInitCompilerPass(ArchKind arch, CompilationMode compilationMode,
                                                              Optional<int> numOfDPUGroups, Logger log) {
    return std::make_unique<InitCompilerPass>(arch, compilationMode, numOfDPUGroups, log);
}
