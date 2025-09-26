//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <fstream>
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/act_kernels/shave_binary_resources.h"

namespace vpux::IE {
#define GEN_PASS_DECL_LOADEXTERNALKERNELRESOURCES
#define GEN_PASS_DEF_LOADEXTERNALKERNELRESOURCES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// LoadExternalKernelResources
//

class LoadExternalKernelResources final :
        public IE::impl::LoadExternalKernelResourcesBase<LoadExternalKernelResources> {
public:
    explicit LoadExternalKernelResources(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void LoadExternalKernelResources::safeRunOnFunc() {
    auto func = getOperation();

    auto extKernelOps = func.getOps<IE::ExternalKernelOp>();
    if (extKernelOps.empty()) {
        _log.debug("No ExternalKernels encountered - loading external kernel resources can be skipped.");
        return;
    }

    auto& shaveBinResources = ShaveBinaryResources::getInstance();
    auto arch = config::getArch(func);

    // Keep track of loaded kernel resources in order to avoid multiple retrivals
    std::set<std::string> loadedKernels;

    for (auto extKernelOp : extKernelOps) {
        auto kernelPath = extKernelOp.getKernelPath().str();
        auto kernelUniqueId = extKernelOp.getUniqueId().str();

        // Skip loading if resources have already been loaded previously
        if (loadedKernels.find(kernelUniqueId) != loadedKernels.end()) {
            continue;
        }

        _log.debug("Loading resources for ExternalKernel {0} from path {1}", kernelUniqueId, kernelPath);
        std::vector<uint8_t> kernelElfBinary{};
        std::ifstream ifileElf(kernelPath, std::ifstream::in);
        VPUX_THROW_UNLESS(ifileElf.is_open(), "ExternalKernelOp: Kernel Path {0} is not valid", kernelPath);

        // Get length of file:
        ifileElf.seekg(0, std::ios::end);
        auto length = ifileElf.tellg();
        ifileElf.seekg(0, std::ios::beg);

        auto buffer = std::vector<char>(length);
        ifileElf.read(buffer.data(), length);
        ifileElf.close();

        // Check the ELF Magic to determine if the path points to a valid ELF file
        if (buffer[1] != 'E' || buffer[2] != 'L' || buffer[3] != 'F') {
            extKernelOp->emitError("Path " + kernelPath + " does not point to a valid ELF binary");
            signalPassFailure();
            return;
        }

        kernelElfBinary.insert(kernelElfBinary.end(), buffer.begin(), buffer.end());
        shaveBinResources.addCompiledElf(kernelUniqueId, kernelElfBinary, arch);
        loadedKernels.insert(std::move(kernelUniqueId));
    }
}

}  // namespace

//
// createLoadExternalKernelResourcesPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createLoadExternalKernelResourcesPass(Logger log) {
    return std::make_unique<LoadExternalKernelResources>(log);
}
