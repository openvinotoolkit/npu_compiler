//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <fstream>
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_parser.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/act_kernels/shave_binary_resources.h"

namespace vpux::Shave {
#define GEN_PASS_DECL_LOADEXTERNALKERNELRESOURCES
#define GEN_PASS_DEF_LOADEXTERNALKERNELRESOURCES
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace vpux::Shave

using namespace vpux;

namespace {

//
// LoadExternalKernelResources
//

class LoadExternalKernelResources final :
        public Shave::impl::LoadExternalKernelResourcesBase<LoadExternalKernelResources> {
public:
    explicit LoadExternalKernelResources(std::string customKernelConfigPath, Logger log)
            : _customKernelConfigPath(std::move(customKernelConfigPath)), _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    std::string _customKernelConfigPath;
    Logger _log;
};

void LoadExternalKernelResources::safeRunOnModule() {
    auto module = getOperation();

    bool hasExternalKernelOps = false;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        if (!func.getOps<Shave::ExternalKernelOp>().empty()) {
            hasExternalKernelOps = true;
            break;
        }
    }
    if (!hasExternalKernelOps) {
        return;
    }

    if (_customKernelConfigPath.empty()) {
        _log.debug("No custom kernel config path provided, skipping ExternalKernel resource loading");
        return;
    }

    Shave::CustomLayerConfigMap customLayerConfigs;
    Shave::CustomLayerConfigParser configParser(_customKernelConfigPath, _log);
    configParser.loadConfigs(customLayerConfigs);

    // Keep track of loaded kernel resources in order to avoid multiple retrivals across all functions in the module
    std::set<std::string> loadedKernels;

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        auto extKernelOps = func.getOps<Shave::ExternalKernelOp>();
        if (extKernelOps.empty()) {
            continue;
        }

        auto& shaveBinResources = getShaveBinaryResources(&getContext());
        auto arch = config::getArch(func);

        for (auto extKernelOp : extKernelOps) {
            auto kernelUniqueId = extKernelOp.getUniqueId().str();

            // Skip loading if resources have already been loaded previously
            if (loadedKernels.find(kernelUniqueId) != loadedKernels.end()) {
                continue;
            }

            auto configIt = customLayerConfigs.find(kernelUniqueId);
            if (configIt == customLayerConfigs.end()) {
                extKernelOp->emitError("No custom kernel config found for ExternalKernel " + kernelUniqueId);
                signalPassFailure();
                return;
            }
            auto kernelPath = configIt->second->kernelConfig.sourcePath;

            _log.debug("Loading resources for ExternalKernel {0} from path {1}", kernelUniqueId, kernelPath);
            std::vector<uint8_t> kernelElfBinary{};
            std::ifstream ifileElf(kernelPath, std::ifstream::in | std::ifstream::binary);
            VPUX_THROW_UNLESS(ifileElf.is_open(), "ExternalKernelOp: Kernel Path {0} is not valid", kernelPath);

            // Get length of file:
            ifileElf.seekg(0, std::ios::end);
            auto length = ifileElf.tellg();
            ifileElf.seekg(0, std::ios::beg);

            constexpr std::streamoff elfMagicSize = 4;
            if (length < elfMagicSize) {
                extKernelOp->emitError("Path " + kernelPath + " does not point to a valid ELF binary");
                signalPassFailure();
                return;
            }

            auto buffer = std::vector<char>(static_cast<size_t>(length));
            ifileElf.read(buffer.data(), length);
            ifileElf.close();

            // Check the ELF Magic to determine if the path points to a valid ELF file
            if (buffer[0] != '\x7f' || buffer[1] != 'E' || buffer[2] != 'L' || buffer[3] != 'F') {
                extKernelOp->emitError("Path " + kernelPath + " does not point to a valid ELF binary");
                signalPassFailure();
                return;
            }

            kernelElfBinary.insert(kernelElfBinary.end(), buffer.begin(), buffer.end());
            shaveBinResources.addCompiledElf(kernelUniqueId, kernelElfBinary, arch);
            loadedKernels.insert(std::move(kernelUniqueId));
        }
    }
}

}  // namespace

//
// createLoadExternalKernelResourcesPass
//
std::unique_ptr<mlir::Pass> vpux::Shave::createLoadExternalKernelResourcesPass(std::string customKernelConfigPath,
                                                                               Logger log) {
    return std::make_unique<LoadExternalKernelResources>(std::move(customKernelConfigPath), log);
}
