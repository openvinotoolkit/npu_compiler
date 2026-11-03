//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <fstream>
#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_factory.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_parser.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
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

    // Loaders
    void loadPrebuiltBinary(ShaveBinaryResources& shaveBinaryResources, const std::string& kernelUniqueId,
                            const std::string& kernelPath, config::ArchKind arch);

private:
    std::string _customKernelConfigPath;
    Logger _log;
    std::set<std::string> _loadedKernels{};
};

void LoadExternalKernelResources::loadPrebuiltBinary(ShaveBinaryResources& shaveBinaryResources,
                                                     const std::string& kernelUniqueId, const std::string& kernelPath,
                                                     config::ArchKind arch) {
    // Skip loading if resources have already been loaded previously
    if (_loadedKernels.find(kernelUniqueId) != _loadedKernels.end()) {
        return;
    }

    _log.debug("Loading resources for ExternalKernel {0} from path {1}", kernelUniqueId, kernelPath);
    std::vector<uint8_t> kernelElfBinary{};
    std::ifstream ifileElf(kernelPath, std::ifstream::in | std::ios::binary);
    VPUX_THROW_UNLESS(ifileElf.is_open(), "ExternalKernelOp: Kernel Path {0} is not valid", kernelPath);

    // Get length of file:
    ifileElf.seekg(0, std::ios::end);
    const auto length = ifileElf.tellg();
    VPUX_THROW_UNLESS(length >= 0, "ExternalKernelOp: Could not determine size of file '{0}'", kernelPath);
    ifileElf.seekg(0, std::ios::beg);

    auto buffer = std::vector<char>(static_cast<size_t>(length));
    ifileElf.read(buffer.data(), static_cast<std::streamsize>(length));
    VPUX_THROW_UNLESS(ifileElf.gcount() == static_cast<std::streamsize>(length),
                      "ExternalKernelOp: Could not fully read file '{0}'", kernelPath);
    ifileElf.close();

    // Check the ELF Magic to determine if the path points to a valid ELF file
    if (buffer.size() < 4 || buffer[0] != '\x7f' || buffer[1] != 'E' || buffer[2] != 'L' || buffer[3] != 'F') {
        _log.error("ExternalKernelOp: Kernel binary at path '{0}' is not a valid ELF file", kernelPath);
        signalPassFailure();
        return;
    }

    kernelElfBinary.insert(kernelElfBinary.end(), buffer.begin(), buffer.end());
    shaveBinaryResources.addCompiledElf(kernelUniqueId, kernelElfBinary, arch);
    _loadedKernels.insert(kernelUniqueId);
    _log.debug("Successfully loaded Kernel Binary for ExternalKernel {0} from path {1}", kernelUniqueId, kernelPath);
}

void LoadExternalKernelResources::safeRunOnModule() {
    auto module = getOperation();

    std::vector<Shave::ExternalKernelOp> extKernelOps;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        auto extKernelOpsInFunc = func.getOps<Shave::ExternalKernelOp>();
        extKernelOps.insert(extKernelOps.end(), extKernelOpsInFunc.begin(), extKernelOpsInFunc.end());
    }

    if (extKernelOps.empty()) {
        return;
    }

    if (_customKernelConfigPath.empty()) {
        _log.warning("No custom kernel config path provided, skipping ExternalKernel resource loading.\nUnless this is "
                     "for a test, this will likely result in a compilation error downstream.");
        return;
    }

    _log.debug("Stage 1: Load configurations for ExternalKernels from XML config file at path {0}",
               _customKernelConfigPath);
    auto swModule = VPUIP::getVPUSWModule(module, _log);
    Shave::CustomLayerConfigParser configParser(_customKernelConfigPath, _log);
    Shave::CustomLayerConfigMap configMap{};
    configParser.loadConfigs(configMap);

    auto builder = mlir::OpBuilder::atBlockBegin(swModule.getBody());
    Shave::CustomLayerConfigFactory irConfigFactory(configMap, builder);
    irConfigFactory.setLocation(swModule->getLoc());

    auto configCollectionOp = irConfigFactory.buildConfigCollection();
    // Flat config collection symbol
    auto configCollectionSymbol = mlir::SymbolRefAttr::get(&getContext(), configCollectionOp.getSymName());

    _log.debug("Stage 2: Attach kernel resource configurations to ExternalKernelOps");
    for (auto extKernelOp : extKernelOps) {
        auto kernelUniqueId = extKernelOp.getUniqueId().str();

        auto configOp = configCollectionOp.lookupSymbol(kernelUniqueId);
        // If a config was detected for a specific External Kernel, attach the config to the op
        if (configOp) {
            // Flat Individual Config sym
            auto flatConfigSymbol = mlir::SymbolRefAttr::get(&getContext(), kernelUniqueId);

            // Full nested sym
            auto fullConfigSymbol = mlir::SymbolRefAttr::get(&getContext(), swModule.getName().value(),
                                                             {configCollectionSymbol, flatConfigSymbol});
            extKernelOp.setConfigAttr(fullConfigSymbol);
        } else {
            extKernelOp.emitError(
                    "No matching entry in the custom kernel config XML; kernel binary will not be loaded.");
            signalPassFailure();
            return;
        }
    }

    _log.debug("Stage 3: Load kernel binaries for ExternalKernels based on the attached configurations");
    auto arch = config::getArch(module);
    auto& shaveBinResources = getShaveBinaryResources(&getContext());

    for (auto individualConfig : configCollectionOp.getOps<Shave::CustomOpConfigOp>()) {
        auto kernelConfigOps = individualConfig.getOps<Shave::CustomOpKernelConfigOp>();

        VPUX_THROW_UNLESS(std::distance(kernelConfigOps.begin(), kernelConfigOps.end()) == 1,
                          "There should only be a single Kernel Config Op per Custom Layer");
        auto kernelConfigOp = *(kernelConfigOps.begin());

        switch (kernelConfigOp.getKernelType()) {
        case Shave::KernelType::PREBUILT:
            loadPrebuiltBinary(shaveBinResources, individualConfig.getSymName().str(), kernelConfigOp.getSource().str(),
                               arch);
            break;
        case Shave::KernelType::TRITON:
            VPUX_THROW("Not implemented");
        };
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
