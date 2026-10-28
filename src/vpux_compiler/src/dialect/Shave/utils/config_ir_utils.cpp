//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/utils/config_ir_utils.hpp"
#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux::Shave;

//
// Config IR Utils
//

CustomLayerConfig::HeapConfig vpux::Shave::parseHeapConfig(mlir::Operation* op) {
    auto heapConfigOp = mlir::dyn_cast<CustomOpHeapConfigOp>(op);
    VPUX_THROW_UNLESS(heapConfigOp, "Invalid op for parsing Heap Config");

    CustomLayerConfig::HeapConfig heapConfig{};
    heapConfig.size = heapConfigOp.getSize();
    return heapConfig;
}

CustomLayerConfig::KernelConfig vpux::Shave::parseKernelConfig(mlir::Operation* op) {
    auto kernelConfigOp = mlir::dyn_cast<CustomOpKernelConfigOp>(op);
    VPUX_THROW_UNLESS(kernelConfigOp, "Invalid op for parsing Kernel Config");

    CustomLayerConfig::KernelConfig kernelConfig{};
    kernelConfig.sourcePath = kernelConfigOp.getSource().str();
    kernelConfig.kernelType = kernelConfigOp.getKernelType();
    // usesDMA/usesDPU are UnitAttrs: present means true, absent means false
    kernelConfig.usesDMA = kernelConfigOp.getUsesDma();
    kernelConfig.usesDPU = kernelConfigOp.getUsesDpu();
    return kernelConfig;
}

CustomLayerConfig::TilingConfig vpux::Shave::parseTilingConfig(mlir::Operation* op) {
    auto tilingConfigOp = mlir::dyn_cast<CustomOpTilingConfigOp>(op);
    VPUX_THROW_UNLESS(tilingConfigOp, "Invalid op for parsing Tiling Config");

    CustomLayerConfig::TilingConfig tilingConfig{};
    tilingConfig.tilingMode = tilingConfigOp.getMode();
    return tilingConfig;
}

CustomLayerConfig::IOConfig vpux::Shave::parseIOConfig(mlir::Operation* op) {
    auto ioConfigOp = mlir::dyn_cast<CustomOpIOConfigOp>(op);
    VPUX_THROW_UNLESS(ioConfigOp, "Invalid op for parsing IO Config");

    CustomLayerConfig::IOConfig ioConfig{};

    auto& ioConfigRegion = ioConfigOp.getConfigs();
    for (auto& op : ioConfigRegion.getOps()) {
        if (auto tensorConfigOp = mlir::dyn_cast<CustomOpIOTensorConfigOp>(&op)) {
            CustomLayerConfig::IOTensorConfig tensorConfig{};
            tensorConfig.argIndex = tensorConfigOp.getArgIndex();
            tensorConfig.ioType = tensorConfigOp.getIoType();

            for (auto rankEnumVal : tensorConfigOp.getSupportedRankValues()) {
                tensorConfig.supportedRanks.push_back(rankEnumVal);
            }

            for (auto dataTypeEnumVal : tensorConfigOp.getSupportedDtypeValues()) {
                tensorConfig.supportedDataTypes.push_back(dataTypeEnumVal);
            }

            // Parse enforced fields (optional)
            if (auto enforcedRankAttr = tensorConfigOp.getEnforcedRank()) {
                tensorConfig.enforcedRank = enforcedRankAttr.value();
            }

            if (auto enforcedDtypeAttr = tensorConfigOp.getEnforcedDtype()) {
                tensorConfig.enforcedDataType = enforcedDtypeAttr.value();
            }

            if (auto enforcedLayoutAttr = tensorConfigOp.getEnforcedLayout()) {
                tensorConfig.enforcedLayout = enforcedLayoutAttr.value();
            }

            ioConfig.ioTensorConfigs.push_back(std::move(tensorConfig));
        } else if (auto scratchConfigOp = mlir::dyn_cast<CustomOpIOScratchConfigOp>(&op)) {
            CustomLayerConfig::ScratchConfig scratchConfig{};
            scratchConfig.argIndex = scratchConfigOp.getArgIndex();
            scratchConfig.size = scratchConfigOp.getSize();
            scratchConfig.location = scratchConfigOp.getLocation();
            ioConfig.scratchConfigs.push_back(scratchConfig);
        }
    }
    return ioConfig;
}
