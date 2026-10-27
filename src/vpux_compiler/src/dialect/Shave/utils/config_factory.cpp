//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/utils/config_factory.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"

using namespace vpux::Shave;

//
// CustomLayerConfigFactory
//

CustomLayerConfigFactory::CustomLayerConfigFactory(CustomLayerConfigMap& configCollection, mlir::OpBuilder& builder)
        : _configCollection(configCollection), _builder(builder), _loc(builder.getUnknownLoc()) {};

void CustomLayerConfigFactory::setLocation(mlir::Location loc) {
    _loc = loc;
}

vpux::Shave::CustomOpConfigCollectionOp CustomLayerConfigFactory::buildConfigCollection() {
    constexpr auto DEFAULT_COFIG_COLLECTION_SYM_NAME = "Config";
    auto configCollectionOp =
            Shave::CustomOpConfigCollectionOp::create(_builder, _loc, DEFAULT_COFIG_COLLECTION_SYM_NAME);

    // Create the block for individual configs
    auto& configCollectionRegion = configCollectionOp.getConfigs();
    auto& configCollectionBlock = configCollectionRegion.emplaceBlock();

    for (auto& config : _configCollection) {
        _log.debug("Creating config for Custom Layer {0}", config.first);

        // Always reset insertion point at end of config collection block
        _builder.setInsertionPointToEnd(&configCollectionBlock);

        // build single layer config
        buildSingleConfig(*config.second, config.first);
    }

    return configCollectionOp;
}

void CustomLayerConfigFactory::buildSingleConfig(const CustomLayerConfig& config, const std::string& uniqueId) {
    auto configOp = Shave::CustomOpConfigOp::create(_builder, _loc, uniqueId);

    // Create the block for specific configs
    auto& configOpRegion = configOp.getConfigs();
    auto& configOpBlock = configOpRegion.emplaceBlock();

    _builder.setInsertionPointToEnd(&(configOpBlock));
    buildKernelConfig(config.kernelConfig);

    _builder.setInsertionPointToEnd(&(configOpBlock));
    buildIOConfig(config.ioConfig);

    _builder.setInsertionPointToEnd(&(configOpBlock));
    buildHeapConfig(config.heapConfig);

    _builder.setInsertionPointToEnd(&(configOpBlock));
    buildTilingConfig(config.tilingConfig);
}

void CustomLayerConfigFactory::buildKernelConfig(const CustomLayerConfig::KernelConfig& config) {
    Shave::CustomOpKernelConfigOp::create(_builder, _loc, config.sourcePath, config.kernelType,
                                          _builder.getStringAttr(""), config.usesDMA, config.usesDPU);
}

void CustomLayerConfigFactory::buildIOConfig(const CustomLayerConfig::IOConfig& config) {
    auto ioConfig = Shave::CustomOpIOConfigOp::create(_builder, _loc);

    // Create the block for specific IO configs
    auto& ioConfigOpRegion = ioConfig.getConfigs();
    auto& ioConfigOpBlock = ioConfigOpRegion.emplaceBlock();

    _builder.setInsertionPointToEnd(&(ioConfigOpBlock));

    for (auto& tensorConfig : config.ioTensorConfigs) {
        llvm::SmallVector<mlir::Attribute> supportedRanks;
        llvm::SmallVector<mlir::Attribute> supportedDataTypes;

        for (auto& supportedRank : tensorConfig.supportedRanks) {
            supportedRanks.emplace_back(Shave::TensorRankAttr::get(_builder.getContext(), supportedRank));
        }

        for (auto& supportedDataType : tensorConfig.supportedDataTypes) {
            supportedDataTypes.emplace_back(Shave::TensorDataTypeAttr::get(_builder.getContext(), supportedDataType));
        }

        // Create optional enforced attributes
        auto enforcedRankAttr =
                tensorConfig.enforcedRank.has_value()
                        ? Shave::TensorRankAttr::get(_builder.getContext(), tensorConfig.enforcedRank.value())
                        : nullptr;

        auto enforcedDataTypeAttr =
                tensorConfig.enforcedDataType.has_value()
                        ? Shave::TensorDataTypeAttr::get(_builder.getContext(), tensorConfig.enforcedDataType.value())
                        : nullptr;

        auto enforcedLayoutAttr =
                tensorConfig.enforcedLayout.has_value()
                        ? Shave::TensorLayoutAttr::get(_builder.getContext(), tensorConfig.enforcedLayout.value())
                        : nullptr;

        Shave::CustomOpIOTensorConfigOp::create(
                _builder, _loc, tensorConfig.argIndex, tensorConfig.ioType, _builder.getArrayAttr(supportedRanks),
                _builder.getArrayAttr(supportedDataTypes), enforcedRankAttr, enforcedDataTypeAttr, enforcedLayoutAttr);
    }

    for (auto& scratchConfig : config.scratchConfigs) {
        Shave::CustomOpIOScratchConfigOp::create(_builder, _loc, scratchConfig.argIndex, scratchConfig.size,
                                                 scratchConfig.location);
    }
}

void CustomLayerConfigFactory::buildHeapConfig(const CustomLayerConfig::HeapConfig& config) {
    Shave::CustomOpHeapConfigOp::create(_builder, _loc, config.size);
}

void CustomLayerConfigFactory::buildTilingConfig(const CustomLayerConfig::TilingConfig& config) {
    Shave::CustomOpTilingConfigOp::create(_builder, _loc, config.tilingMode);
}
