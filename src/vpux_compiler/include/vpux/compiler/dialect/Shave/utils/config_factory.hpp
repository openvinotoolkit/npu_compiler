//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::Shave {

class CustomLayerConfigFactory {
public:
    CustomLayerConfigFactory() = delete;
    CustomLayerConfigFactory(CustomLayerConfigMap& configCollection, mlir::OpBuilder& builder);

    vpux::Shave::CustomOpConfigCollectionOp buildConfigCollection();
    void setLocation(mlir::Location loc);

private:
    void buildSingleConfig(const CustomLayerConfig& config, const std::string& uniqueId);

    void buildKernelConfig(const CustomLayerConfig::KernelConfig& config);
    void buildIOConfig(const CustomLayerConfig::IOConfig& config);
    void buildHeapConfig(const CustomLayerConfig::HeapConfig& config);
    void buildTilingConfig(const CustomLayerConfig::TilingConfig& config);

private:
    CustomLayerConfigMap _configCollection;
    mlir::OpBuilder _builder;
    vpux::Logger _log = vpux::Logger("custom-layer-config-factory", vpux::Logger::global().level());
    mlir::Location _loc;
};
}  // namespace vpux::Shave
