//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"

namespace vpux::Shave {

//
// Custom Layer Configuration
//
// Shall reflect the structure of the custom layer configuration XML
//

class CustomLayerConfig {
public:
    CustomLayerConfig() = default;

    struct KernelConfig {
        std::string sourcePath;
        KernelType kernelType;
        bool usesDMA = false;
        bool usesDPU = false;
    };

    struct IOTensorConfig {
        size_t argIndex;
        IOType ioType;
        std::vector<TensorRank> supportedRanks;
        std::vector<TensorDataType> supportedDataTypes;

        // Actual compiler-commanding configs
        std::optional<TensorRank> enforcedRank;
        std::optional<TensorDataType> enforcedDataType;
        std::optional<TensorLayout> enforcedLayout;
    };

    struct ScratchConfig {
        size_t argIndex;
        ScratchLocation location;
        size_t size;
    };

    struct IOConfig {
        std::vector<IOTensorConfig> ioTensorConfigs;
        std::vector<ScratchConfig> scratchConfigs;
    };

    struct HeapConfig {
        size_t size;
    };

    struct TilingConfig {
        TilingMode tilingMode = TilingMode::OFF;
    };

    enum class SpecificConfigType { KernelConfig, IOConfig, HeapConfig, TilingConfig, UNDEFINED };

public:
    void setUniqueId(const std::string& id) {
        _uniqueId = id;
    }

    std::string getUniqueId() const {
        return _uniqueId;
    }

public:
    KernelConfig kernelConfig{};
    IOConfig ioConfig{};
    HeapConfig heapConfig{};
    TilingConfig tilingConfig{};

private:
    std::string _uniqueId{};
};

// Top level definitions for config collection handling
using CustomLayerConfigPtr = std::shared_ptr<class CustomLayerConfig>;
using CustomLayerConfigMap = std::map<std::string, CustomLayerConfigPtr>;

}  // namespace vpux::Shave
