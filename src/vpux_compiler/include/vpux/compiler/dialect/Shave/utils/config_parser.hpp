//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"

namespace vpux::Shave {

// Implementation is hidden behind a PIMPL so that <pugixml.hpp> does not leak to every
// includer of this header; it is only included in config_parser.cpp.
class CustomLayerConfigParserImpl;

class CustomLayerConfigParser {
public:
    CustomLayerConfigParser() = delete;
    explicit CustomLayerConfigParser(const std::string& configFilePath, Logger log = Logger::global());
    ~CustomLayerConfigParser();

    CustomLayerConfigParser(const CustomLayerConfigParser&) = delete;
    CustomLayerConfigParser& operator=(const CustomLayerConfigParser&) = delete;
    CustomLayerConfigParser(CustomLayerConfigParser&&) noexcept;
    CustomLayerConfigParser& operator=(CustomLayerConfigParser&&) noexcept;

    void loadConfigs(CustomLayerConfigMap& configs);

private:
    std::unique_ptr<CustomLayerConfigParserImpl> _impl;
};

}  // namespace vpux::Shave
