//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/opsets/opset11_decl.hpp>
#include <openvino/runtime/core.hpp>

// create dummy network for tests
std::shared_ptr<ov::Model> buildSingleLayerSoftMaxNetwork();

// create dummy network for ws tests
std::shared_ptr<ov::Model> buildSingleWsFriendlyNetwork(ov::element::Type_t inType, const ov::Shape& inputShape);

std::shared_ptr<ov::Model> createModelWithLargeSize();

// class encapsulated Platform getting from environmental variable
class PlatformEnvironment {
public:
    static const std::string PLATFORM;
};
