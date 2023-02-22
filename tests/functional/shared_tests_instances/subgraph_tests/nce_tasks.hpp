// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <array>

#pragma once

namespace NCETasksHelpers {

enum NCEOpType {
    AveragePooling,
    Conv2d,
    EltwiseAdd,
    GroupConv2d,
    MaxPooling,
};

std::shared_ptr<ov::Node> buildNCETask(const ov::Output<ov::Node>& param, const NCEOpType& opType);
std::shared_ptr<ov::Node> quantize(const ov::Output<ov::Node>& producer, const std::array<float, 4>& fqRange);
std::string NCEOpTypeToString(const NCEOpType& opType);

}  // namespace NCETasksHelpers
