//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/parameter.hpp"

#include <memory>

namespace ov::test::subgraph {

// Builds the dynamic YUV -> RGB color-space-conversion (CSC) subgraph that FuseColorConversion matches:
//
//   Y  -> Convert -> DynamicReshape ------------------+
//                                                     +-> Concat -> FQ -> Conv -> Add -> FQ -> Transpose
//   UV -> Convert -> Transpose -> Interpolate(2x) -> FQ
//
// The op structure mirrors a customer ONNX model
//
// yInput/uvInput are expected to be f16 parameters with physically NHWC layout:
//   Y  = [1, H, W, 1]
//   UV = [1, H/2, W/2, 2]
//
// Returns the CSC output in NHWC layout: [1, H, W, 3], f16, BGR channel order.
std::shared_ptr<ov::Node> buildDynYuvToRgbPattern(const std::shared_ptr<ov::op::v0::Parameter>& yInput,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& uvInput);

}  // namespace ov::test::subgraph
