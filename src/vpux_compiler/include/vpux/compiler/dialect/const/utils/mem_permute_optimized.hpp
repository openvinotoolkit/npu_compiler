//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux {
namespace Const {
namespace details {

// checks capability to perform trivial NCHW<->NHWC and NDHWC<->NCDHW
bool isOptimizedTransformationSupported(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                        const DimsOrder& permOrder);

//
// Performs specialized supported transformations
//
void memPermuteTransformationOptimized(vpux::Const::Content& input, vpux::Const::Content& output);

}  // namespace details
}  // namespace Const
}  // namespace vpux
