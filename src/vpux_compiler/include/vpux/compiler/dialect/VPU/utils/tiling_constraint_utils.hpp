//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux {
namespace VPU {

double getFragmentationAvoidRatioPipeliningLargeWeights(config::ArchKind archKind);
}  // namespace VPU
}  // namespace vpux
