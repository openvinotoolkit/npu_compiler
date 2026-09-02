//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {
bool isNotDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape);
}  // namespace vpux
