//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

NDTypeInterface getDilatedType(vpux::NDTypeInterface origType, ShapeRef dilations);

}  // namespace vpux
