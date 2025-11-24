//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

class ShapeRef;

namespace vpux::VPUIP::ExpandDMA {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef outShape, ShapeRef offset);

}  // namespace vpux::VPUIP::ExpandDMA
