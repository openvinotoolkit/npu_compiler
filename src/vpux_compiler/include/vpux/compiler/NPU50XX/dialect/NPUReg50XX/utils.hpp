//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"

namespace vpux {
namespace NPUReg50XX {

VPUMI40XX::TaskDescriptorConstraintsPassArgs getTaskDescriptorConstraintsArgs();

}  // namespace NPUReg50XX
}  // namespace vpux
