//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux::IE {

// Shape verification is disabled by default and enabled by a dedicated pass once
// shape legalization reaches a valid state.
inline constexpr StringLiteral SHAPE_VERIFICATION_ENABLED = "IE.shape_verification_enabled";

}  // namespace vpux::IE
