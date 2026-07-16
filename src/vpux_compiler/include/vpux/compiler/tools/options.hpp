//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <optional>

namespace vpux {

std::optional<config::Platform> parsePlatform(int argc, char* argv[]);

}  // namespace vpux
