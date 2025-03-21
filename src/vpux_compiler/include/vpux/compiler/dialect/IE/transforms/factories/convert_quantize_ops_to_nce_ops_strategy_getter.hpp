//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_quantize_ops_to_nce_ops_strategy.hpp"

namespace vpux::IE {

std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> createConvertQuantizeOpsToNceOpsStrategy();

}  // namespace vpux::IE
