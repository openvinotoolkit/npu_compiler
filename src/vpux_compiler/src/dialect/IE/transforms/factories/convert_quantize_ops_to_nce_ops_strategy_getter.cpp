//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_quantize_ops_to_nce_ops_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"

using namespace vpux;

namespace vpux::IE {

std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> createConvertQuantizeOpsToNceOpsStrategy() {
    return std::make_unique<IE::arch37xx::ConvertQuantizeOpsToNceOpsStrategy>();
}

}  // namespace vpux::IE
