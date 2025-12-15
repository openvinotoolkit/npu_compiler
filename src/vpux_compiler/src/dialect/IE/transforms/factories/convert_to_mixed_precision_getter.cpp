//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_to_mixed_precision_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IConvertToMixedPrecisionStrategy> createConvertToMixedPrecisionStrategy(
        mlir::func::FuncOp funcOp, const bool enableFloatInQuantWeightsMixedMode) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return std::make_unique<arch37xx::ConvertToMixedPrecisionStrategy>(enableFloatInQuantWeightsMixedMode);
    case config::ArchKind::NPU50XX:
        return std::make_unique<arch50xx::ConvertToMixedPrecisionStrategy>(enableFloatInQuantWeightsMixedMode);
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace vpux::IE
