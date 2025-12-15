//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/weights_dequantize_to_fakequantize_strategy_getter.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "vpux/compiler/NPU37XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

std::unique_ptr<IGreedilyPassStrategy> IE::createWeightsDequantizeToFakeQuantizeStrategy(mlir::func::FuncOp funcOp) {
    const auto arch = config::getArch(funcOp);

    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX: {
        return std::make_unique<arch37xx::WeightsDequantizeToFakeQuantizeStrategy>();
    }
    default: {
        return std::make_unique<arch50xx::WeightsDequantizeToFakeQuantizeStrategy>();
    }
    }
    VPUX_THROW("Unable to get WeightsDequantizeToFakeQuantizeStrategy for arch {0}", arch);
}
