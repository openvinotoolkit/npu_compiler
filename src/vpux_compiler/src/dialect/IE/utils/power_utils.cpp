//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"

namespace vpux {
namespace IE {

std::optional<float> getExponentSplatVal(IE::PowerOp powerOp) {
    auto exponentInput = powerOp.getInput2();
    auto exponentCstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(exponentInput.getDefiningOp());
    if (exponentCstOp == nullptr) {
        return std::nullopt;
    }

    // Exponent must be a scalar or tensor with all elements equal
    const auto& constAttr = exponentCstOp.getContentAttr();
    if (!constAttr.isSplat()) {
        return std::nullopt;
    }

    return constAttr.fold().getSplatValue<float>();
}

}  // namespace IE
}  // namespace vpux
