//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_quantize_ops_to_nce_ops_strategy.hpp"

#include <functional>

namespace vpux::IE::arch60xx {
class ConvertQuantizeOpsToNceOpsStrategy final : public vpux::IE::IConvertQuantizeOpsToNceOpsStrategy {
public:
    ConvertQuantizeOpsToNceOpsStrategy();

    void prepareAvgPool(mlir::RewritePatternSet& toAvgPoolPatterns, mlir::MLIRContext& ctx, Logger& log) const override;
    void prepareEltwise(mlir::RewritePatternSet& toEltwisePatterns, mlir::MLIRContext& ctx, Logger& log) const override;
    void prepareQuantToConv(mlir::RewritePatternSet& quantToConvPatterns, mlir::MLIRContext& ctx,
                            Logger& log) const override;

private:
    const bool _canUseCMajor = false;
    std::function<bool(IE::QuantizeOp)> _canSkipQuantizeAvgPoolConversion;
    std::function<bool(IE::DequantizeOp)> _canSkipDequantizeAvgPoolConversion;
    std::function<bool(IE::QuantizeOp)> _canSkipQuantizeEltwiseConversion;
    std::function<bool(IE::DequantizeOp)> _canSkipDequantizeEltwiseConversion;
    std::function<bool(IE::QuantizeOp)> _canSkipQuantizeToConvConversion;
    std::function<bool(IE::DequantizeOp)> _canSkipDequantizeToConvConversion;
};

}  // namespace vpux::IE::arch60xx
