//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

namespace vpux::IE::arch37xx {

ConvertQuantizeOpsToNceOpsStrategy::ConvertQuantizeOpsToNceOpsStrategy() {
    _canSkipQuantizeAvgPoolConversion = [canUseCMajor = _canUseCMajor](IE::QuantizeOp quantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getInput().getType());
        auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
        auto rank = outType.getRank();
        if (!is8BitStorageType(outType)) {
            return true;
        }

        return isQuantizedPerAxis(quantizeOp.getOutput()) || shouldConvertQuantizeOp(quantizeOp, canUseCMajor) ||
               rank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(quantizeOp) &&
                !vpux::isFloat8Quantized(quantizeOp.getDstElemType()));
    };

    _canSkipDequantizeAvgPoolConversion = [](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        auto rank = inType.getRank();
        if (!is8BitStorageType(inType)) {
            return true;
        }

        return isQuantizedPerAxis(dequantizeOp.getInput()) || rank != 4 || hasConstProducer(dequantizeOp);
    };

    _canSkipQuantizeEltwiseConversion = [canUseCMajor = _canUseCMajor](IE::QuantizeOp quantizeOp) {
        auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
        if (!is8BitStorageType(outType)) {
            return true;
        }
        return isQuantizedPerAxis(quantizeOp.getOutput()) || IE::shouldConvertQuantizeOp(quantizeOp, canUseCMajor);
    };

    _canSkipDequantizeEltwiseConversion = [](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        if (!is8BitStorageType(inType)) {
            return true;
        }
        // Don't support per axis quant of any kind
        return isQuantizedPerAxis(dequantizeOp.getInput()) || hasConstProducer(dequantizeOp);
    };

    _canSkipQuantizeToConvConversion = [canUseCMajor = _canUseCMajor](IE::QuantizeOp quantizeOp) {
        auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
        if (!is8BitStorageType(outType)) {
            return true;
        }
        const auto isPerChannelQuantized = isPerChannelQuantizedType(quantizeOp.getOutput());
        // Explicitly support only for per channel axis quantization, other cases will either run as
        // AvgPool/Eltwise or will remain as Quant/Dequant
        if (isQuantizedPerAxis(quantizeOp.getOutput()) && !isPerChannelQuantized) {
            return true;
        }
        auto rank = outType.getRank();
        return IE::shouldConvertQuantizeOp(quantizeOp, canUseCMajor) || rank != 4;
    };

    _canSkipDequantizeToConvConversion = [](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        if (!is8BitStorageType(inType)) {
            return true;
        }

        // Explicitly support only for per channel axis quantization, other cases will either run as
        // AvgPool/Eltwise or will remain as Quant/Dequant
        const auto isPerChannelQuantized = isPerChannelQuantizedType(dequantizeOp.getInput());
        if (isQuantizedPerAxis(dequantizeOp.getInput()) && !isPerChannelQuantized) {
            return true;
        }
        auto rank = inType.getRank();
        return hasConstProducer(dequantizeOp) || rank != 4;
    };
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareAvgPool(mlir::RewritePatternSet& toAvgPoolPatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) const {
    // HW Eltwise and AvgPool supports only per-tensor bias/scale parameters
    // perTensor quantize/dequantize convert to avgpool
    // E#98802 avgpool is faster than add for big input size.
    // avgpool support rank >= 3, and currently convert shape to 4D does not support avgpool, so here limit to rank = 4

    // Dummy AvgPool's numerical stability is preferred for low-precision floating-point quantization (FP8),
    // eltwise Add with self could shift values further away from 0.0 causing precision loss.

    toAvgPoolPatterns.add<IE::QuantizeDequantizeToAvgPool<IE::QuantizeOp>>(&ctx, _canSkipQuantizeAvgPoolConversion,
                                                                           log);
    toAvgPoolPatterns.add<IE::QuantizeDequantizeToAvgPool<IE::DequantizeOp>>(&ctx, _canSkipDequantizeAvgPoolConversion,
                                                                             log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareEltwise(mlir::RewritePatternSet& toEltwisePatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) const {
    toEltwisePatterns.add<IE::DequantizeToAddRewriter>(&ctx, _canSkipDequantizeEltwiseConversion, log);
    toEltwisePatterns.add<IE::QuantizeToAddRewriter>(&ctx, _canSkipQuantizeEltwiseConversion, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareQuantToConv(mlir::RewritePatternSet& quantToConvPatterns,
                                                            mlir::MLIRContext& ctx, Logger& log) const {
    quantToConvPatterns.add<IE::QuantizeToDwRewriter>(&ctx, _canSkipQuantizeToConvConversion, log);
    quantToConvPatterns.add<IE::DequantizeToDwRewriter>(&ctx, _canSkipDequantizeToConvConversion, log);
}

}  // namespace vpux::IE::arch37xx
