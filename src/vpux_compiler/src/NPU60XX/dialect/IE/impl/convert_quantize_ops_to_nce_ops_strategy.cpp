//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/NPU60XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux::IE::arch60xx {

ConvertQuantizeOpsToNceOpsStrategy::ConvertQuantizeOpsToNceOpsStrategy() {
    _canSkipQuantizeAvgPoolConversion = [canUseCMajor = _canUseCMajor](IE::QuantizeOp quantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getInput().getType());
        auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
        if (!is8BitStorageType(outType)) {
            return true;
        }
        auto inRank = inType.getRank();
        // Don't support per axis quant of any kind
        return isQuantizedPerAxis(quantizeOp.getOutput()) || shouldConvertQuantizeOp(quantizeOp, canUseCMajor) ||
               inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(quantizeOp) &&
                !vpux::isFloat8Quantized(quantizeOp.getDstElemType()));
    };

    _canSkipDequantizeAvgPoolConversion = [](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        if (!is8BitStorageType(inType)) {
            return true;
        }
        auto inRank = inType.getRank();

        return isQuantizedPerAxis(dequantizeOp.getInput()) || inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(dequantizeOp) &&
                !vpux::isFloat8Quantized(inType.getElementType())) ||
               hasConstProducer(dequantizeOp);
    };

    _canSkipQuantizeEltwiseConversion = [canUseCMajor = _canUseCMajor](IE::QuantizeOp quantizeOp) {
        auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
        if (!is8BitStorageType(outType)) {
            return true;
        }
        // Don't support per axis quant of any kind
        return isQuantizedPerAxis(quantizeOp.getOutput()) || shouldConvertQuantizeOp(quantizeOp, canUseCMajor);
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
        if (!(is8BitStorageType(outType) || is16BitStorageType(outType))) {
            return true;
        }
        // For NPU6, revision < B it is needed to mitigate the HW erratum related to PPE limitation of representing zero
        // point in 16 bits. Details available on E-#182547 and E-#211644.
        const auto arch = config::getArch(quantizeOp);
        if (arch == config::ArchKind::NPU60XX && config::getRevisionID(quantizeOp) >= config::RevisionID::REVISION_A0 &&
            config::getRevisionID(quantizeOp) < config::RevisionID::REVISION_B) {
            auto qType = mlir::cast<mlir::quant::QuantizedType>(outType.getElementType());
            const bool hasInvalid16BitWithZeroPoint =
                    qType.getStorageTypeIntegralWidth() == 16 && !IE::areAllQuantTypeZeroPointsEqualToZero(qType);
            if (hasInvalid16BitWithZeroPoint) {
                return true;
            }
        }
        // Explicitly support only for per channel axis quantization, other cases will either run as
        // AvgPool/Eltwise or will remain as Quant/Dequant
        const auto isPerChannelQuantized = isPerChannelQuantizedType(quantizeOp.getOutput());
        if (isQuantizedPerAxis(quantizeOp.getOutput()) && !isPerChannelQuantized) {
            return true;
        }
        auto rank = outType.getRank();
        return IE::shouldConvertQuantizeOp(quantizeOp, canUseCMajor) || rank != 4;
    };

    _canSkipDequantizeToConvConversion = [](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        // DCIM supports I16A I8W or U16A U8W Convolutions. If the operation Dequantize consumes has 16 integer bit
        // activations, it can be transformed into QuantCast -> dummy Convolution
        // Allow 16BIT per tensor and per axis quantization to run as Conv by way of DCIM feature
        if (!(is8BitStorageType(inType) || is16BitStorageType(inType))) {
            return true;
        }

        // Explicitly support only for per channel axis quantization, other cases will either run as
        // AvgPool/Eltwise or will remain as Quant/Dequant
        const auto isPerChannelQuantized = isPerChannelQuantizedType(dequantizeOp.getInput());
        if (isQuantizedPerAxis(dequantizeOp.getInput()) && !isPerChannelQuantized) {
            return true;
        }
        auto rank = inType.getRank();
        if (hasConstProducer(dequantizeOp) || rank != 4) {
            return true;
        }

        if (!is16BitStorageType(inType)) {
            return false;
        }

        // Build a prospective DcimMappingInfo for the identity Convolution that DequantizeToConvRewriter
        // would materialize. If that prospective Conv would not satisfy the DCIM SW + HW heuristics,
        // skip the conversion so the Dequantize remains in the graph instead of falling back to an
        // expensive 16-bit NCE convolution.
        const auto outType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getOutput().getType());
        const auto inShape = inType.getShape();
        const int64_t inChannels = inShape[Dims4D::Act::C];
        const auto filterElemType = Const::getWeightsConstFilterElemType(inType.getElementType());
        const auto filterType = mlir::cast<vpux::NDTypeInterface>(
                mlir::RankedTensorType::get({inChannels, inChannels, 1, 1}, filterElemType));
        const auto info =
                getProspectiveConvDcimMappingInfo(inType, filterType, outType, config::isDCIMEnabled(dequantizeOp),
                                                  config::isPowerOptimized(dequantizeOp));
        if (!passesSWHeuristicForDCIM(info, /*checkChannelAlignment=*/false) || !respectHWConstraintsForDCIM(info)) {
            return true;
        }
        return false;
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
    quantToConvPatterns.add<IE::DequantizeToConvRewriter>(&ctx, _canSkipDequantizeToConvConversion, log);
}
}  // namespace vpux::IE::arch60xx
