//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/quantized_layer_op_models.hpp"
#include "vpux/compiler/dialect/IE/utils/input_types_utils.hpp"

using namespace vpux;
using namespace IE;

namespace {

// Requirements: input u8|fp8|u16, filter u8|fp8
bool meetsInputRequirements(mlir::Type t) {
    static constexpr auto requiredBitWidth8 = 8;
    static constexpr auto requiredBitWidth16 = 16;
    return t.isUnsignedInteger(requiredBitWidth8) || t.isSignlessInteger(requiredBitWidth8) ||
           t.isUnsignedInteger(requiredBitWidth16) || t.isSignlessInteger(requiredBitWidth16) || vpux::isFloat8(t);
}

bool meetsFilterRequirements(mlir::Type t) {
    static constexpr auto requiredBitWidth8 = 8;
    return t.isUnsignedInteger(requiredBitWidth8) || t.isSignlessInteger(requiredBitWidth8) || vpux::isFloat8(t);
}

struct Arch60xxBase {
    static bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) {
        return IE::arch50xx::isMixPrecisionSupported(op, isPReLUSupported, vpux::Logger::global());
    }
    static bool checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput,
                            bool isFloatInput) {
        return IE::arch50xx::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
    }
    static int64_t getMaximumQuantizationLevels(mlir::Operation* op) {
        auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(op);
        if (mpeEngineInterface == nullptr) {
            return QuantizationLevels::QUANT_LEVELS_8BIT;
        }
        auto mpeEngineAttr = mlir::dyn_cast<VPU::MPEEngine60XXAttr>(mpeEngineInterface.getMPEEngine());
        if (mpeEngineAttr == nullptr) {
            return QuantizationLevels::QUANT_LEVELS_8BIT;
        }
        if (mpeEngineAttr.getMode().getValue() == VPU::MPEEngine60XXMode::DCIM) {
            return QuantizationLevels::QUANT_LEVELS_16BIT;
        }
        return QuantizationLevels::QUANT_LEVELS_8BIT;
    }
};

struct Arch60xxConvUtils : Arch60xxBase {
    static constexpr IE::InputTypeConstraints inputConstraints{meetsInputRequirements, meetsFilterRequirements};
};

using ConvModel = ConvLikeQuantModel<Arch60xxConvUtils, IE::ConvolutionOp, /*CheckPerAxisOnInput=*/true,
                                     /*CheckInputTypes=*/true>;
using GroupConvModel = ConvLikeQuantModel<Arch60xxConvUtils, IE::GroupConvolutionOp, false>;
using TransposedConvModel = ConvLikeQuantModel<Arch60xxConvUtils, IE::TransposedConvolutionOp>;
using GroupTransposedConvModel = ConvLikeQuantModel<Arch60xxConvUtils, IE::GroupTransposedConvolutionOp>;
using AddModel = EltwiseQuantModel<Arch60xxBase, IE::AddOp, VPU::EltwiseType::ADD>;
using MultiplyModel = EltwiseQuantModel<Arch60xxBase, IE::MultiplyOp, VPU::EltwiseType::MULTIPLY>;
using SubtractModel = EltwiseQuantModel<Arch60xxBase, IE::SubtractOp, VPU::EltwiseType::SUBTRACT>;
using MaxPoolModel = MaxPoolQuantModel<Arch60xxBase, false>;
using AvgPoolModel = AvgPoolQuantModel<Arch60xxBase>;
using MatMulModel = MatMulQuantModel<Arch60xxConvUtils>;

}  // namespace

void vpux::IE::arch60xx::registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<ConvModel>(*ctx);
        IE::GroupConvolutionOp::attachInterface<GroupConvModel>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<TransposedConvModel>(*ctx);
        IE::GroupTransposedConvolutionOp::attachInterface<GroupTransposedConvModel>(*ctx);
        IE::AddOp::attachInterface<AddModel>(*ctx);
        IE::MultiplyOp::attachInterface<MultiplyModel>(*ctx);
        IE::SubtractOp::attachInterface<SubtractModel>(*ctx);
        IE::MaxPoolOp::attachInterface<MaxPoolModel>(*ctx);
        IE::AvgPoolOp::attachInterface<AvgPoolModel>(*ctx);
        IE::MatMulOp::attachInterface<MatMulModel>(*ctx);
    });
}
