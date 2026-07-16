//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/quantized_layer_op_models.hpp"
#include "vpux/compiler/dialect/IE/utils/input_types_utils.hpp"

using namespace vpux;
using namespace IE;

namespace {

// Requirements: input u8|fp8, filter u8|fp8
bool meetsInputRequirements(mlir::Type t) {
    constexpr auto requiredBitWidth = 8;
    return t.isUnsignedInteger(requiredBitWidth) || t.isSignlessInteger(requiredBitWidth) || vpux::isFloat8(t);
}

struct Arch50xxBase {
    static bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) {
        return IE::arch50xx::isMixPrecisionSupported(op, isPReLUSupported, vpux::Logger::global());
    }
    static bool checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput,
                            bool isFloatInput) {
        return IE::arch50xx::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
    }
    static int64_t getMaximumQuantizationLevels(mlir::Operation*) {
        return QuantizationLevels::QUANT_LEVELS_8BIT;
    }
};

struct Arch50xxConvUtils : Arch50xxBase {
    static constexpr IE::InputTypeConstraints inputConstraints{meetsInputRequirements, meetsInputRequirements};
};

using ConvModel = ConvLikeQuantModel<Arch50xxConvUtils, IE::ConvolutionOp, /*CheckPerAxisOnInput=*/true,
                                     /*CheckInputTypes=*/true>;
using GroupConvModel = ConvLikeQuantModel<Arch50xxConvUtils, IE::GroupConvolutionOp, false>;
using TransposedConvModel = ConvLikeQuantModel<Arch50xxConvUtils, IE::TransposedConvolutionOp>;
using GroupTransposedConvModel = ConvLikeQuantModel<Arch50xxConvUtils, IE::GroupTransposedConvolutionOp>;
using AddModel = EltwiseQuantModel<Arch50xxBase, IE::AddOp, VPU::EltwiseType::ADD>;
using MultiplyModel = EltwiseQuantModel<Arch50xxBase, IE::MultiplyOp, VPU::EltwiseType::MULTIPLY>;
using SubtractModel = EltwiseQuantModel<Arch50xxBase, IE::SubtractOp, VPU::EltwiseType::SUBTRACT>;
using MaxPoolModel = MaxPoolQuantModel<Arch50xxBase, false>;
using AvgPoolModel = AvgPoolQuantModel<Arch50xxBase>;
using MatMulModel = MatMulQuantModel<Arch50xxConvUtils>;

}  // namespace

void vpux::IE::arch50xx::registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // Register the interface for operations that support mixed precision and can be lowered to NCE
        // Note: arch50xx checks for LayerWithPostOpInterface in isMixPrecisionSupported, so we register
        // for all operations that have that interface
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
