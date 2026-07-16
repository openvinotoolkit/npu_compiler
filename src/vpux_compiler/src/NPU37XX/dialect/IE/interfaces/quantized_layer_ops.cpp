//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/quantized_layer_op_models.hpp"
#include "vpux/compiler/dialect/IE/utils/input_types_utils.hpp"

using namespace vpux;
using namespace IE;

namespace {

// Requirements: input u8, filter u8
bool meetsInputRequirements(mlir::Type t) {
    constexpr auto requiredBitWidth = 8;
    return t.isUnsignedInteger(requiredBitWidth) || t.isSignlessInteger(requiredBitWidth);
}

struct Arch37xxBase {
    static bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) {
        return IE::arch37xx::isMixPrecisionSupported(op, isPReLUSupported, vpux::Logger::global());
    }
    static bool checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput,
                            bool isFloatInput) {
        return IE::arch37xx::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
    }
    static int64_t getMaximumQuantizationLevels(mlir::Operation*) {
        return QuantizationLevels::QUANT_LEVELS_8BIT;
    }
};

struct Arch37xxConvUtils : Arch37xxBase {
    static constexpr IE::InputTypeConstraints inputConstraints{meetsInputRequirements, meetsInputRequirements,
                                                               /*allowPerAxisInput=*/false};
};

using ConvModel = ConvLikeQuantModel<Arch37xxConvUtils, IE::ConvolutionOp, /*CheckPerAxisOnInput=*/true,
                                     /*CheckInputTypes=*/true>;
using GroupConvModel = ConvLikeQuantModel<Arch37xxConvUtils, IE::GroupConvolutionOp, false>;
using TransposedConvModel = ConvLikeQuantModel<Arch37xxConvUtils, IE::TransposedConvolutionOp>;
using AddModel = EltwiseQuantModel<Arch37xxBase, IE::AddOp, VPU::EltwiseType::ADD>;
using MaxPoolModel = MaxPoolQuantModel<Arch37xxBase>;  // RejectPerAxisOutput=true (default)
using AvgPoolModel = AvgPoolQuantModel<Arch37xxBase>;
using MatMulModel = MatMulQuantModel<Arch37xxConvUtils>;

}  // namespace

void vpux::IE::arch37xx::registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // Register the interface for operations that support mixed precision and can be lowered to NCE
        IE::ConvolutionOp::attachInterface<ConvModel>(*ctx);
        IE::GroupConvolutionOp::attachInterface<GroupConvModel>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<TransposedConvModel>(*ctx);
        IE::AddOp::attachInterface<AddModel>(*ctx);
        IE::MaxPoolOp::attachInterface<MaxPoolModel>(*ctx);
        IE::AvgPoolOp::attachInterface<AvgPoolModel>(*ctx);
        IE::MatMulOp::attachInterface<MatMulModel>(*ctx);
    });
}
