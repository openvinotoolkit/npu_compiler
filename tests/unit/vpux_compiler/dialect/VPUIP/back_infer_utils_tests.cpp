//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/back_infer_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

// // Run cmd: npuUnitTests --gtest_filter="MLIR_VPUIP_BackInfer.*"

using namespace vpux;

namespace {

// Return the first op of the requested VPUIP view type found while walking the module.
template <typename OpType>
OpType findFirstOp(mlir::ModuleOp module) {
    OpType result = nullptr;
    module.walk([&](OpType op) {
        if (result == nullptr) {
            result = op;
        }
    });
    return result;
}

// Assert two types match on the core NDType facets used across back inference.
void expectSameNDType(mlir::Type actual, mlir::Type expected) {
    ASSERT_NE(actual, nullptr);
    auto actualND = mlir::dyn_cast<vpux::NDTypeInterface>(actual);
    auto expectedND = mlir::dyn_cast<vpux::NDTypeInterface>(expected);
    ASSERT_NE(actualND, nullptr);
    ASSERT_NE(expectedND, nullptr);
    EXPECT_EQ(actualND.getShape(), expectedND.getShape());
    EXPECT_EQ(actualND.getElementType(), expectedND.getElementType());
    EXPECT_EQ(actualND.getDimsOrder(), expectedND.getDimsOrder());
    EXPECT_EQ(actualND.getMemSpace(), expectedND.getMemSpace());
}

vpux::NDTypeInterface asND(mlir::Type type) {
    return mlir::cast<vpux::NDTypeInterface>(type);
}

}  // namespace

class MLIR_VPUIP_BackInfer : public MLIR_UnitBase {
public:
    MLIR_VPUIP_BackInfer(): ctx(registry) {
        ctx.loadDialect<VPUIP::VPUIPDialect>();
    }

    mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
        EXPECT_TRUE(module.get() != nullptr) << "Failed to parse IR";
        return module;
    }

    // A DDR memspace symbol for building alternative mem-space types.
    vpux::IndexedSymbolAttr ddrMemSpace() {
        return vpux::IndexedSymbolAttr::get(&ctx, "DDR");
    }

    mlir::MLIRContext ctx;
};

//
// ViewOp: pure buffer reinterpretation.
//

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ForwardKeepsResultType) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getResult().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ForwardAllowsShapeLayoutReinterpret) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x64x225x16xf16, [@CMX_NN, 0]>)
                    -> memref<1x16x64x225xf16, {order = #NHWC}, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0
                        : memref<1x64x225x16xf16, [@CMX_NN, 0]>
                        to memref<1x16x64x225xf16, {order = #NHWC}, [@CMX_NN, 0]>
                return %0 : memref<1x16x64x225xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getResult().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ForwardRejectsMemSpaceRemap) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto newInput = asND(op.getSource().getType()).changeMemSpace(ddrMemSpace());
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, newInput).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_AllowsResultSmallerThanSourceAllocation) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xui8, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xui8, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xui8, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferredOutput = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferredOutput.has_value());
    expectSameNDType(inferredOutput.value(), op.getResult().getType());

    const auto inferredInput = VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType());
    ASSERT_TRUE(inferredInput.has_value());
    expectSameNDType(inferredInput.value(), op.getSource().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_RejectsResultLargerThanSourceAllocation) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x17xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x17xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x17xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType()).has_value());
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType()).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ReverseKeepsInputType) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getSource().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ReverseRejectsMemSpaceRemap) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto desiredOut = asND(op.getResult().getType()).changeMemSpace(ddrMemSpace());
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, desiredOut).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, ViewOp_ForwardRejectsNonNDType) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x16x16x16xf16, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, [@CMX_NN, 0]> {
                %0 = VPUIP.ViewOp %arg0 : memref<1x16x16x16xf16, [@CMX_NN, 0]> to memref<1x16x16x16xf16, [@CMX_NN, 0]>
                return %0 : memref<1x16x16x16xf16, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto badType = mlir::IntegerType::get(&ctx, 32);
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, badType).has_value());
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, badType).has_value());
}

//
// QuantizeCastOp: element type changes, shape/strides preserved.
//

TEST_F(MLIR_VPUIP_BackInfer, QuantizeCast_ForwardKeepsOutputElemType) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !qElemType = !quant.uniform<u8:f16, 1.0>
        module @test {
            func.func @main(%arg0: memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}> {
                %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}>
                return %0 : memref<1x3x16x16x!qElemType, {order = #NHWC}>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::QuantizeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original storage input -> output keeps the quantized element type.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getInput().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getOutput().getType());
    EXPECT_TRUE(mlir::isa<mlir::quant::QuantizedType>(asND(inferred.value()).getElementType()));
}

TEST_F(MLIR_VPUIP_BackInfer, QuantizeCast_ForwardRejectsBitWidthMismatch) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !qElemType = !quant.uniform<u8:f16, 1.0>
        module @test {
            func.func @main(%arg0: memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}> {
                %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}>
                return %0 : memref<1x3x16x16x!qElemType, {order = #NHWC}>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::QuantizeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // The inferred output keeps the u8 (8-bit) quantized element type but follows the new input's
    // shape/strides. A 32-bit f32 input against an 8-bit quantized output fails the QuantizeCast
    // bit-width legality check -> rejected.
    const auto newInput = asND(op.getInput().getType()).changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, newInput).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, QuantizeCast_ForwardRejectsNonNDType) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !qElemType = !quant.uniform<u8:f16, 1.0>
        module @test {
            func.func @main(%arg0: memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}> {
                %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}>
                return %0 : memref<1x3x16x16x!qElemType, {order = #NHWC}>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::QuantizeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto badType = mlir::IntegerType::get(&ctx, 32);
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, badType).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, QuantizeCast_ReverseInfersStorageInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !qElemType = !quant.uniform<u8:f16, 1.0>
        module @test {
            func.func @main(%arg0: memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}> {
                %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<1x3x16x16xui8, {order = #NHWC}>) -> memref<1x3x16x16x!qElemType, {order = #NHWC}>
                return %0 : memref<1x3x16x16x!qElemType, {order = #NHWC}>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::QuantizeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getOutput().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getInput().getType());
}

//
// GenericReshapeOp: same element count, shape changes.
//

TEST_F(MLIR_VPUIP_BackInfer, GenericReshape_ForwardInfersOutputShape) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1x1x1000xf16> {
                %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
                return %0 : memref<1x1x1x1000xf16>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::GenericReshapeOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original input reproduces the original output shape.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getInput().getType());
    ASSERT_TRUE(inferred.has_value());
    EXPECT_EQ(asND(inferred.value()).getShape(), ShapeRef({1, 1, 1, 1000}));
    EXPECT_EQ(asND(inferred.value()).getNumElements(), 1000);
}

TEST_F(MLIR_VPUIP_BackInfer, GenericReshape_ForwardRejectsElementCountMismatch) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1x1x1000xf16> {
                %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
                return %0 : memref<1x1x1x1000xf16>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::GenericReshapeOp>(module.get());
    ASSERT_NE(op, nullptr);

    // The inferred output copies the fixed output shape (1000 elems). An input with a different
    // element count breaks the num-elements invariant -> rejected.
    const auto newInput = asND(op.getInput().getType()).changeShape(ShapeRef({1, 512}));
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, newInput).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, GenericReshape_ReverseInfersInputShape) {
    constexpr llvm::StringLiteral ir = R"(
        module @test {
            func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1x1x1000xf16> {
                %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
                return %0 : memref<1x1x1x1000xf16>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::GenericReshapeOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getOutput().getType());
    ASSERT_TRUE(inferred.has_value());
    EXPECT_EQ(asND(inferred.value()).getShape(), ShapeRef({1, 1000}));
}

TEST_F(MLIR_VPUIP_BackInfer, GenericReshape_ForwardInfersDistributedAxisThroughLeadingUnitDim) {
    constexpr llvm::StringLiteral ir = R"(
        #I4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #I5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        !InType = !VPUIP.DistributedBuffer<1x16x64x1xf16, #I4, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        !OutType = !VPUIP.DistributedBuffer<1x16x1x8x8xf16, #I5, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2, 1], num_clusters = 2 : i64}>
        module @test {
            func.func private @main(%arg0: !InType) -> !OutType
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_NE(func, nullptr);

    const auto sourceType = mlir::cast<vpux::NDTypeInterface>(func.getFunctionType().getInput(0));
    const auto targetType = mlir::cast<vpux::NDTypeInterface>(func.getFunctionType().getResult(0));
    const auto sourceDistType = mlir::cast<VPUIP::DistributedBufferType>(sourceType);
    const auto axesMapping =
            VPUIP::inferReshapeDistributedAxesMapping(sourceType, targetType, sourceDistType.getDistribution());
    ASSERT_TRUE(axesMapping.has_value());
    EXPECT_EQ(axesMapping->first, 2);
    EXPECT_EQ(axesMapping->second, 3);
}

//
// SubViewOp: forward infers a tile; reverse is intentionally unsupported.
//

TEST_F(MLIR_VPUIP_BackInfer, SubView_ForwardInfersTile) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    -> memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]> {
                %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 8, 8] :
                    memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> to
                    memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
                return %0 : memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::SubViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original input reproduces the sliced output shape.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    EXPECT_EQ(asND(inferred.value()).getShape(), ShapeRef({1, 16, 8, 8}));
}

TEST_F(MLIR_VPUIP_BackInfer, SubView_ForwardRejectsTileLargerThanInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    -> memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]> {
                %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 8, 8] :
                    memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> to
                    memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
                return %0 : memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::SubViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Shrink the input on the sliced axis (C=8) below the tile size (16): the [0,16] tile no
    // longer fits -> isValidSubViewTile fails -> rejected.
    const auto newInput = asND(op.getSource().getType()).changeShape(ShapeRef({1, 8, 8, 8}));
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, newInput).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, SubView_ReverseIsUnsupported) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    -> memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]> {
                %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 8, 8] :
                    memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> to
                    memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
                return %0 : memref<1x16x8x8xf16, {order = #NHWC, strides = [4096, 1, 512, 64]}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::SubViewOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Reverse inference is intentionally not supported for SubView (a tile cannot uniquely
    // reconstruct its source) and must always yield nullopt.
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType()).has_value());
}

//
// PermuteCastOp: dims order + optional per-axis quant axis change.
//

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_ForwardInfersOutputOrder) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original input reproduces the original permuted (NHWC) output type.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getResult().getType());
    EXPECT_EQ(asND(inferred.value()).getDimsOrder(), DimsOrder::NHWC);
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_ForwardRejectsNonNDType) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // A non-ND type is rejected by both directions.
    const auto badType = mlir::IntegerType::get(&ctx, 32);
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, badType).has_value());
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, badType).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_ReverseInfersInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x96x2x2xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Reverse from the original NHWC output reconstructs the NCHW input, and the round-trip guard
    // (forward of the reconstructed input == requested output) accepts it.
    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getSource().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_LayoutCastOriginForwardPreservesLogicalShape) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: memref<1x3x32x32xf16, {order = #NWCH}, @DDR>)
                    -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, is_layout_cast, mem_perm = #NHWC} inputs(%arg0 : memref<1x3x32x32xf16, {order = #NWCH}, @DDR>) -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op.getIsLayoutCastAttr(), nullptr);

    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getResult().getType());
    EXPECT_EQ(asND(inferred.value()).getShape(), asND(op.getSource().getType()).getShape());
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_LayoutCastOriginReversePreservesLogicalShape) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: memref<1x3x32x32xf16, {order = #NWCH}, @DDR>)
                    -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, is_layout_cast, mem_perm = #NHWC} inputs(%arg0 : memref<1x3x32x32xf16, {order = #NWCH}, @DDR>) -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op.getIsLayoutCastAttr(), nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getSource().getType());
    EXPECT_EQ(asND(inferred.value()).getShape(), asND(op.getResult().getType()).getShape());
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_LayoutCastOriginForwardRemapsExplicitStrides) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: memref<1x3x32x32xf16, {order = #NWCH}, @DDR>)
                    -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, is_layout_cast, mem_perm = #NHWC} inputs(%arg0 : memref<1x3x32x32xf16, {order = #NWCH}, @DDR>) -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op.getIsLayoutCastAttr(), nullptr);

    const SmallVector<Bit> stridedInputStrides{524288_Bit, 512_Bit, 16_Bit, 16384_Bit};
    const auto stridedInput = asND(op.getSource().getType()).changeStrides(StridesRef(stridedInputStrides));
    const SmallVector<Bit> expectedOutputStrides{524288_Bit, 16_Bit, 16384_Bit, 512_Bit};

    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, stridedInput);
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getResult().getType());
    EXPECT_EQ(asND(inferred.value()).getStrides(), StridesRef(expectedOutputStrides));
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_LayoutCastOriginForwardRejectsInvalidRemappedStrides) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: memref<1x3x32x32xf16, {order = #NWCH}, @DDR>)
                    -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, is_layout_cast, mem_perm = #NHWC} inputs(%arg0 : memref<1x3x32x32xf16, {order = #NWCH}, @DDR>) -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op.getIsLayoutCastAttr(), nullptr);

    // These strides are legal for the NWCH input, but after preserving the same
    // memory strides under NHWC the resulting MemRefAttr would overlap and is
    // rejected quietly by speculative back-inference.
    const SmallVector<Bit> stridedInputStrides{98304_Bit, 1024_Bit, 32_Bit, 3072_Bit};
    const auto stridedInput = asND(op.getSource().getType()).changeStrides(StridesRef(stridedInputStrides));

    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, stridedInput).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, PermuteCast_LayoutCastOriginReverseRemapsExplicitStrides) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: memref<1x3x32x32xf16, {order = #NWCH}, @DDR>)
                    -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR> {
                %0 = VPUIP.PermuteCast {dst_order = #NHWC, is_layout_cast, mem_perm = #NHWC} inputs(%arg0 : memref<1x3x32x32xf16, {order = #NWCH}, @DDR>) -> memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
                return %0 : memref<1x3x32x32xf16, {order = #NHWC}, @DDR>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::PermuteCastOp>(module.get());
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op.getIsLayoutCastAttr(), nullptr);

    const SmallVector<Bit> stridedOutputStrides{524288_Bit, 16_Bit, 16384_Bit, 512_Bit};
    const auto stridedOutput = asND(op.getResult().getType()).changeStrides(StridesRef(stridedOutputStrides));
    const SmallVector<Bit> expectedInputStrides{524288_Bit, 512_Bit, 16_Bit, 16384_Bit};

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, stridedOutput);
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getSource().getType());
    EXPECT_EQ(asND(inferred.value()).getStrides(), StridesRef(expectedInputStrides));
}

//
// ShapeCastOp (distributed input): shape change with distribution re-derivation.
//

TEST_F(MLIR_VPUIP_BackInfer, ShapeCast_ForwardInfersDistributedOutput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x16x60x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        !OutType = !VPUIP.DistributedBuffer<1x16x30x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> !OutType {
                %0 = VPUIP.ShapeCast {shape = [1, 16, 30, 16]} inputs(%arg0 : !InType) -> !OutType
                return %0 : !OutType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ShapeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original distributed input reproduces the reshaped distributed output.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType());
    ASSERT_TRUE(inferred.has_value());
    EXPECT_EQ(asND(inferred.value()).getShape(), ShapeRef({1, 16, 30, 16}));
    EXPECT_TRUE(mlir::isa<VPUIP::DistributedBufferType>(inferred.value()));
}

TEST_F(MLIR_VPUIP_BackInfer, ShapeCast_ForwardRejectsNonNDType) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x16x60x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        !OutType = !VPUIP.DistributedBuffer<1x16x30x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> !OutType {
                %0 = VPUIP.ShapeCast {shape = [1, 16, 30, 16]} inputs(%arg0 : !InType) -> !OutType
                return %0 : !OutType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ShapeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto badType = mlir::IntegerType::get(&ctx, 32);
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, badType).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, ShapeCast_ForwardQuietlyRejectsUnsupportedPerAxisQuantizedShape) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x4x4x4xf16, {order = #NHWC}, @CMX_NN>)
                    -> memref<1x2x8x4xf16, {order = #NHWC}, @CMX_NN> {
                %0 = VPUIP.ShapeCast {shape = [1, 2, 8, 4]}
                        inputs(%arg0 : memref<1x4x4x4xf16, {order = #NHWC}, @CMX_NN>)
                        -> memref<1x2x8x4xf16, {order = #NHWC}, @CMX_NN>
                return %0 : memref<1x2x8x4xf16, {order = #NHWC}, @CMX_NN>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ShapeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const SmallVector<double> scales{1.0, 2.0, 3.0, 4.0};
    const SmallVector<int64_t> zeroPoints{0, 0, 0, 0};
    const auto perAxisElemType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), scales, zeroPoints, Dims4D::Act::C.ind(), 0, 255);
    const auto newInputType = asND(op.getSource().getType()).changeElemType(perAxisElemType);

    // The candidate changes C and H, while the quiet NHWC per-axis helper only supports C/W reshapes.
    // Back-inference is speculative, so unsupported quantized shape casts must return nullopt instead of throwing.
    EXPECT_NO_THROW({
        const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, newInputType);
        EXPECT_FALSE(inferred.has_value());
    });
}

TEST_F(MLIR_VPUIP_BackInfer, ShapeCast_RejectsUnequalElementCountBackInfer) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: memref<1x4x112x112xf16, {order = #NHWC}, @CMX_NN>)
                    -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN> {
                %0 = VPUIP.ShapeCast {shape = [1, 16, 112, 112]}
                        inputs(%arg0 : memref<1x4x112x112xf16, {order = #NHWC}, @CMX_NN>)
                        -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
                return %0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ShapeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, op.getSource().getType()).has_value());
    EXPECT_FALSE(VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType()).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, ShapeCast_ReverseInfersDistributedInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x16x60x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        !OutType = !VPUIP.DistributedBuffer<1x16x30x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> !OutType {
                %0 = VPUIP.ShapeCast {shape = [1, 16, 30, 16]} inputs(%arg0 : !InType) -> !OutType
                return %0 : !OutType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::ShapeCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getResult().getType());
    ASSERT_TRUE(inferred.has_value());
    EXPECT_EQ(asND(inferred.value()).getShape(), ShapeRef({1, 16, 60, 8}));
}

//
// DistributedCastOp: re-project distribution; requires distributed buffers on both sides.
//

TEST_F(MLIR_VPUIP_BackInfer, DistributedCast_ForwardIdenticalDistribution) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !DistType = !VPUIP.DistributedBuffer<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !DistType) -> !DistType {
                %0 = VPUIP.DistributedCast inputs(%arg0 : !DistType) -> !DistType
                return %0 : !DistType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::DistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Identical input/output distributions are trivially cast-compatible.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getInput().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getOutput().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, DistributedCast_ForwardRejectsNonDistributedInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !DistType = !VPUIP.DistributedBuffer<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !DistType) -> !DistType {
                %0 = VPUIP.DistributedCast inputs(%arg0 : !DistType) -> !DistType
                return %0 : !DistType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::DistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // inferDistributedCastOutput requires the new input to be a distributed buffer; a plain memref
    // cannot be re-projected -> nullopt.
    const auto plainMemref = mlir::MemRefType::get({1, 16, 112, 112}, mlir::Float16Type::get(&ctx));
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, plainMemref).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, DistributedCast_ReverseIdenticalDistribution) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !DistType = !VPUIP.DistributedBuffer<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !DistType) -> !DistType {
                %0 = VPUIP.DistributedCast inputs(%arg0 : !DistType) -> !DistType
                return %0 : !DistType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::DistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getOutput().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getInput().getType());
}

TEST_F(MLIR_VPUIP_BackInfer, DistributedCast_ForwardSparseBuffer) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !DataType = !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        !SMapType = !VPUIP.DistributedBuffer<1x16x32x32xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        !SparseType = !VPUIP.SparseBuffer<data=!DataType, sparsity_map=!SMapType>
        module @test {
            func.func @main(%arg0: !SparseType) -> !SparseType {
                %0 = VPUIP.DistributedCast inputs(%arg0 : !SparseType) -> !SparseType
                return %0 : !SparseType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::DistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getInput().getType());
    ASSERT_TRUE(inferred.has_value());
    auto inferredSparse = mlir::dyn_cast<VPUIP::SparseBufferType>(inferred.value());
    auto outputSparse = mlir::cast<VPUIP::SparseBufferType>(op.getOutput().getType());
    ASSERT_NE(inferredSparse, nullptr);
    expectSameNDType(inferred.value(), op.getOutput().getType());
    EXPECT_EQ(inferredSparse.getData(), outputSparse.getData());
    EXPECT_EQ(inferredSparse.getSparsityMap(), outputSparse.getSparsityMap());
}

TEST_F(MLIR_VPUIP_BackInfer, DistributedCast_ReverseSparseBuffer) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !DataType = !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        !SMapType = !VPUIP.DistributedBuffer<1x16x32x32xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        !SparseType = !VPUIP.SparseBuffer<data=!DataType, sparsity_map=!SMapType>
        module @test {
            func.func @main(%arg0: !SparseType) -> !SparseType {
                %0 = VPUIP.DistributedCast inputs(%arg0 : !SparseType) -> !SparseType
                return %0 : !SparseType
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::DistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getOutput().getType());
    ASSERT_TRUE(inferred.has_value());
    auto inferredSparse = mlir::dyn_cast<VPUIP::SparseBufferType>(inferred.value());
    auto inputSparse = mlir::cast<VPUIP::SparseBufferType>(op.getInput().getType());
    ASSERT_NE(inferredSparse, nullptr);
    expectSameNDType(inferred.value(), op.getInput().getType());
    EXPECT_EQ(inferredSparse.getData(), inputSparse.getData());
    EXPECT_EQ(inferredSparse.getSparsityMap(), inputSparse.getSparsityMap());
}

//
// NonDistributedCastOp: distributed (DUPLICATED/SEGMENTED|MULTICASTED) input <-> plain memref.
//

TEST_F(MLIR_VPUIP_BackInfer, NonDistributedCast_ForwardToPlainMemref) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> {
                %0 = VPUIP.NonDistributedCastOp inputs(%arg0 : !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
                return %0 : memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::NonDistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward with the original DUPLICATED input reproduces the plain memref output.
    const auto inferred = VPUIP::BackInferUtils::inferOutputType(op, op.getInput().getType());
    ASSERT_TRUE(inferred.has_value());
    expectSameNDType(inferred.value(), op.getOutput().getType());
    EXPECT_FALSE(mlir::isa<VPUIP::DistributedBufferType>(inferred.value()));
}

TEST_F(MLIR_VPUIP_BackInfer, NonDistributedCast_ForwardRejectsNonDistributedInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> {
                %0 = VPUIP.NonDistributedCastOp inputs(%arg0 : !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
                return %0 : memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::NonDistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Forward inference requires the new input to be a distributed buffer -> a plain memref input is rejected.
    const auto plainMemref = mlir::cast<mlir::Type>(op.getOutput().getType());
    EXPECT_FALSE(VPUIP::BackInferUtils::inferOutputType(op, plainMemref).has_value());
}

TEST_F(MLIR_VPUIP_BackInfer, NonDistributedCast_ReverseInfersDistributedInput) {
    constexpr llvm::StringLiteral ir = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        module @test {
            func.func @main(%arg0: !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> {
                %0 = VPUIP.NonDistributedCastOp inputs(%arg0 : !InType) -> memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
                return %0 : memref<1x128x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        }
    )";
    auto module = parse(ir);
    ASSERT_TRUE(module.get() != nullptr);
    auto op = findFirstOp<VPUIP::NonDistributedCastOp>(module.get());
    ASSERT_NE(op, nullptr);

    // Reverse from the plain memref output reconstructs a distributed input carrying the original
    // op input's distribution; the round-trip guard accepts it. (The reconstructed mem space follows
    // the output's per-cluster mem space, so only shape/elem/order/distribution are asserted here.)
    const auto inferred = VPUIP::BackInferUtils::reverseInferInputType(op, op.getOutput().getType());
    ASSERT_TRUE(inferred.has_value());
    auto inferredDist = mlir::dyn_cast<VPUIP::DistributedBufferType>(inferred.value());
    ASSERT_NE(inferredDist, nullptr);
    auto origInputDist = mlir::cast<VPUIP::DistributedBufferType>(op.getInput().getType());
    EXPECT_EQ(inferredDist.getShape(), origInputDist.getShape());
    EXPECT_EQ(inferredDist.getElementType(), origInputDist.getElementType());
    EXPECT_EQ(inferredDist.getDimsOrder(), origInputDist.getDimsOrder());
    EXPECT_EQ(inferredDist.getDistribution(), origInputDist.getDistribution());
}
