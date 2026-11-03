//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

using MLIR_NCEFwdInferTest = vpux::VPU::arch40xx::UnitTest;

namespace {

/// Parse @p inputIR and return the first op of type ConcreteOp, keeping the module alive in @p module.
template <typename ConcreteOp>
ConcreteOp parseFirstOp(mlir::MLIRContext& ctx, llvm::StringRef inputIR, mlir::OwningOpRef<mlir::ModuleOp>& module) {
    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    if (module.get() == nullptr) {
        return nullptr;
    }

    ConcreteOp found = nullptr;
    module.get()->walk([&](ConcreteOp op) {
        if (found == nullptr) {
            found = op;
        }
    });
    return found;
}

/// Forward-infer through the TilingBuilder interface method (the entry point the SCF workload pass
/// uses) from a single activation tile, and return the resulting shape for easy comparison.
SmallVector<int64_t> fwdInferShape(mlir::Operation* op, ArrayRef<int64_t> inputTileShape, bool& succeeded) {
    // TileInfo takes a ShapeRef and the rvalue conversion from Shape is deleted, so the Shape has to
    // outlive the TileInfo as a named local.
    const Shape inShape(inputTileShape);
    const SmallVector<vpux::TileInfo> inputTiles{vpux::TileInfo(inShape)};

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (tilingBuilder == nullptr) {
        succeeded = false;
        return {};
    }

    const auto outputTile = tilingBuilder.fwdInferOutputTile(inputTiles, vpux::Logger::global());
    succeeded = mlir::succeeded(outputTile);
    if (!succeeded) {
        return {};
    }
    return SmallVector<int64_t>(outputTile->shape.raw());
}

}  // namespace

// ---------------------------------------------------------------------------
// NCE ops without an ODU transform
// ---------------------------------------------------------------------------

// A 1x1 stride-1 convolution keeps H/W and takes the output channel count from the weights, so a
// half-height input tile maps to a half-height output tile that still has all 32 output channels.
TEST_F(MLIR_NCEFwdInferTest, ConvTakesChannelsFromWeights) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    strides = [1, 1]
                } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                  -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                return %0 : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto convOp = parseFirstOp<VPU::NCEConvolutionOp>(ctx, inputIR, module);
    ASSERT_TRUE(convOp != nullptr);

    bool succeeded = false;
    const auto outShape = fwdInferShape(convOp.getOperation(), {1, 16, 8, 16}, succeeded);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(outShape, SmallVector<int64_t>({1, 32, 8, 16}));
}

// A 3x3 stride-2 convolution with 1-pixel padding halves H/W: (8 - 3 + 1 + 1) / 2 + 1 = 4.
TEST_F(MLIR_NCEFwdInferTest, ConvAppliesStrideAndPadWindow) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 3, 3] {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    strides = [2, 2]
                } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
                  -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                return %0 : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto convOp = parseFirstOp<VPU::NCEConvolutionOp>(ctx, inputIR, module);
    ASSERT_TRUE(convOp != nullptr);

    bool succeeded = false;
    const auto outShape = fwdInferShape(convOp.getOperation(), {1, 16, 8, 8}, succeeded);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(outShape, SmallVector<int64_t>({1, 32, 4, 4}));
}

// MaxPool has no weights and preserves channels, so a tile that narrows C must narrow the output C
// too - taking it from the declared output would overstate the tile.
TEST_F(MLIR_NCEFwdInferTest, MaxPoolTakesChannelsFromInput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %0 = VPU.NCE.MaxPool(%arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    strides = [1, 1]
                } : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                  -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                return %0 : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto poolOp = parseFirstOp<VPU::NCEMaxPoolOp>(ctx, inputIR, module);
    ASSERT_TRUE(poolOp != nullptr);

    bool succeeded = false;
    const auto outShape = fwdInferShape(poolOp.getOperation(), {1, 32, 8, 16}, succeeded);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(outShape, SmallVector<int64_t>({1, 32, 8, 16}));
}

// ---------------------------------------------------------------------------
// Default body for ops carrying the EltwiseOp trait
// ---------------------------------------------------------------------------

// The trait declares a fully parallel iteration space, so the output tile is the input tile and no
// per-op override is needed.
TEST_F(MLIR_NCEFwdInferTest, EltwiseTraitIsIdentity) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16>, %arg1: tensor<1x64x16x16xf16>)
                    -> tensor<1x64x16x16xf16> {
                %0 = VPU.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
                    : tensor<1x64x16x16xf16>, tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
                return %0 : tensor<1x64x16x16xf16>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto addOp = parseFirstOp<VPU::AddOp>(ctx, inputIR, module);
    ASSERT_TRUE(addOp != nullptr);
    ASSERT_TRUE(addOp->hasTrait<VPU::EltwiseOp>());

    bool succeeded = false;
    const auto outShape = fwdInferShape(addOp.getOperation(), {1, 64, 4, 16}, succeeded);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(outShape, SmallVector<int64_t>({1, 64, 4, 16}));
}

// Differently shaped input tiles mean a broadcast. Only equal-or-one is a legal broadcast, so the
// output extent of every dimension is the largest of the inputs here the 1x1 operand is stretched
// over the 4x16 one.
TEST_F(MLIR_NCEFwdInferTest, EltwiseBroadcastTakesPerDimensionMaximum) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16>, %arg1: tensor<1x64x1x1xf16>)
                    -> tensor<1x64x16x16xf16> {
                %0 = VPU.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
                    : tensor<1x64x16x16xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x16x16xf16>
                return %0 : tensor<1x64x16x16xf16>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto addOp = parseFirstOp<VPU::AddOp>(ctx, inputIR, module);
    ASSERT_TRUE(addOp != nullptr);

    const Shape lhsShape(SmallVector<int64_t>({1, 64, 4, 16}));
    const Shape rhsShape(SmallVector<int64_t>({1, 64, 1, 1}));
    const SmallVector<vpux::TileInfo> inputTiles{vpux::TileInfo(lhsShape), vpux::TileInfo(rhsShape)};

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(addOp.getOperation());
    ASSERT_TRUE(tilingBuilder != nullptr);
    const auto outputTile = tilingBuilder.fwdInferOutputTile(inputTiles, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(outputTile));
    EXPECT_EQ(SmallVector<int64_t>(outputTile->shape.raw()), SmallVector<int64_t>({1, 64, 4, 16}));

    // The maximum must be taken per dimension, not per operand: neither input tile is the answer
    // when each one is the larger in a different dimension.
    const Shape tallShape(SmallVector<int64_t>({1, 64, 4, 1}));
    const Shape wideShape(SmallVector<int64_t>({1, 64, 1, 16}));
    const SmallVector<vpux::TileInfo> crossTiles{vpux::TileInfo(tallShape), vpux::TileInfo(wideShape)};
    const auto crossTile = tilingBuilder.fwdInferOutputTile(crossTiles, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(crossTile));
    EXPECT_EQ(SmallVector<int64_t>(crossTile->shape.raw()), SmallVector<int64_t>({1, 64, 4, 16}));
}

// Two extents that differ and are both greater than one do not broadcast, so no output shape exists.
// Taking the maximum would invent an extent; report failure and leave the caller on its conservative
// fallback instead.
TEST_F(MLIR_NCEFwdInferTest, EltwiseWithNonBroadcastableTilesReportsFailure) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16>, %arg1: tensor<1x64x16x16xf16>)
                    -> tensor<1x64x16x16xf16> {
                %0 = VPU.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
                    : tensor<1x64x16x16xf16>, tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
                return %0 : tensor<1x64x16x16xf16>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto addOp = parseFirstOp<VPU::AddOp>(ctx, inputIR, module);
    ASSERT_TRUE(addOp != nullptr);

    const Shape lhsShape(SmallVector<int64_t>({1, 64, 4, 16}));
    const Shape rhsShape(SmallVector<int64_t>({1, 64, 3, 16}));
    const SmallVector<vpux::TileInfo> inputTiles{vpux::TileInfo(lhsShape), vpux::TileInfo(rhsShape)};

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(addOp.getOperation());
    ASSERT_TRUE(tilingBuilder != nullptr);
    EXPECT_TRUE(mlir::failed(tilingBuilder.fwdInferOutputTile(inputTiles, vpux::Logger::global())));
}

// A dynamic input tile cannot produce an inferable output: every output dimension of an NCE op is
// derived from the input extents, so the sentinel would propagate straight through the window
// formula. Fail early instead of only catching it on the output.
TEST_F(MLIR_NCEFwdInferTest, DynamicInputTileReportsFailure) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.MaxPool(%arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                    strides = [1, 1]
                } : tensor<1x64x16x16xf16, {order = #NHWC}> -> tensor<1x64x16x16xf16, {order = #NHWC}>
                return %0 : tensor<1x64x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto poolOp = parseFirstOp<VPU::NCEMaxPoolOp>(ctx, inputIR, module);
    ASSERT_TRUE(poolOp != nullptr);

    bool succeeded = false;
    fwdInferShape(poolOp.getOperation(), {1, 64, mlir::ShapedType::kDynamic, 16}, succeeded);
    EXPECT_FALSE(succeeded);
}

// ---------------------------------------------------------------------------
// getConstantTripCount - shared by CreateWorkloadsForNCEOpsSCF and FullUnrollSCFLoop
// ---------------------------------------------------------------------------

class MLIR_SCFTripCountTest : public MLIR_UnitBase {
public:
    MLIR_SCFTripCountTest() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<mlir::scf::SCFDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    }

protected:
    mlir::Operation* parseFirstLoop(llvm::StringRef inputIR) {
        module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
        if (module.get() == nullptr) {
            return nullptr;
        }

        mlir::Operation* found = nullptr;
        module.get()->walk([&](mlir::Operation* op) {
            if (found == nullptr && mlir::isa<mlir::scf::ForOp, mlir::scf::ForallOp>(op)) {
                found = op;
            }
        });
        return found;
    }

    mlir::MLIRContext ctx;
    mlir::OwningOpRef<mlir::ModuleOp> module;
};

TEST_F(MLIR_SCFTripCountTest, ForOpWithConstantBounds) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() {
                %lb = arith.constant 0 : index
                %ub = arith.constant 10 : index
                %step = arith.constant 3 : index
                scf.for %i = %lb to %ub step %step {
                }
                return
            }
        }
    )";

    auto* loopOp = parseFirstLoop(inputIR);
    ASSERT_TRUE(loopOp != nullptr);

    // divideCeil(10 - 0, 3) == 4
    const auto tripCount = VPU::getConstantTripCount(loopOp);
    ASSERT_TRUE(tripCount.has_value());
    EXPECT_EQ(tripCount.value(), 4);
}

// scf.forall has one trip count per dimension; the helper returns the size of the whole index space.
// The ForOp-only helper this replaced could not describe a forall at all.
TEST_F(MLIR_SCFTripCountTest, ForallMultipliesPerDimensionCounts) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() {
                scf.forall (%i, %j) = (0, 0) to (6, 4) step (2, 1) {
                    scf.forall.in_parallel {
                    }
                }
                return
            }
        }
    )";

    auto* loopOp = parseFirstLoop(inputIR);
    ASSERT_TRUE(loopOp != nullptr);

    // 3 iterations over the first dim, 4 over the second
    const auto tripCount = VPU::getConstantTripCount(loopOp);
    ASSERT_TRUE(tripCount.has_value());
    EXPECT_EQ(tripCount.value(), 12);
}

// A loop tiled against an extent that is only known at runtime - as in the HostCompile pipeline -
// has no compile-time trip count.
TEST_F(MLIR_SCFTripCountTest, NonConstantUpperBoundIsUnknown) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%ub: index) {
                %lb = arith.constant 0 : index
                %step = arith.constant 1 : index
                scf.for %i = %lb to %ub step %step {
                }
                return
            }
        }
    )";

    auto* loopOp = parseFirstLoop(inputIR);
    ASSERT_TRUE(loopOp != nullptr);

    EXPECT_FALSE(VPU::getConstantTripCount(loopOp).has_value());
}

// A non-positive step would not terminate; report it as unknown rather than dividing by it.
TEST_F(MLIR_SCFTripCountTest, NonPositiveStepIsUnknown) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() {
                %lb = arith.constant 0 : index
                %ub = arith.constant 10 : index
                %step = arith.constant 0 : index
                scf.for %i = %lb to %ub step %step {
                }
                return
            }
        }
    )";

    auto* loopOp = parseFirstLoop(inputIR);
    ASSERT_TRUE(loopOp != nullptr);

    EXPECT_FALSE(VPU::getConstantTripCount(loopOp).has_value());
}

// An upper bound below the lower bound means the body never runs; clamp to 0 instead of returning a
// negative count.
TEST_F(MLIR_SCFTripCountTest, UpperBoundBelowLowerBoundIsZero) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() {
                %lb = arith.constant 8 : index
                %ub = arith.constant 2 : index
                %step = arith.constant 1 : index
                scf.for %i = %lb to %ub step %step {
                }
                return
            }
        }
    )";

    auto* loopOp = parseFirstLoop(inputIR);
    ASSERT_TRUE(loopOp != nullptr);

    const auto tripCount = VPU::getConstantTripCount(loopOp);
    ASSERT_TRUE(tripCount.has_value());
    EXPECT_EQ(tripCount.value(), 0);
}

// Any op that is neither scf.for nor scf.forall - including a null pointer - yields nullopt.
TEST_F(MLIR_SCFTripCountTest, NonLoopOpIsUnknown) {
    EXPECT_FALSE(VPU::getConstantTripCount(nullptr).has_value());
}
