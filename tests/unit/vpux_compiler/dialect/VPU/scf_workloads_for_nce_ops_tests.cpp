//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

// ---------------------------------------------------------------------------
// Test fixture — lightweight base (fresh context per test helper call)
// ---------------------------------------------------------------------------

using MLIR_SCFWorkloadTest = MLIR_UnitBase;

namespace {

/// Helper: parse IR, set up platform/resources, run cost-model-construct +
/// create-workloads-for-nce-ops-scf, walk NCE ops and count DPU.Workload ops.
struct WorkloadResult {
    int64_t numWorkloads = 0;
    bool hasScfForInWorkloads = false;
    bool hasScfIfInWorkloads = false;
};

WorkloadResult runPassAndCountWorkloads(llvm::StringRef inputIR) {
    // Create fresh registry + context per invocation (pattern from
    // vpunn_cost_model_analysis_test.cpp — avoids singleton conflicts)
    const auto platform = config::Platform::NPU4000;
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, platform);
    VPU::registerStrategies(registry, platform);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    EXPECT_TRUE(module.get() != nullptr);
    if (!module.get()) {
        return {};
    }

    // Add platform + resources via InitCompiler pipeline
    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 6;
    initCompilerOptions.allowCustomValues = true;
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    // Pre-compute cost model analysis so the workloads pass finds it cached
    pm.addPass(VPU::createCostModelAnalysisConstructPass(vpux::Logger::global()));
    pm.addNestedPass<mlir::func::FuncOp>(VPU::createWorkloadsForNCEOpsSCFPass());

    auto result = pm.run(module.get());
    EXPECT_TRUE(mlir::succeeded(result));
    if (mlir::failed(result)) {
        return {};
    }

    WorkloadResult wr;
    module.get()->walk([&](VPU::NCEOpInterface nceOp) {
        auto& workloads = nceOp.getWorkloads();
        if (workloads.empty()) {
            return;
        }
        workloads.walk([&](VPU::DPUWorkloadOp) {
            ++wr.numWorkloads;
        });
        workloads.walk([&](mlir::scf::ForOp) {
            wr.hasScfForInWorkloads = true;
        });
        workloads.walk([&](mlir::scf::IfOp) {
            wr.hasScfIfInWorkloads = true;
        });
    });

    return wr;
}

}  // namespace

// ---------------------------------------------------------------------------
// NCE.Convolution — static output inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvStaticInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %conv into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                            : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
    EXPECT_FALSE(wr.hasScfIfInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Convolution — NOT inside scf.forall (should remain untouched)
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvOutsideForallNotTouched) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    strides = [1, 1]
                } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                  -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                return %conv : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_EQ(wr.numWorkloads, 0);
}

// ---------------------------------------------------------------------------
// NCE.Convolution — dynamic height (bounded), inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvDynamicHeightInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        !OutType = tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: !InType,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<1x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<1x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : !InType, tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !OutType
                    %cast = tensor.cast %conv : !OutType to tensor<1x32x11x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %h_offset = affine.apply affine_map<(d0) -> (d0 * 11)>(%tid)
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %cast into %o[0, 0, %h_offset, 0] [1, 32, 11, 16] [1, 1, 1, 1]
                            : tensor<1x32x11x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<1x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<1x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
}

// ---------------------------------------------------------------------------
// NCE.DepthConvolution — static output inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, DepthConvStaticInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %dw = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [64, 1, 3, 3] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %dw into %o[%tid, 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
                            : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.MaxPool — dynamic channel inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, MaxPoolDynamicChannelInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !BoundedType = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: !BoundedType)
                    -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %pool = VPU.NCE.MaxPool(%arg0) {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } : !BoundedType -> !BoundedType
                    %cast = tensor.cast %pool : !BoundedType
                                              to tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %c_offset = affine.apply affine_map<(d0) -> (d0 * 48)>(%tid)
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %cast into %o[0, %c_offset, 0, 0] [1, 48, 16, 16] [1, 1, 1, 1]
                            : tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_TRUE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.MaxPool — static output inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, MaxPoolStaticInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %pool = VPU.NCE.MaxPool(%arg0) {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %pool into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                            : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.MaxPool — dynamic channel with scf.for tiling (multi-shape -> scf.if)
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, MaxPoolDynamicChannelScfForTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InBounded = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        !OutBounded = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        !AccType = tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> !AccType {
                %acc = tensor.empty() : !AccType
                %c0 = arith.constant 0 : index
                %c256 = arith.constant 256 : index
                %c640 = arith.constant 640 : index
                %result = scf.for %iv = %c0 to %c640 step %c256 iter_args(%out = %acc) -> (!AccType) {
                    %tile_sz = affine.min affine_map<(d0) -> (256, 640 - d0)>(%iv)
                    %slice = tensor.extract_slice %arg0[0, %iv, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
                        : tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to !InBounded
                    %pool = VPU.NCE.MaxPool(%slice) {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } : !InBounded -> !OutBounded
                    %ins = tensor.insert_slice %pool into %out[0, %iv, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
                        : !OutBounded into !AccType
                    scf.yield %ins : !AccType
                }
                return %result : !AccType
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    // scf.for over 640 channels with step=256 -> tiles {256, 256, 128}
    // Multiple unique shapes -> scf.if in workloads region
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_TRUE(wr.hasScfIfInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Eltwise — static output inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, EltwiseStaticInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %elt = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        op_type = #VPU.eltwise_type<ADD>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
                    } -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %elt into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                            : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Eltwise — dynamic height inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, EltwiseDynamicHeightInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !BoundedType = tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 8, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: !BoundedType, %arg1: !BoundedType)
                    -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %elt = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        op_type = #VPU.eltwise_type<ADD>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
                    } -> !BoundedType
                    %cast = tensor.cast %elt : !BoundedType
                                             to tensor<1x32x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %h_offset = affine.apply affine_map<(d0) -> (d0 * 8)>(%tid)
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %cast into %o[0, 0, %h_offset, 0] [1, 32, 8, 16] [1, 1, 1, 1]
                            : tensor<1x32x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
}

// ---------------------------------------------------------------------------
// NCE.Convolution — SOK multiclustering inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvSOKInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %offset_c = affine.apply affine_map<(d0) -> (d0 * 32)>(%tid)
                    %w_slice = tensor.extract_slice %weights[%offset_c, 0, 0, 0] [32, 16, 1, 1] [1, 1, 1, 1]
                        : tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [32, 16, 1, 1] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %conv into %o[0, %offset_c, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                            : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.MaxPool — inside scf.for (not scf.forall), static shapes
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, MaxPoolInsideScfForStatic) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %c0 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %acc = tensor.empty() : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.for %iv = %c0 to %c32 step %c16 iter_args(%out_arg = %acc)
                        -> (tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, 16, 32] [1, 1, 1, 1]
                        : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %pool = VPU.NCE.MaxPool(%slice) {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1],
                        kernel_size = [1, 1]
                    } : tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %ins = tensor.insert_slice %pool into %out_arg[0, 0, %iv, 0] [1, 64, 16, 32] [1, 1, 1, 1]
                        : tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        into tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.yield %ins : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                }
                return %result : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
    EXPECT_FALSE(wr.hasScfIfInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Eltwise — inside scf.for with static shapes
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, EltwiseInsideScfForStatic) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %arg1: tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %c0 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %acc = tensor.empty() : tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.for %iv = %c0 to %c32 step %c16 iter_args(%out_arg = %acc)
                        -> (tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %slice0 = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                        : tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %slice1 = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                        : tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %elt = VPU.NCE.Eltwise(%slice0, %slice1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        op_type = #VPU.eltwise_type<ADD>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
                    } -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %ins = tensor.insert_slice %elt into %out_arg[0, 0, %iv, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                        : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        into tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.yield %ins : tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                }
                return %result : tensor<1x32x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
    EXPECT_FALSE(wr.hasScfIfInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Convolution with 3x3 kernel and padding inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvWithPaddingInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 3, 3] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %conv into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                            : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.DepthConvolution — static, large channel count (multi-workload split)
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, DepthConvLargeChannelsSplit) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %dw = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [128, 1, 3, 3] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %dw into %o[%tid, 0, 0, 0] [1, 128, 16, 16] [1, 1, 1, 1]
                            : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 1);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.MaxPool — 3x3 kernel with strides, inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, MaxPool3x3StridedInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %pool = VPU.NCE.MaxPool(%arg0) {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [2, 2],
                        kernel_size = [3, 3]
                    } : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %pool into %o[%tid, 0, 0, 0] [1, 64, 15, 15] [1, 1, 1, 1]
                            : tensor<1x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x64x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.Convolution — dynamic height inside scf.for (uniform tile shapes)
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, ConvDynamicHeightScfForUniformTiles) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InBounded = tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        !OutBounded = tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: tensor<1x16x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %acc = tensor.empty() : tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %c0 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %c48 = arith.constant 48 : index
                %result = scf.for %iv = %c0 to %c48 step %c16 iter_args(%out = %acc)
                        -> (tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %tile_sz = affine.min affine_map<(d0) -> (16, 48 - d0)>(%iv)
                    %slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 16, %tile_sz, 16] [1, 1, 1, 1]
                        : tensor<1x16x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                        to !InBounded
                    %conv = VPU.NCE.Convolution(%slice, %weights) rawFilterShape [32, 16, 1, 1] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : !InBounded, tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !OutBounded
                    %ins = tensor.insert_slice %conv into %out[0, 0, %iv, 0] [1, 32, %tile_sz, 16] [1, 1, 1, 1]
                        : !OutBounded into tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.yield %ins : tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                }
                return %result : tensor<1x32x48x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    // scf.for over H: 48/16=3 uniform tiles -> single shape, no scf.if
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfIfInWorkloads);
}

// ---------------------------------------------------------------------------
// NCE.DepthConvolution — dynamic height inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, DepthConvDynamicHeightInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        !InType = tensor<1x64x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        !OutType = tensor<1x64x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        module @test {
            func.func @main(%arg0: !InType,
                            %weights: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %dw = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [64, 1, 3, 3] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : !InType, tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !OutType
                    %cast = tensor.cast %dw : !OutType to tensor<1x64x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    %h_offset = affine.apply affine_map<(d0) -> (d0 * 8)>(%tid)
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %cast into %o[0, 0, %h_offset, 0] [1, 64, 8, 16] [1, 1, 1, 1]
                            : tensor<1x64x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
}

// ---------------------------------------------------------------------------
// NCE.Convolution — 1x1 conv with large output channels inside scf.forall
// ---------------------------------------------------------------------------

TEST_F(MLIR_SCFWorkloadTest, Conv1x1LargeOCInsideForall) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x64x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<256x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                    -> tensor<2x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                %out = tensor.empty() : tensor<2x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %result = scf.forall (%tid) = (0) to (2) step (1)
                    shared_outs(%o = %out) -> (tensor<2x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
                    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [256, 64, 1, 1] {
                        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                        strides = [1, 1]
                    } : tensor<1x64x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                        tensor<256x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                      -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %conv into %o[%tid, 0, 0, 0] [1, 256, 8, 8] [1, 1, 1, 1]
                            : tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                            into tensor<2x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    }
                }
                return %result : tensor<2x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }
    )";

    auto wr = runPassAndCountWorkloads(inputIR);
    EXPECT_GT(wr.numWorkloads, 0);
    EXPECT_FALSE(wr.hasScfForInWorkloads);
}
