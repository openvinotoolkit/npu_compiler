//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using namespace vpux;
using namespace VPUMI40XX;
using namespace VPURegMapped;

namespace {

template <class Range>
size_t size(Range range) {
    size_t result = 0;
    for ([[maybe_unused]] auto&& _ : range) {
        ++result;
    }
    return result;
}

class MLIR_TaskRangeTest : public MLIR_UnitBase {
public:
    MLIR_TaskRangeTest(): context(registry) {
    }

    void init(std::string_view ir) {
        module = mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
        ASSERT_TRUE(module.get() != nullptr);

        function = module.get().lookupSymbol<mlir::func::FuncOp>("main");
        ASSERT_TRUE(function != nullptr);
    }

    OpRanges getRanges() {
        assert(function);
        return mlir::cast<OpRanges>(function.getBlocks().front().getTerminator());
    }

    IndexType getIndex(uint32_t tile, uint32_t list) {
        return IndexType::get(&context, tile, list, 0);
    }

    auto getReferenceForwardRange(TaskType type, uint32_t tile, uint32_t list) {
        const auto isIndexCompatible = [&](auto index) {
            return tile == index.getTileIdx() && list == index.getListIdx();
        };
        const auto isInRange = [&](auto task) {
            return type == task.getTaskType() && isIndexCompatible(task.getIndexType());
        };
        return to_small_vector(function.getOps<TaskOpInterface>() | filtered(isInRange));
    }

    auto getReferenceBackwardRange(TaskType type, uint32_t tile, uint32_t list) {
        auto reversedRange = getReferenceForwardRange(type, tile, list);
        std::reverse(::std::begin(reversedRange), ::std::end(reversedRange));
        return reversedRange;
    }

    mlir::SmallVector<std::pair<TaskType, IndexType>> getAllPossibleRanges() {
        // keep vector in automatic storage to be re-initialized on each call
        // mlir::Type storage is specific to mlir::ModuleOp which is re-assigned on each test
        // making allRanges static would keep IndexTypes from mlir::ModuleOp of the 1st test
        // after mlir::ModuleOp is re-assigned for the 2nd test, all indexes dangle
        mlir::SmallVector<std::pair<TaskType, IndexType>> allRanges = {
                {TaskType::DMA, getIndex(0, 0)},
                {TaskType::DMA, getIndex(0, 1)},
                {TaskType::DMA, getIndex(1, 0)},
                {TaskType::DMA, getIndex(1, 1)},
                {TaskType::DPUInvariant, getIndex(0, 0)},
                {TaskType::DPUInvariant, getIndex(1, 0)},
                {TaskType::DPUInvariant, getIndex(2, 0)},
                {TaskType::DPUInvariant, getIndex(3, 0)},
                {TaskType::DPUInvariant, getIndex(4, 0)},
                {TaskType::DPUInvariant, getIndex(5, 0)},
                {TaskType::DPUVariant, getIndex(0, 0)},
                {TaskType::DPUVariant, getIndex(1, 0)},
                {TaskType::DPUVariant, getIndex(2, 0)},
                {TaskType::DPUVariant, getIndex(3, 0)},
                {TaskType::DPUVariant, getIndex(4, 0)},
                {TaskType::DPUVariant, getIndex(5, 0)},
                {TaskType::ActKernelInvocation, getIndex(0, 0)},
                {TaskType::ActKernelInvocation, getIndex(1, 0)},
                {TaskType::ActKernelInvocation, getIndex(2, 0)},
                {TaskType::ActKernelInvocation, getIndex(3, 0)},
                {TaskType::ActKernelInvocation, getIndex(4, 0)},
                {TaskType::ActKernelInvocation, getIndex(5, 0)},
                {TaskType::ActKernelRange, getIndex(0, 0)},
                {TaskType::ActKernelRange, getIndex(1, 0)},
                {TaskType::ActKernelRange, getIndex(2, 0)},
                {TaskType::ActKernelRange, getIndex(3, 0)},
                {TaskType::ActKernelRange, getIndex(4, 0)},
                {TaskType::ActKernelRange, getIndex(5, 0)},
        };
        return allRanges;
    }

    template <TaskType... targets>
    mlir::SmallVector<std::pair<TaskType, IndexType>> getAllTargetRanges() {
        // llvm::make_filter_range used in vpux::operator| with vpux::details::FilterRangeTag
        // doesn't support rvalue references; enforce lvalue to get code below to compile
        // see https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/STLExtras.h#L566
        auto lvalue = getAllPossibleRanges();
        return vpux::to_small_vector(lvalue | vpux::filtered([](auto range) {
                                         return ((range.first == targets) || ...);
                                     }));
    }

public:
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp function;
};

TEST_F(MLIR_TaskRangeTest, Empty) {
    constexpr std::string_view inputIR = R"(
        module @EmptyOpRanges attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
        config.Resources 6 of @NCE at 1.700000e+03 MHz {
            config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
            config.ExecutorResource 2 of @SHAVE_ACT
            config.ExecutorResource 1 of @DPU
        }
        config.ExecutorResource 1 of @M2I
        config.ExecutorResource 2 of @DMA_NN
        config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
        net.NetworkInfo entryPoint : @main inputsInfo : {
            DataInfo "input_0" : tensor<1x2x3x4xf16>
        } outputsInfo : {
            DataInfo "output_0" : tensor<1x2x3x4xf16>
        }
        func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
            %0 = VPUMI40XX.MappedInference dmaCount([[0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
            VPUMI40XX.OpRanges
        }
        }
    )";

    init(inputIR);
    for (auto [type, index] : getAllPossibleRanges()) {
        ASSERT_TRUE(getRanges().getForwardRange(type, index).empty());
        ASSERT_TRUE(getRanges().getBackwardRange(type, index).empty());
    }
}

// test fixtures below are splitted intentionally
// to decrease peak string literal length
// if they will be merged with single literal with
// all task types (DMA, Shave, DPU), then build on MSVC
// fail with:
// "error C2026: string too big, trailing characters truncated"

TEST_F(MLIR_TaskRangeTest, DMA) {
    constexpr std::string_view inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @MultiOpRanges attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
            config.Resources 6 of @NCE at 1.700000e+03 MHz {
                config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
                config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
                config.ExecutorResource 2 of @SHAVE_ACT
                config.ExecutorResource 1 of @DPU
            }
            config.ExecutorResource 1 of @M2I
            config.ExecutorResource 2 of @DMA_NN
            config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
            net.NetworkInfo entryPoint : @main inputsInfo : {
                DataInfo "input_0" : tensor<1x2x3x4xf16>
            } outputsInfo : {
                DataInfo "output_0" : tensor<1x2x3x4xf16>
            }
            func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
                %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
                %1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
                %2 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%1 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
                %3 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
                %4 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%3 : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
                %5 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
                %6 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%5 : !VPURegMapped.Index<1:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
                %7 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:0>
                %8 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<1:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:1>
                VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DMA>]) begins(%1, %3, %5, %7 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>) ends(%2, %4, %6, %8 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:1:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<1:1:1>)
            }
        }
    )";

    init(inputIR);

    for (auto [type, index] : getAllTargetRanges<TaskType::DMA>()) {
        auto forwardReference = getReferenceForwardRange(type, index.getTileIdx(), index.getListIdx());
        auto forwardRange = getRanges().getForwardRange(type, index);
        ASSERT_THAT(forwardReference,
                    ::testing::ElementsAreArray(::std::begin(forwardRange), ::std::end(forwardRange)));

        auto backwardReference = getReferenceBackwardRange(type, index.getTileIdx(), index.getListIdx());
        auto backwardRange = getRanges().getBackwardRange(type, index);
        ASSERT_THAT(backwardReference,
                    ::testing::ElementsAreArray(::std::begin(backwardRange), ::std::end(backwardRange)));
    }
}

TEST_F(MLIR_TaskRangeTest, Shave) {
    constexpr std::string_view inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @MultiOpRanges attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
            config.Resources 6 of @NCE at 1.700000e+03 MHz {
                config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
                config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
                config.ExecutorResource 2 of @SHAVE_ACT
                config.ExecutorResource 1 of @DPU
            }
            config.ExecutorResource 1 of @M2I
            config.ExecutorResource 2 of @DMA_NN
            config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
            net.NetworkInfo entryPoint : @main inputsInfo : {
                DataInfo "input_0" : tensor<1x2x3x4xf16>
            } outputsInfo : {
                DataInfo "output_0" : tensor<1x2x3x4xf16>
            }
            module @VPU.SW {
                func.func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
            }
            func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
                %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
                %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
                %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
                %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
                %14 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>
                %15 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
                %16 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
                %40 = VPUMI40XX.KernelParams inputs(%0 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) kernel_type("activation_softmax") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>
                %41 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
                %42 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
                %43 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
                %44 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
                %45 = VPUMI40XX.ActKernelInvocation range_index(%44 : <0:0:0>) kernel_params(%40 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
                %48 = VPUMI40XX.ActKernelRange previousTask(%44 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
                %49 = VPUMI40XX.ActKernelInvocation previousTask(%45 : !VPURegMapped.Index<0:0:0>) range_index(%48 : <0:0:1>) kernel_params(%40 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
                %52 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
                %53 = VPUMI40XX.ActKernelInvocation range_index(%52 : <1:0:0>) kernel_params(%40 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
                %56 = VPUMI40XX.ActKernelRange previousTask(%52 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
                %57 = VPUMI40XX.ActKernelInvocation previousTask(%53 : !VPURegMapped.Index<1:0:0>) range_index(%56 : <1:0:1>) kernel_params(%40 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
                %60 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<2:0:0>
                %61 = VPUMI40XX.ActKernelInvocation range_index(%60 : <2:0:0>) kernel_params(%40 : <0:0:0>) tile(2) start_after(0) clean_after(0) -> !VPURegMapped.Index<2:0:0>
                %64 = VPUMI40XX.ActKernelRange previousTask(%60 : !VPURegMapped.Index<2:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<2:0:1>
                %65 = VPUMI40XX.ActKernelInvocation previousTask(%61 : !VPURegMapped.Index<2:0:0>) range_index(%64 : <2:0:1>) kernel_params(%40 : <0:0:0>) tile(2) start_after(0) clean_after(0) -> !VPURegMapped.Index<2:0:1>
                %68 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<3:0:0>
                %69 = VPUMI40XX.ActKernelInvocation range_index(%68 : <3:0:0>) kernel_params(%40 : <0:0:0>) tile(3) start_after(0) clean_after(0) -> !VPURegMapped.Index<3:0:0>
                %72 = VPUMI40XX.ActKernelRange previousTask(%68 : !VPURegMapped.Index<3:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<3:0:1>
                %73 = VPUMI40XX.ActKernelInvocation previousTask(%69 : !VPURegMapped.Index<3:0:0>) range_index(%72 : <3:0:1>) kernel_params(%40 : <0:0:0>) tile(3) start_after(0) clean_after(0) -> !VPURegMapped.Index<3:0:1>
                %76 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<4:0:0>
                %77 = VPUMI40XX.ActKernelInvocation range_index(%76 : <4:0:0>) kernel_params(%40 : <0:0:0>) tile(4) start_after(0) clean_after(0) -> !VPURegMapped.Index<4:0:0>
                %80 = VPUMI40XX.ActKernelRange previousTask(%76 : !VPURegMapped.Index<4:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<4:0:1>
                %81 = VPUMI40XX.ActKernelInvocation previousTask(%77 : !VPURegMapped.Index<4:0:0>) range_index(%80 : <4:0:1>) kernel_params(%40 : <0:0:0>) tile(4) start_after(0) clean_after(0) -> !VPURegMapped.Index<4:0:1>
                %84 = VPUMI40XX.ActKernelRange kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<5:0:0>
                %85 = VPUMI40XX.ActKernelInvocation range_index(%84 : <5:0:0>) kernel_params(%40 : <0:0:0>) tile(5) start_after(0) clean_after(0) -> !VPURegMapped.Index<5:0:0>
                %88 = VPUMI40XX.ActKernelRange previousTask(%84 : !VPURegMapped.Index<5:0:0>) kernel_text_index(%41 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%43 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%42 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<5:0:1>
                %89 = VPUMI40XX.ActKernelInvocation previousTask(%85 : !VPURegMapped.Index<5:0:0>) range_index(%88 : <5:0:1>) kernel_params(%40 : <0:0:0>) tile(5) start_after(0) clean_after(0) -> !VPURegMapped.Index<5:0:1>
                VPUMI40XX.OpRanges types([#VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>, #VPURegMapped.task_type<ActKernelRange>, #VPURegMapped.task_type<ActKernelInvocation>]) begins(%44, %45, %52, %53, %60, %61, %68, %69, %76, %77, %84, %85 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>, !VPURegMapped.Index<5:0:0>) ends(%48, %49, %56, %57, %64, %65, %72, %73, %80, %81, %88, %89 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<5:0:1>, !VPURegMapped.Index<5:0:1>)
            }
        }
    )";

    init(inputIR);

    for (auto [type, index] : getAllTargetRanges<TaskType::ActKernelInvocation, TaskType::ActKernelRange>()) {
        auto forwardReference = getReferenceForwardRange(type, index.getTileIdx(), index.getListIdx());
        auto forwardRange = getRanges().getForwardRange(type, index);
        ASSERT_THAT(forwardReference,
                    ::testing::ElementsAreArray(::std::begin(forwardRange), ::std::end(forwardRange)));

        auto backwardReference = getReferenceBackwardRange(type, index.getTileIdx(), index.getListIdx());
        auto backwardRange = getRanges().getBackwardRange(type, index);
        ASSERT_THAT(backwardReference,
                    ::testing::ElementsAreArray(::std::begin(backwardRange), ::std::end(backwardRange)));
    }
}

TEST_F(MLIR_TaskRangeTest, DPU) {
    constexpr std::string_view inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @MultiOpRanges attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
        config.Resources 6 of @NCE at 1.700000e+03 MHz {
            config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
            config.ExecutorResource 2 of @SHAVE_ACT
            config.ExecutorResource 1 of @DPU
        }
        config.ExecutorResource 1 of @M2I
        config.ExecutorResource 2 of @DMA_NN
        config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
        net.NetworkInfo entryPoint : @main inputsInfo : {
            DataInfo "input_0" : tensor<1x2x3x4xf16>
        } outputsInfo : {
            DataInfo "output_0" : tensor<1x2x3x4xf16>
        }
        func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
            %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
            %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
            %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
            %16 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
            %17 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
            }
            %18 = VPUMI40XX.DPUVariant calls(%17 : <0:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <0:0:0>
            %19 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%17 : !VPURegMapped.Index<0:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:1> PPE : {
            }
            %20 = VPUMI40XX.DPUVariant previousTask(%18 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <0:0:1>
            %21 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <1:0:0> PPE : {
            }
            %22 = VPUMI40XX.DPUVariant calls(%21 : <1:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 1 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <1:0:0>
            %23 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%21 : !VPURegMapped.Index<1:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <1:0:1> PPE : {
            }
            %24 = VPUMI40XX.DPUVariant previousTask(%22 : !VPURegMapped.Index<1:0:0>) calls(%23 : <1:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 1 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <1:0:1>
            %25 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <2:0:0> PPE : {
            }
            %26 = VPUMI40XX.DPUVariant calls(%25 : <2:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 2 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <2:0:0>
            %27 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%25 : !VPURegMapped.Index<2:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <2:0:1> PPE : {
            }
            %28 = VPUMI40XX.DPUVariant previousTask(%26 : !VPURegMapped.Index<2:0:0>) calls(%27 : <2:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 2 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <2:0:1>
            %29 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <3:0:0> PPE : {
            }
            %30 = VPUMI40XX.DPUVariant calls(%29 : <3:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 3 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <3:0:0>
            %31 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%29 : !VPURegMapped.Index<3:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <3:0:1> PPE : {
            }
            %32 = VPUMI40XX.DPUVariant previousTask(%30 : !VPURegMapped.Index<3:0:0>) calls(%31 : <3:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 3 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <3:0:1>
            %33 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <4:0:0> PPE : {
            }
            %34 = VPUMI40XX.DPUVariant calls(%33 : <4:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 4 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <4:0:0>
            %35 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%33 : !VPURegMapped.Index<4:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <4:0:1> PPE : {
            }
            %36 = VPUMI40XX.DPUVariant previousTask(%34 : !VPURegMapped.Index<4:0:0>) calls(%35 : <4:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 4 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <4:0:1>
            %37 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <5:0:0> PPE : {
            }
            %38 = VPUMI40XX.DPUVariant calls(%37 : <5:0:0>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 5 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <5:0:0>
            %39 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%37 : !VPURegMapped.Index<5:0:0>) input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <5:0:1> PPE : {
            }
            %40 = VPUMI40XX.DPUVariant previousTask(%38 : !VPURegMapped.Index<5:0:0>) calls(%39 : <5:0:1>) weights(%11 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {cluster_id = 5 : ui64, end = [31, 31, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} -> <5:0:1>
            VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>]) begins(%17, %18, %21, %22, %25, %26, %29, %30, %33, %34, %37, %38 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<3:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<4:0:0>, !VPURegMapped.Index<5:0:0>, !VPURegMapped.Index<5:0:0>) ends(%19, %20, %23, %24, %27, %28, %31, %32, %35, %36, %39, %40 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<1:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<2:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<3:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<4:0:1>, !VPURegMapped.Index<5:0:1>, !VPURegMapped.Index<5:0:1>)
        }
        })";

    init(inputIR);

    for (auto [type, index] : getAllTargetRanges<TaskType::DPUInvariant, TaskType::DPUVariant>()) {
        auto forwardReference = getReferenceForwardRange(type, index.getTileIdx(), index.getListIdx());
        auto forwardRange = getRanges().getForwardRange(type, index);
        ASSERT_THAT(forwardReference,
                    ::testing::ElementsAreArray(::std::begin(forwardRange), ::std::end(forwardRange)));

        auto backwardReference = getReferenceBackwardRange(type, index.getTileIdx(), index.getListIdx());
        auto backwardRange = getRanges().getBackwardRange(type, index);
        ASSERT_THAT(backwardReference,
                    ::testing::ElementsAreArray(::std::begin(backwardRange), ::std::end(backwardRange)));
    }
}

}  // namespace
