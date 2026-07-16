//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <common_test_utils/file_utils.hpp>

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

    // Loads IR from a file co-located with the test binary.
    // Pass a bare filename, e.g. "op_ranges_dma.mlir".
    void initFromFile(std::string_view filename) {
        const auto path = (std::filesystem::path(ov::test::utils::getExecutableDirectory()) /
                           "vpux_compiler/dialect/VPUMI40XX/input_ir" / filename)
                                  .string();
        module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
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
        module @EmptyOpRanges attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU4000>} {
        config.Resources 6 of @NCE at 1.700000e+03 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
            config.ExecutorResource 2 of @SHAVE_ACT
            config.ExecutorResource 1 of @DPU
        }
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

TEST_F(MLIR_TaskRangeTest, DMA) {
    initFromFile("op_ranges_dma.mlir");

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
    initFromFile("op_ranges_shave.mlir");

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
    initFromFile("op_ranges_dpu.mlir");

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
