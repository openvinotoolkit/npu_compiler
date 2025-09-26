//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/workload_utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

struct WorkloadTestParams {
    VPUIP::WorkloadComponents inputWorkload;
    SmallVector<int64_t> kernelSize;
    VPUIP::WorkloadComponents expectedWorkload;
};

std::ostream& operator<<(std::ostream& os, const WorkloadTestParams& params) {
    return os << formatv("{ inputWorkload={0}, kernelSize={1}, expectedWorkload={2} }", params.inputWorkload,
                         params.kernelSize, params.expectedWorkload)
                         .str();
}

using MLIR_WorkloadTest = testing::TestWithParam<WorkloadTestParams>;

TEST_P(MLIR_WorkloadTest, ReduceToOneOutputPixel) {
    const auto& params = GetParam();
    const auto workload = VPUIP::reduceToOneOutputPixel(params.inputWorkload, params.kernelSize);
    EXPECT_EQ(workload.inStart, params.expectedWorkload.inStart);
    EXPECT_EQ(workload.inEnd, params.expectedWorkload.inEnd);
    EXPECT_EQ(workload.outStart, params.expectedWorkload.outStart);
    EXPECT_EQ(workload.outEnd, params.expectedWorkload.outEnd);
    EXPECT_EQ(workload.pad, params.expectedWorkload.pad);
}

namespace {
const auto noKernelSize = SmallVector<int64_t>{};
const auto threeKernelSize = SmallVector<int64_t>{3, 3};
const auto noPad = VPU::Padding(0, 0, 0, 0);
const auto onePad = VPU::Padding(1, 1, 1, 1);
const auto leftTopPad = VPU::Padding(1, 0, 1, 0);
}  // namespace

INSTANTIATE_TEST_SUITE_P(
        Simple, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{9, 9, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 15}, noPad},
                                   noKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, noPad},
                                   noKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 0}, noPad},
                                   noKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 0}, noPad}}));

INSTANTIATE_TEST_SUITE_P(
        KernelSize, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{11, 11, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 15}, noPad},
                                   threeKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{2, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, noPad},
                                   threeKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{2, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 0}, noPad},
                                   threeKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{7, 7, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 0}, noPad}}));

INSTANTIATE_TEST_SUITE_P(
        Padded, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{
                        VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{11, 11, 15},
                                                  /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 15}, onePad},
                        threeKernelSize,
                        VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{1, 1, 15},
                                                  /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, leftTopPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, onePad},
                                   threeKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{1, 1, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, leftTopPad}},
                WorkloadTestParams{
                        VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 15},
                                                  /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 0}, onePad},
                        threeKernelSize,
                        VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{6, 6, 15},
                                                  /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 0}, leftTopPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 0}, onePad},
                                   threeKernelSize,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 15},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 0}, onePad}}));
