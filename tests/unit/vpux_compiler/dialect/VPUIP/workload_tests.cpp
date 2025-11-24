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
    bool minimizeChannels;
    VPUIP::WorkloadComponents expectedWorkload;
};

std::ostream& operator<<(std::ostream& os, const WorkloadTestParams& params) {
    return os << formatv("{ inputWorkload={0}, kernelSize={1}, expectedWorkload={2} }", params.inputWorkload,
                         params.kernelSize, params.expectedWorkload)
                         .str();
}

using MLIR_WorkloadTest = testing::TestWithParam<WorkloadTestParams>;

TEST_P(MLIR_WorkloadTest, MinimizeWorkloadSize) {
    const auto& params = GetParam();
    const auto workload = VPUIP::minimizeWorkloadSize(params.inputWorkload, params.kernelSize, params.minimizeChannels);
    EXPECT_EQ(workload.inStart, params.expectedWorkload.inStart);
    EXPECT_EQ(workload.inEnd, params.expectedWorkload.inEnd);
    EXPECT_EQ(workload.outStart, params.expectedWorkload.outStart);
    EXPECT_EQ(workload.outEnd, params.expectedWorkload.outEnd);
    EXPECT_EQ(workload.pad, params.expectedWorkload.pad);
}

namespace {
const auto doNotMinimizeChannels = false;
const auto minimizeChannels = true;
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
                                   noKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, noPad},
                                   noKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 16}, noPad},
                                   noKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, noPad}}));

INSTANTIATE_TEST_SUITE_P(
        KernelSize, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{11, 11, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 15}, noPad},
                                   threeKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{2, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, noPad},
                                   threeKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{2, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 16}, noPad},
                                   threeKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{7, 7, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, noPad}}));

INSTANTIATE_TEST_SUITE_P(
        Padded, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{
                        VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{11, 11, 15},
                                                  /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 15}, onePad},
                        threeKernelSize, doNotMinimizeChannels,
                        VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{1, 1, 15},
                                                  /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, leftTopPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, onePad},
                                   threeKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{1, 1, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, leftTopPad}},
                WorkloadTestParams{
                        VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 16},
                                                  /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 16}, onePad},
                        threeKernelSize, doNotMinimizeChannels,
                        VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{6, 6, 16},
                                                  /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, leftTopPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, onePad},
                                   threeKernelSize, doNotMinimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 16},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, onePad}}));

INSTANTIATE_TEST_SUITE_P(
        MinimizeChannels, MLIR_WorkloadTest,
        testing::Values(
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{9, 9, 31},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{9, 9, 31}, noPad},
                                   noKernelSize, minimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 15},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 15}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{3, 2, 0},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{3, 2, 0}, noPad},
                                   noKernelSize, minimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{0, 0, 0}, /*inEnd=*/{0, 0, 0},
                                                             /*outStart=*/{0, 0, 0}, /*outEnd=*/{0, 0, 0}, noPad}},
                WorkloadTestParams{VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{8, 7, 31},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{8, 7, 16}, noPad},
                                   noKernelSize, minimizeChannels,
                                   VPUIP::WorkloadComponents{/*inStart=*/{5, 5, 16}, /*inEnd=*/{5, 5, 31},
                                                             /*outStart=*/{5, 5, 16}, /*outEnd=*/{5, 5, 16}, noPad}}));
