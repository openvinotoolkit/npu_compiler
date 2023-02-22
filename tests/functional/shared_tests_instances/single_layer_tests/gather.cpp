// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather.hpp"
#include <random>
#include <vector>
#include <vpux/vpux_plugin_config.hpp>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

void checkInOutRank(int inputRank, int indexRank, int batchDims) {
    if (inputRank != 4) {
        throw LayerTestsUtils::KmbSkipTestException("Gather only supports 4D input shape, inRank = " +
                                                    std::to_string(inputRank));
    }

    auto outRank = inputRank + indexRank - 1 - batchDims;
    if (outRank != 4) {
        throw LayerTestsUtils::KmbSkipTestException("Gather only supports 4D output shape, outRank = " +
                                                    std::to_string(outRank));
    }
}

class KmbGatherLayerTest : public GatherLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::vector<size_t> inputShape;
        std::string device;
        std::tie(std::ignore, std::ignore, std::ignore, inputShape, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, device) = GetParam();

        if (inputShape.size() != 4) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports 4D input shape");
        }

        if (device == "EMULATOR") {
            throw LayerTestsUtils::KmbSkipTestException(
                    "Emulator device does not support Gather with I32 second input");
        }
    }
};

class KmbGatherLayerTest_VPU3720 : public GatherLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class KmbGather7LayerTest : public Gather7LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        auto inputRank = std::get<0>(GetParam()).size();
        auto indexRank = std::get<1>(GetParam()).size();
        auto batchDims = std::get<1>(std::get<2>(GetParam()));
        checkInOutRank(inputRank, indexRank, batchDims);
    }
};

class KmbGather7LayerTest_VPU3720 : public Gather7LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        auto inputRank = std::get<0>(GetParam()).size();
        auto indexRank = std::get<1>(GetParam()).size();
        auto batchDims = std::get<1>(std::get<2>(GetParam()));
        checkInOutRank(inputRank, indexRank, batchDims);
    }
};

class KmbGather8LayerTest : public Gather8LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        auto inputRank = std::get<0>(GetParam()).size();
        auto indexRank = std::get<1>(GetParam()).size();
        auto batchDims = std::get<1>(std::get<2>(GetParam()));
        checkInOutRank(inputRank, indexRank, batchDims);
    }
};

class KmbGather8LayerTest_VPU3720 : public Gather8LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbGatherLayerTest, CompareWithRefs) {
    // Enable NCHW layout
    core->SetConfig({}, LayerTestsUtils::testPlatformTargetDevice);
    Run();
}

TEST_P(KmbGatherLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbGatherLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbGather7LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbGather7LayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbGather8LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbGather8LayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShapes = {
        // std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{5, 6, 7, 8},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
        std::vector<size_t>{4},
        // std::vector<size_t>{2, 2}  //  Only 1D shape for indices is supported
};

const std::vector<int> axes = {0, 1, 2, 3, /*-1*/};  // Only positive axis value is supported

const auto params = testing::Combine(
        testing::ValuesIn(indices), testing::ValuesIn(indicesShapes), testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

// nGraph parser doesn't contain specific gather parser
// [Track number: S#40603]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Gather1, KmbGatherLayerTest, params, KmbGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather1_VPU3720, KmbGatherLayerTest_VPU3720, params,
                        KmbGatherLayerTest_VPU3720::getTestCaseName);
}  // namespace

namespace {  // conformance scenarios

const auto genParams(const std::vector<size_t> inputShape, const int axis, const size_t idxNum) {
    std::vector<int> _indices(idxNum, 0);

    if (axis >= inputShape.size()) {
        std::cout << "error: axis=" << axis << " out of range, ";
        std::cout << "valid range = [0.." << inputShape.size() - 1 << "]" << std::endl;
        abort();
    }

    // Initialize indices within valid range
    const size_t max = inputShape[axis];
    std::default_random_engine gen(123);
    std::uniform_int_distribution<int> distrib(0, max - 1);
    for (size_t i = 0; i < _indices.size(); i++) {
        _indices[i] = distrib(gen);
    }

    return testing::Combine(
            testing::Values(_indices), testing::Values(std::vector<size_t>{idxNum}), testing::Values(axis),
            testing::Values(inputShape), testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice));
}

#define GEN_TEST(no, inputShape, axis, numIndices)                                                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_conform_Gather1_##no, KmbGatherLayerTest,                         \
                            genParams(inputShape, axis, numIndices), KmbGatherLayerTest::getTestCaseName); \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_conform_Gather1_VPU3720_##no, KmbGatherLayerTest_VPU3720,         \
                            genParams(inputShape, axis, numIndices), KmbGatherLayerTest_VPU3720::getTestCaseName)

#define GEN_PRECOMMIT_VPU3720_TEST(no, inputShape, axis, numIndices)                                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_conform_precommit_Gather1_VPU3720_##no, KmbGatherLayerTest_VPU3720, \
                            genParams(inputShape, axis, numIndices), KmbGatherLayerTest_VPU3720::getTestCaseName)

GEN_TEST(0, (std::vector<size_t>{10, 20, 30, 40}), 2, 4);                  //=> {10,20,4,40}
GEN_TEST(1, (std::vector<size_t>{32, 3, 3, 3}), 0, 27);                    //=> {27,3,3,3}
GEN_TEST(2, (std::vector<size_t>{32, 1, 3, 3}), 0, 27);                    //=> {27,1,3,3}
GEN_TEST(3, (std::vector<size_t>{16, 32, 3, 3}), 1, 27);                   //=> {16,27,3,3}
GEN_TEST(4, (std::vector<size_t>{96, 16, 1, 1}), 0, 95);                   //=> {95,16,1,1}
GEN_TEST(5, (std::vector<size_t>{24, 96, 1, 1}), 1, 95);                   //=> {24,95,1,1}
GEN_TEST(6, (std::vector<size_t>{144, 24, 1, 1}), 0, 143);                 //=> {143,24,1,1}
GEN_TEST(7, (std::vector<size_t>{144, 1, 3, 3}), 0, 143);                  //=> {143,1,3,3}
GEN_TEST(8, (std::vector<size_t>{24, 144, 1, 1}), 1, 143);                 //=> {24,143,1,1}
GEN_TEST(9, (std::vector<size_t>{192, 32, 1, 1}), 0, 191);                 //=> {191,32,1,1}
GEN_TEST(10, (std::vector<size_t>{32, 192, 1, 1}), 1, 191);                //=> {32,191,1,1}
GEN_TEST(11, (std::vector<size_t>{384, 1, 3, 3}), 0, 380);                 //=> {380,1,3,3}
GEN_TEST(12, (std::vector<size_t>{576, 1, 3, 3}), 0, 574);                 //=> {574,1,3,3}
GEN_TEST(13, (std::vector<size_t>{576, 1, 3, 3}), 0, 571);                 //=> {571,1,3,3}
GEN_TEST(14, (std::vector<size_t>{960, 1, 3, 3}), 0, 954);                 //=> {954,1,3,3}
GEN_TEST(15, (std::vector<size_t>{960, 1, 3, 3}), 0, 959);                 //=> {959,1,3,3}
GEN_TEST(16, (std::vector<size_t>{2, 64, 1, 1}), 0, 128);                  //=> {128,64,1,1}
GEN_TEST(17, (std::vector<size_t>{2, 64, 1, 1}), 1, 128);                  //=> {2,128,1,1}
GEN_PRECOMMIT_VPU3720_TEST(1, (std::vector<size_t>{16, 3, 3, 3}), 0, 27);  //=> {27,3,3,3}
GEN_PRECOMMIT_VPU3720_TEST(2, (std::vector<size_t>{16, 1, 3, 3}), 0, 27);  //=> {27,1,3,3}

}  // namespace

namespace {  // opset7::Gather tests

#define GEN7_TEST(no, inputShape, indicesShape, axis, batch_dims)                                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Gather7_##no, KmbGather7LayerTest,                         \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),             \
                                             testing::Values(std::vector<size_t> indicesShape),           \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),     \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
                            KmbGather7LayerTest::getTestCaseName);                                        \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Gather7_VPU3720_##no, KmbGather7LayerTest_VPU3720,         \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),             \
                                             testing::Values(std::vector<size_t> indicesShape),           \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),     \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
                            KmbGather7LayerTest_VPU3720::getTestCaseName)

#define GEN7_PRECOMMIT_VPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_Gather7_VPU3720_##no, KmbGather7LayerTest_VPU3720, \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),               \
                                             testing::Values(std::vector<size_t> indicesShape),             \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),       \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Layout::ANY),                 \
                                             testing::Values(InferenceEngine::Layout::ANY),                 \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)),   \
                            KmbGather7LayerTest_VPU3720::getTestCaseName)

GEN7_TEST(0, ({3, 5, 1, 1}), ({3, 2}), 1, 1);
GEN7_TEST(1, ({4, 3, 5, 1}), ({4, 4}), 2, 1);
GEN7_TEST(2, ({3, 2, 1, 1}), ({3, 2}), 1, 1);
GEN7_TEST(3, ({2, 2, 5, 1}), ({2, 2, 3}), 2, 2);
GEN7_TEST(4, ({2, 1, 5, 4}), ({2, 3}), 2, 1);
GEN7_TEST(5, ({2, 5, 2, 1}), ({2, 2, 3}), 1, 1);
GEN7_TEST(6, ({2, 5, 1, 1}), ({2, 3}), 1, 1);
GEN7_PRECOMMIT_VPU3720_TEST(0, ({3, 4, 1, 1}), ({3, 1}), 1, 1);
GEN7_PRECOMMIT_VPU3720_TEST(1, ({3, 2, 4, 1}), ({3, 3}), 2, 1);

}  // namespace

namespace {  // opset8::Gather tests

#define GEN8_TEST(no, inputShape, indicesShape, axis, batch_dims)                                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Gather8_##no, KmbGather8LayerTest,                         \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),             \
                                             testing::Values(std::vector<size_t> indicesShape),           \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),     \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
                            KmbGather8LayerTest::getTestCaseName);                                        \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Gather8_VPU3720_##no, KmbGather8LayerTest_VPU3720,         \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),             \
                                             testing::Values(std::vector<size_t> indicesShape),           \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),     \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
                            KmbGather8LayerTest::getTestCaseName)

#define GEN8_PRECOMMIT_VPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                         \
    INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_Gather8_VPU3720_##no, KmbGather8LayerTest_VPU3720, \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),               \
                                             testing::Values(std::vector<size_t> indicesShape),             \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),       \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Precision::FP16),             \
                                             testing::Values(InferenceEngine::Layout::ANY),                 \
                                             testing::Values(InferenceEngine::Layout::ANY),                 \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)),   \
                            KmbGather8LayerTest::getTestCaseName)

#define GEN8_TILING_VPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                          \
    INSTANTIATE_TEST_CASE_P(smoke_Gather8_Tiling_VPU3720_##no, KmbGather8LayerTest_VPU3720,               \
                            testing::Combine(testing::Values(std::vector<size_t> inputShape),             \
                                             testing::Values(std::vector<size_t> indicesShape),           \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),     \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Precision::FP16),           \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(InferenceEngine::Layout::ANY),               \
                                             testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
                            KmbGather8LayerTest::getTestCaseName)

GEN8_TEST(0, ({3, 5, 1, 1}), ({3, 2}), 1, 1);
GEN8_TEST(1, ({4, 3, 5, 1}), ({4, 4}), 2, 1);
GEN8_TEST(2, ({3, 2, 1, 1}), ({3, 2}), 1, 1);
GEN8_TEST(3, ({2, 2, 5, 1}), ({2, 2, 3}), 2, 2);
GEN8_TEST(4, ({2, 1, 5, 4}), ({2, 3}), 2, 1);
GEN8_TEST(5, ({2, 5, 1, 1}), ({2, 3}), 1, 1);
GEN8_TILING_VPU3720_TEST(6, ({4004, 320}), ({1}), 0, 0);
GEN8_TILING_VPU3720_TEST(7, ({2, 4004, 320}), ({2, 1}), 1, 1);
GEN8_PRECOMMIT_VPU3720_TEST(0, ({2, 3, 1, 1}), ({2, 1}), 1, 1);
GEN8_PRECOMMIT_VPU3720_TEST(1, ({3, 2, 4, 1}), ({3, 3}), 2, 1);

}  // namespace
