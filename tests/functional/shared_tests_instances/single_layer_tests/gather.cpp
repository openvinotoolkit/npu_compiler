// Copyright (C) 2020 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/gather.hpp"
#include <random>
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

void checkInOutRank(int inputRank, int indexRank, int batchDims, std::stringstream& skip) {
    if (inputRank != 4) {
        skip << "Gather only supports 4D input shape, inRank = " + std::to_string(inputRank);
    }

    auto outRank = inputRank + indexRank - 1 - batchDims;
    if (outRank != 4) {
        skip << "Gather only supports 4D output shape, outRank = " + std::to_string(outRank);
    }
}

class GatherLayerTest_NPU3720 : public GatherLayerTest, virtual public VpuOv2LayerTest {};
class GatherLayerTest_NPU4000 : public GatherLayerTest, virtual public VpuOv2LayerTest {};

class Gather7LayerTest_NPU3720 : public Gather7LayerTest, virtual public VpuOv2LayerTest {};

class Gather8LayerTest_NPU3720 : public Gather8LayerTest, virtual public VpuOv2LayerTest {};
class Gather8LayerTest_NPU4000 : public Gather8LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(GatherLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Gather7LayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Gather8LayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(Gather8LayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> inputShapes = {
        // {{10, 20, 30, 40}},
        {{5, 6, 7, 8}},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<ov::Shape> indicesShapes = {
        {4},
        //{2, 2}  //  Only 1D shape for indices is supported
};

const std::vector<int> axes = {0, 1, 2, 3, /*-1*/};  // Only positive axis value is supported

const auto params =
        testing::Combine(testing::ValuesIn(indices),                                            // Indices
                         testing::ValuesIn(indicesShapes),                                      // Indices shape
                         testing::ValuesIn(axes),                                               // Gather axis
                         testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),  // Input shapes
                         testing::ValuesIn(modelTypes),                                         // Model type
                         testing::Values(DEVICE_NPU));                                          // Device name

INSTANTIATE_TEST_CASE_P(smoke_Gather1, GatherLayerTest_NPU3720, params, GatherLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Gather1, GatherLayerTest_NPU4000, params, GatherLayerTest_NPU4000::getTestCaseName);
}  // namespace

namespace {  // conformance scenarios

const auto genParams(const ov::Shape inputShape, const int axis, const size_t idxNum) {
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

    return testing::Combine(testing::Values(_indices), testing::Values(std::vector<size_t>{idxNum}),
                            testing::Values(axis), testing::Values(static_shapes_to_test_representation({inputShape})),
                            testing::ValuesIn(modelTypes), testing::Values(DEVICE_NPU));
}

#define GEN_TEST(no, inputShape, axis, numIndices)                                                                  \
    INSTANTIATE_TEST_CASE_P(conform_Gather1_##no, GatherLayerTest_NPU3720, genParams(inputShape, axis, numIndices), \
                            GatherLayerTest_NPU3720::getTestCaseName)

#define GEN_PRECOMMIT_NPU3720_TEST(no, inputShape, axis, numIndices)                  \
    INSTANTIATE_TEST_SUITE_P(conform_precommit_Gather1_##no, GatherLayerTest_NPU3720, \
                             genParams(inputShape, axis, numIndices), GatherLayerTest_NPU3720::getTestCaseName)

GEN_TEST(0, (ov::Shape{10, 20, 30, 40}), 2, 4);                  //=> {10,20,4,40}
GEN_TEST(1, (ov::Shape{32, 3, 3, 3}), 0, 27);                    //=> {27,3,3,3}
GEN_TEST(2, (ov::Shape{32, 1, 3, 3}), 0, 27);                    //=> {27,1,3,3}
GEN_TEST(3, (ov::Shape{16, 32, 3, 3}), 1, 27);                   //=> {16,27,3,3}
GEN_TEST(4, (ov::Shape{96, 16, 1, 1}), 0, 95);                   //=> {95,16,1,1}
GEN_TEST(5, (ov::Shape{24, 96, 1, 1}), 1, 95);                   //=> {24,95,1,1}
GEN_TEST(6, (ov::Shape{144, 24, 1, 1}), 0, 143);                 //=> {143,24,1,1}
GEN_TEST(7, (ov::Shape{144, 1, 3, 3}), 0, 143);                  //=> {143,1,3,3}
GEN_TEST(8, (ov::Shape{24, 144, 1, 1}), 1, 143);                 //=> {24,143,1,1}
GEN_TEST(9, (ov::Shape{192, 32, 1, 1}), 0, 191);                 //=> {191,32,1,1}
GEN_TEST(10, (ov::Shape{32, 192, 1, 1}), 1, 191);                //=> {32,191,1,1}
GEN_TEST(11, (ov::Shape{384, 1, 3, 3}), 0, 380);                 //=> {380,1,3,3}
GEN_TEST(12, (ov::Shape{576, 1, 3, 3}), 0, 574);                 //=> {574,1,3,3}
GEN_TEST(13, (ov::Shape{576, 1, 3, 3}), 0, 571);                 //=> {571,1,3,3}
GEN_TEST(14, (ov::Shape{960, 1, 3, 3}), 0, 954);                 //=> {954,1,3,3}
GEN_TEST(15, (ov::Shape{960, 1, 3, 3}), 0, 959);                 //=> {959,1,3,3}
GEN_TEST(16, (ov::Shape{2, 64, 1, 1}), 0, 128);                  //=> {128,64,1,1}
GEN_TEST(17, (ov::Shape{2, 64, 1, 1}), 1, 128);                  //=> {2,128,1,1}
GEN_PRECOMMIT_NPU3720_TEST(1, (ov::Shape{16, 3, 3, 3}), 0, 27);  //=> {27,3,3,3}
GEN_PRECOMMIT_NPU3720_TEST(2, (ov::Shape{16, 1, 3, 3}), 0, 27);  //=> {27,1,3,3}

}  // namespace

namespace {  // opset7::Gather tests

#define GEN7_TEST(no, inputShape, indicesShape, axis, batch_dims)                                                 \
    INSTANTIATE_TEST_CASE_P(smoke_Gather7_##no, Gather7LayerTest_NPU3720,                                         \
                            testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                             testing::Values(std::vector<size_t> indicesShape),                   \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                             testing::Values(ov::element::f16), testing::Values(DEVICE_NPU)),     \
                            Gather7LayerTest_NPU3720::getTestCaseName)

#define GEN7_PRECOMMIT_NPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                                \
    INSTANTIATE_TEST_SUITE_P(smoke_precommit_Gather7_##no, Gather7LayerTest_NPU3720,                               \
                             testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                              testing::Values(std::vector<size_t> indicesShape),                   \
                                              testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                              testing::Values(ov::element::f16), testing::Values(DEVICE_NPU)),     \
                             Gather7LayerTest_NPU3720::getTestCaseName)

GEN7_TEST(0, (ov::Shape{3, 5, 1, 1}), ({3, 2}), 1, 1);
GEN7_TEST(1, (ov::Shape{4, 3, 5, 1}), ({4, 4}), 2, 1);
GEN7_TEST(2, (ov::Shape{3, 2, 1, 1}), ({3, 2}), 1, 1);
GEN7_TEST(3, (ov::Shape{2, 2, 5, 1}), ({2, 2, 3}), 2, 2);
GEN7_TEST(4, (ov::Shape{2, 1, 5, 4}), ({2, 3}), 2, 1);
GEN7_TEST(5, (ov::Shape{2, 5, 2, 1}), ({2, 2, 3}), 1, 1);
GEN7_TEST(6, (ov::Shape{2, 5, 1, 1}), ({2, 3}), 1, 1);
GEN7_TEST(7, (ov::Shape{3871, 1}), ({1, 193}), 0, 0);
GEN7_PRECOMMIT_NPU3720_TEST(0, (ov::Shape{3, 4, 1, 1}), ({3, 1}), 1, 1);
GEN7_PRECOMMIT_NPU3720_TEST(1, (ov::Shape{3, 2, 4, 1}), ({3, 3}), 2, 1);

}  // namespace

namespace {  // opset8::Gather tests

const std::vector<ov::element::Type> modelType = {ov::element::f16, ov::element::u8, ov::element::i8};
#define GEN8_TEST(no, inputShape, indicesShape, axis, batch_dims)                                                 \
    INSTANTIATE_TEST_CASE_P(smoke_Gather8_##no, Gather8LayerTest_NPU3720,                                         \
                            testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                             testing::Values(std::vector<size_t> indicesShape),                   \
                                             testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                             testing::ValuesIn(modelType), testing::Values(DEVICE_NPU)),          \
                            Gather8LayerTest_NPU3720::getTestCaseName)

#define GEN8_PRECOMMIT_NPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                                \
    INSTANTIATE_TEST_SUITE_P(smoke_precommit_Gather8_##no, Gather8LayerTest_NPU3720,                               \
                             testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                              testing::Values(std::vector<size_t> indicesShape),                   \
                                              testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                              testing::ValuesIn(modelType), testing::Values(DEVICE_NPU)),          \
                             Gather8LayerTest_NPU3720::getTestCaseName)

#define GEN8_TILING_NPU3720_TEST(no, inputShape, indicesShape, axis, batch_dims)                                   \
    INSTANTIATE_TEST_SUITE_P(smoke_Gather8_Tiling_##no, Gather8LayerTest_NPU3720,                                  \
                             testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                              testing::Values(std::vector<size_t> indicesShape),                   \
                                              testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                              testing::ValuesIn(modelType), testing::Values(DEVICE_NPU)),          \
                             Gather8LayerTest_NPU3720::getTestCaseName)

#define GEN8_PRECOMMIT_NPU4000_TEST(no, inputShape, indicesShape, axis, batch_dims)                                \
    INSTANTIATE_TEST_SUITE_P(smoke_precommit_Gather8_##no, Gather8LayerTest_NPU4000,                               \
                             testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                              testing::Values(std::vector<size_t> indicesShape),                   \
                                              testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                              testing::ValuesIn(modelType), testing::Values(DEVICE_NPU)),          \
                             Gather8LayerTest_NPU4000::getTestCaseName)

#define GEN8_NPU4000_TEST(no, inputShape, indicesShape, axis, batch_dims)                                          \
    INSTANTIATE_TEST_SUITE_P(smoke_Gather8_##no, Gather8LayerTest_NPU4000,                                         \
                             testing::Combine(testing::Values(static_shapes_to_test_representation({inputShape})), \
                                              testing::Values(std::vector<size_t> indicesShape),                   \
                                              testing::Values(std::tuple<int, int>{axis, batch_dims}),             \
                                              testing::ValuesIn(modelType), testing::Values(DEVICE_NPU)),          \
                             Gather8LayerTest_NPU4000::getTestCaseName)
GEN8_TEST(0, (ov::Shape{3, 5, 1, 1}), ({3, 2}), 1, 1);
GEN8_TEST(1, (ov::Shape{4, 3, 5, 1}), ({4, 4}), 2, 1);
GEN8_TEST(2, (ov::Shape{3, 2, 1, 1}), ({3, 2}), 1, 1);
GEN8_TEST(3, (ov::Shape{2, 2, 5, 1}), ({2, 2, 3}), 2, 2);
GEN8_TEST(4, (ov::Shape{2, 1, 5, 4}), ({2, 3}), 2, 1);
GEN8_TEST(5, (ov::Shape{2, 5, 1, 1}), ({2, 3}), 1, 1);
GEN8_TILING_NPU3720_TEST(6, (ov::Shape{4004, 320}), ({1}), 0, 0);
GEN8_TILING_NPU3720_TEST(7, (ov::Shape{2, 4004, 320}), ({2, 1}), 1, 1);
GEN8_TILING_NPU3720_TEST(8, (ov::Shape{387072, 3}), ({1, 387072}), 0, 0);
GEN8_TILING_NPU3720_TEST(9, (ov::Shape{1548288, 1}), ({1, 100}), 0, 0);
GEN8_PRECOMMIT_NPU3720_TEST(0, (ov::Shape{2, 3, 1, 1}), ({2, 1}), 1, 1);
GEN8_PRECOMMIT_NPU3720_TEST(1, (ov::Shape{3, 2, 4, 1}), ({3, 3}), 2, 1);
GEN8_PRECOMMIT_NPU4000_TEST(0, (ov::Shape{2, 3, 1, 1}), ({2, 1}), 1, 1);
GEN8_PRECOMMIT_NPU4000_TEST(1, (ov::Shape{3, 2, 4, 1}), ({3, 3}), 2, 1);
GEN8_PRECOMMIT_NPU4000_TEST(2, (ov::Shape{1, 3, 64, 512}), ({3}), 1, 0);
}  // namespace
