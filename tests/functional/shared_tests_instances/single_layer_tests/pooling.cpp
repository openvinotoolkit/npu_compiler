// Copyright (C) 2020 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/pooling.hpp"
#include <common/functions.h>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>
#include "intel_npu/npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Option added base on CI request to decrease test runtime
// Important to enable macro (remove //) to run full tests in CI every time your change can impact AVG/MAX pool.
// Both operations are transformed in some scenario to NCE task, so it is important to enable testing when touch any of
// this mlir passes.
// #define ENABLE_ALL_POOL_TESTS

#ifdef ENABLE_ALL_POOL_TESTS
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(A, B, C, D)
#else
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(DISABLED_##A, B, C, D)
#endif

void skipCompilationCallbackImplNPU3720(std::stringstream& skip, ov::AnyMap configuration, PoolingTypes poolType,
                                        std::vector<size_t> kernel, std::vector<size_t> strides,
                                        std::vector<size_t> padBegin, std::vector<size_t> padEnd,
                                        ov::op::PadType padType, bool excludePad, ov::element::Type netPrecision,
                                        ov::op::RoundingType roundingMode, ov::Shape inputShapes) {
    // all DefaultHW test Should be enable when E#94485 will be fixed. Convert to HW scenario not implemented
    if ((poolType == PoolingTypes::AVG) && (configuration[ov::intel_npu::compilation_mode.name()] == "DefaultHW") &&
        (strides.size() == 2)) {
        // support exclude pad for reduce number of scenario, when HandleExcludePadForAvgPoolPass fail,
        // excludePad remain, should not validate ConvertIEToVPUNCEPass for AvgPool
        if (excludePad) {
            std::vector<size_t> ones{1, 1};
            if ((padBegin != ones) || (padEnd != ones) || (strides != ones)) {
                skip << "AVGPool convert to NCE with excludePad, invalid conversion";
            }
        }
        // special implementation in reference for CEIL rounding mode with padBegin=0, padEnd=0, and Ceil rounding
        // involve padding with 1. see openvino reference:
        // openvino/src/core/reference/include/ngraph/runtime/reference/avg_pool.hpp
        // if all are 0, and CEIL request in fact padding, then enable excludePad, else, if just 1 of
        // value are not 0, work as expected, divide by constant kernel size.
        // Hw should implement in same way, or allow go to SW implementation.
        if (roundingMode == ov::op::RoundingType::CEIL) {
            std::vector<size_t> zeros{0, 0};
            if ((padBegin == zeros) && (padEnd == zeros)) {
                skip << "AVG pool CEIL rounding with PADS 0 are not proper converted to NCE AvgPool";
            }
        }
        // Default HW pipe produce wrong values for this combination, if input size is just 8x8 that produce 3x3 or
        // 1x1 output, SW reference pipeline pass. Probably align issue in HW version. Padding mode is valid
        if ((inputShapes[3] <= 8) && (inputShapes[2] <= 8) && (strides[0] == 2) && (strides[1] == 2) &&
            (padType == ov::op::PadType::VALID)) {
            skip << "AVG Pool VALID pad type for small resolution invalid conversion to NCE AvgPool";
        }
    }

    // Invalid padding with 0 for MaxPool, should be -MaxFloat
    // src/vpux_compiler/src/dialect/IE/passes/handle_large_pads.cpp pad with 0 as for Avg, but Max is not the same.
    // Remove when  E#99182 will be fixed. Or open a separate ticket related to E#69906
    if ((poolType == PoolingTypes::MAX) && (configuration[ov::intel_npu::compilation_mode.name()] == "DefaultHW") &&
        (strides.size() == 2)) {
        size_t kernel0 = kernel[0];
        size_t kernel1 = kernel[1];
        if (roundingMode == ov::op::RoundingType::CEIL) {
            kernel0 -= 1;
            kernel1 -= 1;
        }
        if ((padBegin[0] >= kernel0) || (padBegin[1] >= kernel1) || (padEnd[0] >= kernel0) || (padEnd[1] >= kernel1)) {
            skip << "MAX pool, Hw NCE version produce invalid 0 values on pad area";
        }
    }

    // MaxPool Opset1 and AvgPool Opset1 does not support i8 or u8 dataType at OV evaluator level, but works in networks
    if (((poolType == PoolingTypes::MAX) || poolType == (PoolingTypes::AVG)) &&
        (netPrecision == ov::element::i8 || netPrecision == ov::element::u8)) {
        skip << "Max pool SingleLayerTest is not enabled with precision: " << netPrecision;
    }
}

class PoolingLayerTest_NPU3720 : public PoolingLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};
class PoolingLayerTest_NPU3720_SOB : public PoolingLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "skip-unroll-batch=true";
    }
};

TEST_P(PoolingLayerTest_NPU3720, SW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    abs_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(PoolingLayerTest_NPU3720, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

using PoolingLayerTest_NPU3720_SingleCluster = PoolingLayerTest_NPU3720;

TEST_P(PoolingLayerTest_NPU3720_SingleCluster, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    setDefaultHardwareMode();
    setSingleClusterMode();
    run(Platform::NPU3720);
}

TEST_P(PoolingLayerTest_NPU3720_SOB, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class PoolingLayerTest_NPU4000 : public PoolingLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

class PoolingLayerTestWithUnrollBatchingCompileMethod : public PoolingLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "batch-compile-method=unroll batch-unroll-settings={skip-unroll-batch=true}";
    }
};

class PoolingLayerTest_NPU4000_SOB : public PoolingLayerTestWithUnrollBatchingCompileMethod {};

class PoolingLayerTest_NPU4000_F32 : public PoolingLayerTest_NPU4000 {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(PoolingLayerTest_NPU4000, SW) {
    abs_threshold = 0.02;
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType = std::get<0>(poolParams);
        const auto netPrecision = std::get<1>(GetParam());
        if (((poolType == PoolingTypes::MAX) || poolType == (PoolingTypes::AVG)) &&
            (netPrecision == ov::element::i8 || netPrecision == ov::element::u8)) {
            skip << "Max pool SingleLayerTest is not enabled with precision: " << netPrecision;
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(PoolingLayerTest_NPU4000_F32, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(PoolingLayerTest_NPU4000_SOB, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
class MaxPoolingV8LayerTestCommon : public MaxPoolingV8LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(MaxPoolingV8LayerTestCommon, NPU3720_SW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto netPrecision = std::get<1>(GetParam());
        if (netPrecision == ov::element::i8 || netPrecision == ov::element::u8) {
            skip << "MaxPool8 SingleLayerTest is not enabled with precision: " << netPrecision;
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MaxPoolingV8LayerTestCommon, NPU4000_SW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto netPrecision = std::get<1>(GetParam());
        if (netPrecision == ov::element::i8 || netPrecision == ov::element::u8) {
            skip << "MaxPool8 SingleLayerTest is not enabled with precision: " << netPrecision;
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ============= AutoPadValid ============= */

std::vector<std::vector<ov::Shape>> inputShapesAutoPadValid = {
        {{1, 8, 32, 32}}, {{1, 16, 24, 24}}, {{1, 24, 16, 16}}, {{1, 32, 8, 8}}};
const auto pool_AutoPadValid = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),     //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {5, 5}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),              //
                           ::testing::Values(ov::op::PadType::VALID),                   //
                           ::testing::Values(false)),  // excludePad,                          //
        ::testing::Values(ov::element::f16),           // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesAutoPadValid)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= ExplicitPadding ============= */

std::vector<std::vector<ov::Shape>> inputShapesExplicitPadding = {{{1, 16, 30, 30}}};
const auto pool_ExplicitPadding = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),                  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR,
                                             ov::op::RoundingType::CEIL),                       //
                           ::testing::Values(ov::op::PadType::EXPLICIT),                        //
                           ::testing::Values(false)),                                           //
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AsymmetricKernel ============= */

const auto pool_AsymmetricKernel = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 1}, {1, 3}}),          // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AsymmetricStrides ============= */

const auto pool_AsymmetricStrides = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 2}, {2, 1}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= LargeSize ============= */

const auto pool_LargeSize1 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(false)),                           // excludePad, //
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 64, 128, 128}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargeSize2 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(false)),                           // excludePad
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 256, 256}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= LargeStrides ============= */

const auto pool_LargeStrides = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{9, 9}}),            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                //
                           ::testing::Values(ov::op::PadType::VALID),                     //
                           ::testing::Values(false)),                                     // excludePad

        ::testing::Values(ov::element::f16),  // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= Padding valitation ( > K_SZ/2) ============= */

const auto pool_LargePadding2 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                        //
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}, {3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),              //
                           ::testing::Values(ov::op::PadType::VALID),                   //
                           ::testing::Values(false)),                                   // excludePad
        ::testing::Values(ov::element::f16),                                            // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding3 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {4, 4}, {5, 5}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding4 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                        //
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}, {5, 5}, {6, 6}, {7, 7}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}}),                          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}}),                          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                              //
                           ::testing::Values(ov::op::PadType::VALID),                                   //
                           ::testing::Values(false)),                                                   // excludePad
        ::testing::Values(ov::element::f16),                                                            // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding5 = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(PoolingTypes::MAX),                                                //
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}),  // kernels
                ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                                  // strides
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}}),                                  // padBegins
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}}),                                  // padEnds
                ::testing::Values(ov::op::RoundingType::FLOOR),                                      //
                ::testing::Values(ov::op::PadType::VALID),                                           //
                ::testing::Values(false)),                                                           // excludePad
        ::testing::Values(ov::element::f16),                                                         // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding6 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),  //
                           ::testing::ValuesIn<std::vector<size_t>>(
                                   {{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),             // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{6, 6}}),             // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{6, 6}}),             // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                 //
                           ::testing::Values(ov::op::PadType::VALID),                      //
                           ::testing::Values(false)),                                      // excludePad
        ::testing::Values(ov::element::f16),                                               // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding7 = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(PoolingTypes::MAX),                                                    //
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                                      // strides
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}}),                                      // padBegins
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}}),                                      // padEnds
                ::testing::Values(ov::op::RoundingType::FLOOR),                                          //
                ::testing::Values(ov::op::PadType::VALID),                                               //
                ::testing::Values(false)),                                                               // excludePad
        ::testing::Values(ov::element::f16),                                                             // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

const auto pool_LargePadding8 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                            //
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                              // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}}),                              // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}}),                              // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                  //
                           ::testing::Values(ov::op::PadType::VALID),                                       //
                           ::testing::Values(false)),  // excludePad
        ::testing::Values(ov::element::f16),           // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AVGPooling / Large Kernels ============= */

const auto avgPool_largeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),    // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad, //
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2048, 23, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AVGPooling / Large KernelsX ============= */

const auto avgPool_largeKernelsX = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 14}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 1, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AVGPooling / Large KernelsY ============= */

const auto avgPool_largeKernelsY = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 1}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad,
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 14, 1}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AVGPooling / Large Prime Kernels ============= */

const auto avgPool_largePrimeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{17, 17}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),    // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad,
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 147, 17, 17}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= MAXPooling / Large Kernels ============= */

const auto maxPool_largeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2048, 23, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= MAXPooling / Large KernelsX ============= */

const auto maxPool_largeKernelsX = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 14}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 1, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= MAXPooling / Large KernelsY ============= */

const auto maxPool_largeKernelsY = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 1}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 14, 1}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ============= AvgPooling / Exclude_Pad Handling ============= */

const auto avgPool_excludePad = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(true)),                            // excludePad,
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 28, 28}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

/* ======================================== NPU 3720 ============================================================= */

/* ==== Custom tests scenario extra added for 3720 ===== */
const auto pool_ExplicitNoPadding_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 14}, {14, 1}, {1, 14}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),                          // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),                          // padBegins
                           ::testing::Values<std::vector<size_t>>({0, 0}),                          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 30, 14, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// U-net usecase
const auto pool_unet_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                           ::testing::Values<std::vector<size_t>>({12, 1}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),   // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),   // padBegins
                           ::testing::Values<std::vector<size_t>>({0, 0}),   // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 12, 176}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// large kernel
const auto pooling_largeKernel_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                           ::testing::ValuesIn<std::vector<size_t>>({{28, 28}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),       // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),       // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),       // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),           //
                           ::testing::Values(ov::op::PadType::VALID),                //
                           ::testing::Values(true)),                                 // excludePad
        ::testing::Values(ov::element::f16),                                         // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 70, 28, 28}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Large kernel with stride 1
const auto pooling_largeKernelStrideOne = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                           ::testing::ValuesIn<std::vector<size_t>>({{71, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),   // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),   // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),   // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),       //
                           ::testing::Values(ov::op::PadType::VALID),            //
                           ::testing::Values(false)),                            // excludePad,              //
        ::testing::Values(ov::element::f16),                                     // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 71, 2}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Test all padding type
const auto poolAllPadTypeParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{5, 7}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({2, 3}),      // strides
                           ::testing::Values<std::vector<size_t>>({2, 3}),      // padBegins
                           ::testing::Values<std::vector<size_t>>({1, 2}),      // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR, ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::EXPLICIT, ov::op::PadType::SAME_LOWER,
                                             ov::op::PadType::SAME_UPPER, ov::op::PadType::VALID),
                           ::testing::Values(true)),                                                      // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2, 30, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// 3D usecase
const auto pool3DParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::SAME_UPPER),
                           ::testing::Values(false)),                                                 // excludePad
        ::testing::Values(ov::element::f16),                                                          // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{3, 4, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// 5d usecase
const auto pool5DParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2, 2}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),
                           ::testing::Values(ov::op::PadType::SAME_UPPER),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 4, 16, 8, 12}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
const auto pooligBigPadEndParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({2, 2}),      // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),      // padBegins
                           ::testing::Values<std::vector<size_t>>({2, 2}),      // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR, ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 4, 54, 54}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// basic usecase
const auto pool_basic_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::Values<std::vector<size_t>>({3, 3}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // strides
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // padBegins
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f32),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2, 16, 24}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Integer Input/Output
const auto pool_inputOutputInteger = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),                      //
                           ::testing::Values<std::vector<size_t>>({{8, 8}}),                             // kernels
                           ::testing::Values<std::vector<size_t>>({{8, 1}}),                             // strides
                           ::testing::Values<std::vector<size_t>>({{0, 0}}),                             // padBegins
                           ::testing::Values<std::vector<size_t>>({{0, 0}}),                             // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                               //
                           ::testing::Values(ov::op::PadType::VALID),                                    //
                           ::testing::Values(false)),                                                    // excludePad,
        ::testing::Values(ov::element::u8, ov::element::i8, ov::element::i32, ov::element::i64),         // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 16, 8}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// SOB multi-clustering
auto generateSOBParamsFromInputShape = [](const std::vector<ov::Shape>& inputShapes) {
    return ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                                                 ::testing::Values<std::vector<size_t>>({3, 3}),  // kernels
                                                 ::testing::Values<std::vector<size_t>>({2, 2}),  // strides
                                                 ::testing::Values<std::vector<size_t>>({1, 1}),  // padBegins
                                                 ::testing::Values<std::vector<size_t>>({1, 1}),  // padEnds
                                                 ::testing::Values(ov::op::RoundingType::FLOOR),
                                                 ::testing::Values(ov::op::PadType::EXPLICIT),
                                                 ::testing::Values(false)),                          // excludePad
                              ::testing::Values(ov::element::f16),                                   // netPrc
                              ::testing::Values(static_shapes_to_test_representation(inputShapes)),  // inputShapes
                              ::testing::Values(DEVICE_NPU));
};

/* ========== MaxPool8 ========== */

const std::vector<std::vector<size_t>> kernels = {{3, 5}};
const std::vector<std::vector<size_t>> kernel_3d = {{2, 2, 2}};

const std::vector<std::vector<size_t>> strides = {{2, 1}};
const std::vector<std::vector<size_t>> strides_3d = {{1, 1, 1}};

const std::vector<std::vector<size_t>> dilation = {{2, 2}};
const std::vector<std::vector<size_t>> dilation_3d = {{1, 1, 1}};

const std::vector<std::vector<size_t>> pad_begins = {{0, 2}};
const std::vector<std::vector<size_t>> pad_begins_3d = {{0, 0, 0}};

const std::vector<std::vector<size_t>> pad_ends = {{0, 2}};
const std::vector<std::vector<size_t>> pad_ends_3d = {{0, 0, 0}};

const std::vector<std::vector<ov::Shape>> inputShape = {{{1, 3, 30, 30}}};
const std::vector<std::vector<ov::Shape>> inputShape_5d = {{{1, 4, 16, 8, 12}}};

/* ========== Explicit Pad Floor Rounding ========== */
const auto maxPool8_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(dilation),
                           ::testing::ValuesIn(pad_begins), ::testing::ValuesIn(pad_ends),
                           ::testing::Values(ov::element::i32), ::testing::Values(0),
                           ::testing::Values(ov::op::RoundingType::FLOOR),
                           ::testing::Values(ov::op::PadType::EXPLICIT)),
        ::testing::Values(ov::element::f16, ov::element::i32, ov::element::f32),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), ::testing::Values(DEVICE_NPU));

/* ========== Same Upper Pad Floor Rounding ========== */
const auto maxPool8_SameUpperPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(dilation),
                           ::testing::ValuesIn(pad_begins), ::testing::ValuesIn(pad_ends),
                           ::testing::Values(ov::element::i32), ::testing::Values(0),
                           ::testing::Values(ov::op::RoundingType::FLOOR),
                           ::testing::Values(ov::op::PadType::SAME_UPPER)),
        ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), ::testing::Values(DEVICE_NPU));

/* ========== Same Lower Pad Floor Rounding ========== */
const auto maxPool8_SameLowerPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(dilation),
                           ::testing::ValuesIn(pad_begins), ::testing::ValuesIn(pad_ends),
                           ::testing::Values(ov::element::i32), ::testing::Values(0),
                           ::testing::Values(ov::op::RoundingType::FLOOR),
                           ::testing::Values(ov::op::PadType::SAME_LOWER)),
        ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), ::testing::Values(DEVICE_NPU));

/* ========= Explicit Pad Ceil Rounding ========== */
const auto maxPool8_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(dilation),
                           ::testing::ValuesIn(pad_begins), ::testing::ValuesIn(pad_ends),
                           ::testing::Values(ov::element::i32), ::testing::Values(0),
                           ::testing::Values(ov::op::RoundingType::CEIL), ::testing::Values(ov::op::PadType::EXPLICIT)),
        ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), ::testing::Values(DEVICE_NPU));

/* ========= Explicit Pad Floor Rounding 5D input========== */
const auto maxPool8_ExplicitPad_FloorRounding_5Dinput_Params =
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernel_3d), ::testing::ValuesIn(strides_3d),
                                              ::testing::Values(dilation_3d[0]), ::testing::ValuesIn(pad_begins_3d),
                                              ::testing::ValuesIn(pad_ends_3d), ::testing::Values(ov::element::i32),
                                              ::testing::Values(0), ::testing::Values(ov::op::RoundingType::FLOOR),
                                              ::testing::Values(ov::op::PadType::EXPLICIT)),
                           ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape_5d)),
                           ::testing::Values(DEVICE_NPU));

/* ========= Same Upper Pad Floor Rounding 5D input========== */
const auto maxPool8_SameUpperPad_FloorRounding_5Dinput_Params =
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernel_3d), ::testing::ValuesIn(strides_3d),
                                              ::testing::ValuesIn(dilation_3d), ::testing::ValuesIn(pad_begins_3d),
                                              ::testing::ValuesIn(pad_ends_3d), ::testing::Values(ov::element::i32),
                                              ::testing::Values(0), ::testing::Values(ov::op::RoundingType::FLOOR),
                                              ::testing::Values(ov::op::PadType::SAME_UPPER)),
                           ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape_5d)),
                           ::testing::Values(DEVICE_NPU));

/* ========= Same Lower Pad Ceil Rounding 5D input========== */
const auto maxPool8_SameLowerPad_CeilRounding_5Dinput_Params =
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernel_3d), ::testing::ValuesIn(strides_3d),
                                              ::testing::ValuesIn(dilation_3d), ::testing::ValuesIn(pad_begins_3d),
                                              ::testing::ValuesIn(pad_ends_3d), ::testing::Values(ov::element::i32),
                                              ::testing::Values(0), ::testing::Values(ov::op::RoundingType::CEIL),
                                              ::testing::Values(ov::op::PadType::SAME_LOWER)),
                           ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape_5d)),
                           ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding, PoolingLayerTest_NPU3720, pool_ExplicitNoPadding_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pool_ExplicitNoPadding_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// U-net usecase
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet, PoolingLayerTest_NPU3720, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_unet_ELF, PoolingLayerTest_NPU3720_SingleCluster, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// large kernel
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel, PoolingLayerTest_NPU3720, pooling_largeKernel_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel3720_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pooling_largeKernel_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// Large kernel with stride 1
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_LargeKernelStrideOne, PoolingLayerTest_NPU3720,
                         pooling_largeKernelStrideOne, PoolingLayerTest_NPU3720::getTestCaseName);
// all PadType
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AllPadType, PoolingLayerTest_NPU3720, poolAllPadTypeParams,
                         PoolingLayerTest::getTestCaseName);
// 3D usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_3D, PoolingLayerTest_NPU3720, pool3DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// 5d usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_5D, PoolingLayerTest_NPU3720, pool5DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_BigPadEndParams, PoolingLayerTest_NPU3720, pooligBigPadEndParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// Integer Input/Output
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_InputOutputInteger, PoolingLayerTest_NPU3720, pool_inputOutputInteger,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// SOB multi-clustering
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_SOB, PoolingLayerTest_NPU3720_SOB,
                         generateSOBParamsFromInputShape({{2, 64, 28, 28}}),
                         PoolingLayerTest_NPU3720_SOB::getTestCaseName);
// previous cip reused tests
/* ============= AutoPadValid ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU3720, pool_AutoPadValid,
                                             PoolingLayerTest::getTestCaseName);
/* ============= ExplicitPadding ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU3720,
                                             pool_ExplicitPadding, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricKernel ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricKernel, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricStrides ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricStrides, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsX ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsY ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Prime Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largePrimeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AvgPooling / Exclude_Pad Handling ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_avgPool_excludePad, PoolingLayerTest_NPU3720, avgPool_excludePad,
                                             PoolingLayerTest::getTestCaseName);

// Max pool ported tests from VPU3720
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU3720, pool_LargeSize1,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU3720, pool_LargeSize2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU3720, pool_LargeStrides,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU3720, pool_LargePadding2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU3720, pool_LargePadding3,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU3720, pool_LargePadding4,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU3720, pool_LargePadding5,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU3720, pool_LargePadding6,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU3720, pool_LargePadding7,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU3720, pool_LargePadding8,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernels, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

/* ======================================== NPU 4000 ============================================================= */

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NoPadding, PoolingLayerTest_NPU4000, pool_ExplicitNoPadding_Params,
                         PoolingLayerTest::getTestCaseName);
// U-net usecase
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet, PoolingLayerTest_NPU4000, pool_unet_Params,
                         PoolingLayerTest::getTestCaseName);
// large kernel
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel, PoolingLayerTest_NPU4000, pooling_largeKernel_Params,
                         PoolingLayerTest::getTestCaseName);
// Large kernel with stride 1
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_LargeKernelStrideOne, PoolingLayerTest_NPU4000,
                         pooling_largeKernelStrideOne, PoolingLayerTest::getTestCaseName);
// all PadType
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AllPadType, PoolingLayerTest_NPU4000, poolAllPadTypeParams,
                         PoolingLayerTest::getTestCaseName);
// 3D usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_3D, PoolingLayerTest_NPU4000, pool3DParams, PoolingLayerTest::getTestCaseName);
// 5d usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_5D, PoolingLayerTest_NPU4000, pool5DParams, PoolingLayerTest::getTestCaseName);
// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_BigPadEndParams, PoolingLayerTest_NPU4000, pooligBigPadEndParams,
                         PoolingLayerTest_NPU4000::getTestCaseName);
// Sanity FP32 kernel check
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_F32, PoolingLayerTest_NPU4000_F32, pool_basic_Params,
                         PoolingLayerTest::getTestCaseName);
// Integer Input/Output
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_InputOutputInteger, PoolingLayerTest_NPU4000, pool_inputOutputInteger,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// SOB multi-clustering
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_SOB, PoolingLayerTest_NPU4000_SOB,
                         generateSOBParamsFromInputShape({{4, 64, 28, 28}}),
                         PoolingLayerTest_NPU4000_SOB::getTestCaseName);
// previous cip reused tests
/* ============= AutoPadValid ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU4000, pool_AutoPadValid,
                                             PoolingLayerTest::getTestCaseName);
/* ============= ExplicitPadding ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU4000,
                                             pool_ExplicitPadding, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricKernel ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU4000,
                                             pool_AsymmetricKernel, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricStrides ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU4000,
                                             pool_AsymmetricStrides, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsX ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsY ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernelsY, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large Prime Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU4000,
                                             avgPool_largePrimeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AvgPooling / Exclude_Pad Handling ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_avgPool_excludePad, PoolingLayerTest_NPU4000, avgPool_excludePad,
                                             PoolingLayerTest::getTestCaseName);

// Max pool ported tests from NPU4000
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU4000, pool_LargeSize1,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU4000, pool_LargeSize2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU4000, pool_LargeStrides,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU4000, pool_LargePadding2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU4000, pool_LargePadding3,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU4000, pool_LargePadding4,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU4000, pool_LargePadding5,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU4000, pool_LargePadding6,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU4000, pool_LargePadding7,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU4000, pool_LargePadding8,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernels, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernelsY, PoolingLayerTest::getTestCaseName);
/* ============= MaxPool8 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_MaxPool8_ExplicitPad_FloorRounding, MaxPoolingV8LayerTestCommon,
                         maxPool8_ExplicitPad_FloorRounding_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_SameUpperPad_FloorRounding, MaxPoolingV8LayerTestCommon,
                         maxPool8_SameUpperPad_FloorRounding_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_SameLowerPad_FloorRounding, MaxPoolingV8LayerTestCommon,
                         maxPool8_SameLowerPad_FloorRounding_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_ExplicitPad_CeilRounding, MaxPoolingV8LayerTestCommon,
                         maxPool8_ExplicitPad_CeilRounding_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_ExplicitPad_FloorRounding5D, MaxPoolingV8LayerTestCommon,
                         maxPool8_ExplicitPad_FloorRounding_5Dinput_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_SameUpperPad_CeilRounding5D, MaxPoolingV8LayerTestCommon,
                         maxPool8_SameUpperPad_FloorRounding_5Dinput_Params, MaxPoolingV8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_SameLowerPad_CeilRounding5D, MaxPoolingV8LayerTestCommon,
                         maxPool8_SameLowerPad_CeilRounding_5Dinput_Params, MaxPoolingV8LayerTest::getTestCaseName);

}  // namespace
