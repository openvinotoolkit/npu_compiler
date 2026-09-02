//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test::subgraph {

using DynDeQuantParams = std::tuple<ov::Shape,           // input
                                    ov::Shape,           // scale
                                    ov::element::Type,   // inputType
                                    ov::element::Type>;  // outputType

class DynDQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynDeQuantParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compiler_dynamic_quantization.name()] = "YES";
    }

public:
    void SetUp() override {
        /* creates subgraph
        input(i4)
           |
        Convert   Scale
              \     /
              Multiply
                 |
               Output
        */
        const auto& [inShape, scaleShape, iType, oType] = GetParam();
        if (utils::getTestDeviceId() == "3720" && iType == ov::element::nf4) {
            GTEST_SKIP() << "nf4 input type is unsupported on NPU3720";
        }

        init_input_shapes(static_shapes_to_test_representation({inShape, scaleShape}));
        const auto input = std::make_shared<ov::opset1::Parameter>(iType, inputDynamicShapes.at(0));
        const auto quantScale = std::make_shared<ov::opset1::Parameter>(oType, inputDynamicShapes.at(1));
        const auto convert0 = std::make_shared<ov::opset1::Convert>(input->output(0), oType);
        const auto mul = std::make_shared<ov::opset1::Multiply>(convert0->output(0), quantScale->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, quantScale}, "DynDQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynDeQuantParams>& obj) {
        const auto& [inShape, scaleShape, iType, oType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "InShape=" << inShape << sep;
        result << "ScaleShape=" << scaleShape;
        result << "inputType=" << iType;
        result << "outputType=" << oType;
        return result.str();
    };
};
class DynDQTestCommonWithoutNF4 : public DynDQTestCommon {};
//
// Platform test definition
//

TEST_P(DynDQTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynDQTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynDQTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(DynDQTestCommon, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DynDeQuantParams> params = {
        {{3, 30, 128}, {3, 30, 1}, ov::element::i4, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i4, ov::element::f16},
        {{3, 14, 12}, {3, 1, 12}, ov::element::i4, ov::element::f16},
        {{16, 8, 32}, {1, 1, 1}, ov::element::nf4, ov::element::f16},
        {{16, 8, 32}, {16, 1, 1}, ov::element::nf4, ov::element::f16},
        {{16, 8, 32}, {1, 1, 32}, ov::element::nf4, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::u8, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::i8, ov::element::f16},
        {{1, 64, 1, 128}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 64, 1, 256}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 128, 1, 768}, {1, 128, 1, 1}, ov::element::i4, ov::element::f16},
};
const std::vector<DynDeQuantParams> paramsI4 = {
        {{3, 30, 128}, {3, 30, 1}, ov::element::i4, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i4, ov::element::f16},
        {{3, 14, 12}, {3, 1, 12}, ov::element::i4, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::u8, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::i8, ov::element::f16},
        {{1, 64, 1, 128}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 64, 1, 256}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 128, 1, 768}, {1, 128, 1, 1}, ov::element::i4, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQ, DynDQTestCommon, ::testing::ValuesIn(params), DynDQTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DynDQ, DynDQTestCommonWithoutNF4, ::testing::ValuesIn(paramsI4),
                         DynDQTestCommonWithoutNF4::getTestCaseName);

class DynDQTestGSSmall : public DynDQTestCommon {};

TEST_P(DynDQTestGSSmall, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynDQTestGSSmall, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynDQTestGSSmall, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynDQTestGSSmall, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DynDeQuantParams> paramsGSSmall = {
        {{80, 1024, 32}, {80, 1024, 1}, ov::element::i4, ov::element::f16},
        {{128, 2560, 32}, {128, 2560, 1}, ov::element::i4, ov::element::f16},
        {{80, 4096, 32}, {80, 4096, 1}, ov::element::i4, ov::element::f16},
        {{80, 1024, 64}, {80, 1024, 1}, ov::element::i4, ov::element::f16},
        {{80, 1024, 96}, {80, 1024, 1}, ov::element::i4, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQ_GSSmall, DynDQTestGSSmall, ::testing::ValuesIn(paramsGSSmall),
                         DynDQTestGSSmall::getTestCaseName);

//
// Asymmetric dynamic dequantization test (Convert -> Subtract(zp) -> Multiply(scale))
// exercises the asymmetric pattern consolidated by ConsolidateWeightsDequantization
// into a single IE::DynamicDequantizeOp. Covers INT8 / UI8 weights with per-channel
// zero-point and per-channel scale.
//

using DynDeQuantAsymParams = std::tuple<ov::Shape,           // weights
                                        ov::Shape,           // zp / scale (broadcastable)
                                        ov::element::Type,   // weightsType (int8/uint8/int4/uint4)
                                        ov::element::Type>;  // outputType (f16)

class DynDQAsymTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynDeQuantAsymParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compiler_dynamic_quantization.name()] = "YES";
    }

public:
    void SetUp() override {
        /* creates subgraph
        weights(int)   zp(int)
            |             |
         Convert       Convert
              \         /
               Subtract     scale(f16)
                    \         /
                    Multiply
                       |
                     Output
        */
        const auto& [weightsShape, zpScaleShape, wType, oType] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({weightsShape, zpScaleShape, zpScaleShape}));
        const auto weights = std::make_shared<ov::opset1::Parameter>(wType, inputDynamicShapes.at(0));
        const auto zp = std::make_shared<ov::opset1::Parameter>(wType, inputDynamicShapes.at(1));
        const auto scale = std::make_shared<ov::opset1::Parameter>(oType, inputDynamicShapes.at(2));

        const auto weightsConvert = std::make_shared<ov::opset1::Convert>(weights->output(0), oType);
        const auto zpConvert = std::make_shared<ov::opset1::Convert>(zp->output(0), oType);
        const auto subtract = std::make_shared<ov::opset1::Subtract>(weightsConvert->output(0), zpConvert->output(0));
        const auto mul = std::make_shared<ov::opset1::Multiply>(subtract->output(0), scale->output(0));

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{weights, zp, scale}, "DynDQAsym");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynDeQuantAsymParams>& obj) {
        const auto& [weightsShape, zpScaleShape, wType, oType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "WeightsShape=" << weightsShape << sep;
        result << "ZpScaleShape=" << zpScaleShape << sep;
        result << "weightsType=" << wType << sep;
        result << "outputType=" << oType;
        return result.str();
    };
};

TEST_P(DynDQAsymTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynDQAsymTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynDQAsymTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynDQAsymTestCommon, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DynDeQuantAsymParams> paramsAsym = {
        {{128, 128}, {128, 1}, ov::element::i8, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::u8, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQAsym, DynDQAsymTestCommon, ::testing::ValuesIn(paramsAsym),
                         DynDQAsymTestCommon::getTestCaseName);
}  // namespace ov::test::subgraph
