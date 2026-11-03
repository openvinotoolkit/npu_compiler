//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/core/type/float16.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

#include <cmath>
#include <random>

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
        // i8->f16 per-row fast path (scale NxCxHx1): W not a multiple of the SIMD width
        // exercises the vector tail, W < SIMD width exercises the zero-padded short-row path.
        {{1, 64, 1, 100}, {1, 64, 1, 1}, ov::element::i8, ov::element::f16},
        {{1, 64, 1, 8}, {1, 64, 1, 1}, ov::element::i8, ov::element::f16},
        // i8->f16 per-column fast path (scale NxCx1xW): exact-multiple, tail and short-row.
        {{3, 30, 128}, {3, 1, 128}, ov::element::i8, ov::element::f16},
        {{3, 30, 100}, {3, 1, 100}, ov::element::i8, ov::element::f16},
        {{3, 30, 12}, {3, 1, 12}, ov::element::i8, ov::element::f16},
        {{1, 64, 1, 128}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 64, 1, 256}, {1, 64, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 128, 1, 768}, {1, 128, 1, 1}, ov::element::i4, ov::element::f16},
        {{1, 10, 64, 128}, {1, 10, 1, 128}, ov::element::i8, ov::element::f16},
        {{1, 10, 8192, 128}, {1, 10, 1, 128}, ov::element::i8, ov::element::f16},
};
const std::vector<DynDeQuantParams> paramsI4 = {
        {{3, 30, 128}, {3, 30, 1}, ov::element::i4, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i4, ov::element::f16},
        {{3, 14, 12}, {3, 1, 12}, ov::element::i4, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::u8, ov::element::f16},
        {{128, 128}, {128, 1}, ov::element::i8, ov::element::f16},
        // Same i8->f16 fast-path tail / short-row / per-column coverage as `params`.
        {{1, 64, 1, 100}, {1, 64, 1, 1}, ov::element::i8, ov::element::f16},
        {{1, 64, 1, 8}, {1, 64, 1, 1}, ov::element::i8, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i8, ov::element::f16},
        {{3, 30, 100}, {3, 1, 100}, ov::element::i8, ov::element::f16},
        {{3, 30, 12}, {3, 1, 12}, ov::element::i8, ov::element::f16},
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
        // i8->f16 per-row asymmetric fast path (dyn_dequant_i8_f16_row_zp): W=128 above is an
        // exact multiple of the SIMD width, so these add the vector-tail (W=100) and the
        // zero-padded short-row (W=8 < SIMD width) cases.
        {{128, 100}, {128, 1}, ov::element::i8, ov::element::f16},
        {{128, 8}, {128, 1}, ov::element::i8, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQAsym, DynDQAsymTestCommon, ::testing::ValuesIn(paramsAsym),
                         DynDQAsymTestCommon::getTestCaseName);

//
// u16 -> f16 dynamic dequantization.
//
// The kernel deliberately computes u16 -> f32 -> multiply -> f16. f16 carries an
// 11-bit mantissa and saturates at 65504, so converting the input first either
// truncates it (2049 -> 2048) or overflows outright (>= 65520 -> infinity).
//
class DynDQU16Test : public DynDQTestCommon {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        static const uint16_t kEdge[] = {0, 1, 2048, 2049, 32769, 65504, 65520, 65523, 65527, 65531, 65535};
        constexpr size_t kEdgeCount = sizeof(kEdge) / sizeof(kEdge[0]);

        auto data = ov::Tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0]);
        auto* dataPtr = data.data<uint16_t>();
        const size_t count = data.get_size();
        std::mt19937 gen(1);
        std::uniform_int_distribution<uint32_t> dist(0, 65535);
        for (size_t i = 0; i < count; i++) {
            dataPtr[i] = static_cast<uint16_t>(dist(gen));
        }
        for (size_t k = 0; k < kEdgeCount && k < count; k++) {
            dataPtr[(k * count) / kEdgeCount] = kEdge[k];
        }
        dataPtr[count - 1] = 65535;
        inputs.insert({funcInputs[0].get_node_shared_ptr(), data});

        auto scale = ov::Tensor(funcInputs[1].get_element_type(), targetInputStaticShapes[1]);
        auto* scalePtr = scale.data<ov::float16>();
        for (size_t i = 0; i < scale.get_size(); i++) {
            scalePtr[i] = ov::float16(std::ldexp(1.0f, -static_cast<int>(1 + i % 8)));
        }
        inputs.insert({funcInputs[1].get_node_shared_ptr(), scale});
    }

public:
    void SetUp() override {
        DynDQTestCommon::SetUp();
        convert_precisions = {{ov::element::f16, ov::element::f32}};
        outType = ov::element::f16;
        rel_threshold = 0.0;
    }
};

TEST_P(DynDQU16Test, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynDQU16Test, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynDQU16Test, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynDQU16Test, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DynDeQuantParams> paramsU16 = {
        {{1, 1, 1, 5120}, {1, 1, 1, 1}, ov::element::u16, ov::element::f16},
        {{1, 64, 1, 100}, {1, 64, 1, 1}, ov::element::u16, ov::element::f16},
        {{1, 64, 1, 8}, {1, 64, 1, 1}, ov::element::u16, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQ_U16, DynDQU16Test, ::testing::ValuesIn(paramsU16), DynDQU16Test::getTestCaseName);

}  // namespace ov::test::subgraph
