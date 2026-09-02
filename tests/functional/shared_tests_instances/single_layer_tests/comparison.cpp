//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/comparison.hpp"
#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ComparisonLayerTest);

class ComparisonLayerTestCommon : public ComparisonLayerTest, virtual public VpuOv2LayerTest {
protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        ComparisonTypes comparisonOpType;
        InputLayerType secondInputType;
        ov::element::Type modelType;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, comparisonOpType, secondInputType, modelType, std::ignore, additionalConfig) =
                this->GetParam();

        init_input_shapes(inputShapes);

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(modelType, inputShapes[0].second[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape(inputShapes[1].second[0]));
            secondInput = param;
            inputs.push_back(param);
        } else {
            auto tensor = create_and_fill_tensor(modelType, ov::Shape(inputShapes[1].second[0]));
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto comparisonNode = make_comparison(inputs[0], secondInput, comparisonOpType);
        auto convertedComparisonNode = std::make_shared<ov::op::v0::Convert>(comparisonNode, modelType);
        function = std::make_shared<ov::Model>(convertedComparisonNode, inputs, "Comparison");
    }
};

// This test class is created for IsNaN, IsInf and IsFinite Op, which overrides OpenVINO's default
// generate_inputs() method. For IsFinite Op, the default generate_inputs() method and comparison::fill_tensor() method
// mix up fp16 and fp32 data types, which leads to core dump when the IsFinite Op test is executed.
// See
// https://github.com/openvinotoolkit/openvino/blob/master/src/tests/functional/base_func_tests/src/base/utils/generate_inputs.cpp#L836
class ComparisonLayerTestIsOp : public ComparisonLayerTestCommon {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0]);
        fill_tensor(tensor);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }

private:
    template <typename T>
    void fill_tensor_typed(ov::Tensor& tensor) {
        const int range = static_cast<int>(ov::shape_size(tensor.get_shape()));
        const float startFrom = -static_cast<float>(range) / 2.f;

        auto pointer = tensor.data<T>();
        testing::internal::Random random(1);
        for (int i = 0; i < range; i++) {
            if (i % 7 == 0) {
                pointer[i] = std::numeric_limits<T>::infinity();
            } else if (i % 7 == 1) {
                pointer[i] = std::numeric_limits<T>::quiet_NaN();
            } else if (i % 7 == 2) {
                if constexpr (std::is_same_v<T, float>) {
                    auto data_ptr_int = reinterpret_cast<int*>(tensor.data());
                    data_ptr_int[i] = 0x7F800000 + random.Generate(range);
                } else {
                    pointer[i] = std::numeric_limits<T>::quiet_NaN();
                }
            } else if (i % 7 == 3) {
                pointer[i] = -std::numeric_limits<T>::infinity();
            } else if (i % 7 == 5) {
                pointer[i] = -std::numeric_limits<T>::quiet_NaN();
            } else {
                pointer[i] = static_cast<T>(startFrom + random.Generate(range));
            }
        }
    }

    void fill_tensor(ov::Tensor& tensor) {
        if (tensor.get_element_type() == ov::element::f16) {
            fill_tensor_typed<ov::float16>(tensor);
        } else if (tensor.get_element_type() == ov::element::f32) {
            fill_tensor_typed<float>(tensor);
        } else {
            OPENVINO_THROW("Unsupported element type: ", tensor.get_element_type());
        }
    }
};

class ComparisonLayerTestDynamic : public ComparisonLayerTest, virtual public VpuOv2LayerTest {
    // A copy of ComparisonLayerTest, because compiler does not support boolean type (it is represented as u8).
    // We cannot use ComparisonLayerTestCommon since it handles inputShapes and inputDynamicShapes differently,
    // which prevents its use with dynamic shapes. The SetUp override can be removed once the compiler supports the
    // boolean type as-is (#109588).
    void SetUp() override {
        std::vector<InputShape> shapes;
        InputLayerType secondInputType;
        std::map<std::string, std::string> additionalConfig;
        ov::element::Type modelType;
        ov::test::utils::ComparisonTypes comparisonOpType;
        std::tie(shapes, comparisonOpType, secondInputType, modelType, targetDevice, additionalConfig) =
                this->GetParam();
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        init_input_shapes(shapes);

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            secondInput = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[1]);
            inputs.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondInput));
        } else {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(modelType, targetStaticShapes.front()[1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto comparisonNode = ov::test::utils::make_comparison(inputs[0], secondInput, comparisonOpType);

        // Workaround added - Convert node to avoid working with boolean as output.
        auto convertedComparisonNode = std::make_shared<ov::op::v0::Convert>(comparisonNode, modelType);
        function = std::make_shared<ov::Model>(convertedComparisonNode, inputs, "Comparison");
    }
};

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ComparisonLayerTest_Tiling);

class ComparisonLayerTest_Tiling : public ComparisonLayerTestCommon {};

class ShaveCodeGenComparisonLayerTestCommon : public ComparisonLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

// The NPU compiler represents boolean as u8 in its compiled metadata (#109588). The Level Zero
// backend allocates I/O buffers using that compiler metadata precision (u8), then during infer()
// copies user tensors into those buffers via ITensor::copy_to(), which requires an exact element
// type match.
//
// infer() - convert boolean inputs to u8 before set_tensor()
// (allowed by the plugin in src\plugins\intel_npu\src\common\src\sync_infer_request.cpp).
// get_plugin_outputs() - convert u8 outputs back to boolean before comparison.
class ComparisonLayerTest_BOOL : public ComparisonLayerTestCommon, virtual public VpuOv2LayerTest {
protected:
    void infer() override {
        inferRequest = compiledModel.create_infer_request();

        auto toU8 = [](const ov::Tensor& src) -> ov::Tensor {
            ov::Tensor dst(ov::element::u8, src.get_shape());
            const auto* s = src.data<const ov::fundamental_type_for<ov::element::boolean>>();
            auto* d = dst.data<uint8_t>();
            for (size_t i = 0; i < src.get_size(); ++i) {
                d[i] = s[i] ? 1 : 0;
            }
            return dst;
        };

        for (const auto& item : inputs) {
            const auto& tensor = item.second;
            inferRequest.set_tensor(item.first,
                                    tensor.get_element_type() == ov::element::boolean ? toU8(tensor) : tensor);
        }

        inferRequest.infer();
    }

    std::vector<ov::Tensor> get_plugin_outputs() override {
        infer();

        auto toBoolean = [](const ov::Tensor& src) -> ov::Tensor {
            ov::Tensor dst(ov::element::boolean, src.get_shape());
            const auto* s = src.data<uint8_t>();
            auto* d = dst.data<ov::fundamental_type_for<ov::element::boolean>>();
            for (size_t i = 0; i < src.get_size(); ++i) {
                d[i] = s[i] != 0;
            }
            return dst;
        };

        std::vector<ov::Tensor> outputs;
        for (size_t i = 0; i < compiledModel.outputs().size(); ++i) {
            auto tensor = inferRequest.get_tensor(compiledModel.output(i));
            outputs.push_back(tensor.get_element_type() == ov::element::u8 ? toBoolean(tensor) : tensor);
        }
        return outputs;
    }
};

TEST_P(ComparisonLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestIsOp, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTest_Tiling, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTest_BOOL, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestDynamic, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTestIsOp, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTestDynamic, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTest_BOOL, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenComparisonLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ComparisonLayerTestIsOp, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ComparisonLayerTestDynamic, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ComparisonLayerTest_BOOL, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ShaveCodeGenComparisonLayerTestCommon, NPU5010) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ComparisonLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(ComparisonLayerTestIsOp, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
std::vector<ComparisonTypes> comparisonOpTypes_MLIR = {
        ComparisonTypes::EQUAL,     ComparisonTypes::LESS,    ComparisonTypes::LESS_EQUAL,
        ComparisonTypes::NOT_EQUAL, ComparisonTypes::GREATER, ComparisonTypes::GREATER_EQUAL,
};

std::vector<ComparisonTypes> comparisonOpTypesDynamic_MLIR = {ComparisonTypes::LESS, ComparisonTypes::LESS_EQUAL,
                                                              ComparisonTypes::EQUAL, ComparisonTypes::GREATER,
                                                              ComparisonTypes::GREATER_EQUAL};

std::vector<InputLayerType> secondInputTypes = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

std::map<std::string, std::string> additionalConfig = {};

auto input_shape_converter = [](const std::vector<std::pair<ov::Shape, ov::Shape>>& shapes) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& shape : shapes) {
        result.push_back({shape.first, shape.second});
    }
    return result;
};

std::map<ov::Shape, std::vector<ov::Shape>> inputShapes = {
        {{5}, {{1}}},
        {{10, 1}, {{1, 50}}},
        {{1, 16, 32}, {{1, 16, 32}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> precommit_inShapes = {{{1, 16, 32}, {{1, 1, 32}}},
                                                                  {{1, 1, 5, 5}, {{1, 1, 1, 1}}}};

std::map<ov::Shape, std::vector<ov::Shape>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

// Only 4D dynamic shapes are supported (3D and 2D require ConvertShapeTo4D support for this layer #163161)
// Test 3 cases - broadcast with max dim, no broadcast and broadcast where input is not a max dim.
std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic_4D = {
        {{{1, 1, ov::Dimension(1, 10), 200}, {{1, 1, 10, 200}, {1, 1, 10, 200}, {1, 1, 5, 200}}},
         {{1, 1, ov::Dimension(1, 10), 200}, {{1, 1, 1, 200}, {1, 1, 10, 200}, {1, 1, 1, 200}}}},
};

std::vector<ov::element::Type> precision = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

auto inputShapesComparisonParams = input_shape_converter(combineParams(inputShapes));
const auto comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesComparisonParams)),
                           ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(precision), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

auto inputShapesPrecommit = input_shape_converter(combineParams(precommit_inShapes));
const auto precommit_comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesPrecommit)),
                           ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(precision), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

auto inputShapesTiling = input_shape_converter(combineParams(tiling_inShapes));
const auto tiling_comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesTiling)),
                           ::testing::Values(ComparisonTypes::EQUAL), ::testing::ValuesIn(secondInputTypes),
                           ::testing::Values(ov::element::f16), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

const auto comparison_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(in_shapes_dynamic_4D), ::testing::ValuesIn(comparisonOpTypesDynamic_MLIR),
        ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(precision),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additionalConfig));

std::vector<ComparisonTypes> comparisonOpTypesBOOL_MLIR = {
        ComparisonTypes::EQUAL,     ComparisonTypes::LESS,    ComparisonTypes::LESS_EQUAL,
        ComparisonTypes::NOT_EQUAL, ComparisonTypes::GREATER,
};
std::vector<ov::element::Type> precision_bool = {ov::element::boolean};

const auto comparison_params_bool =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesComparisonParams)),
                           ::testing::ValuesIn(comparisonOpTypesBOOL_MLIR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(precision_bool), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_Comparison, ComparisonLayerTestCommon, comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison, ComparisonLayerTestCommon, precommit_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Comparison, ComparisonLayerTestCommon, tiling_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

// ShaveCodeGen tests
INSTANTIATE_TEST_SUITE_P(smoke_Comparison, ShaveCodeGenComparisonLayerTestCommon, comparison_params,
                         ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison, ShaveCodeGenComparisonLayerTestCommon, precommit_comparison_params,
                         ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

// E-152367: ShaveCodeGen Tiling support
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_tiling_Comparison, ShaveCodeGenComparisonLayerTestCommon,
                         tiling_comparison_params, ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

// Dynamic shapes cases
INSTANTIATE_TEST_SUITE_P(smoke_ComparisonDynamic, ComparisonLayerTestDynamic, comparison_params_dynamic,
                         ComparisonLayerTestDynamic::getTestCaseName);

// BOOL tests
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_BOOL, ComparisonLayerTest_BOOL, comparison_params_bool,
                         ComparisonLayerTest_BOOL::getTestCaseName);

// isNaN tests
std::vector<std::vector<ov::Shape>> input_shapes_is_ops_static = {{{1}, {1}},
                                                                  {{1, 2}, {1}},
                                                                  {{3, 1}, {1}},
                                                                  {{2, 2}, {1}},
                                                                  {{1, 5, 1}, {1}},
                                                                  {{2, 1, 1, 3, 1}, {1}},
                                                                  {{7, 1, 1, 1, 1}, {1}},
                                                                  {{2, 2, 2}, {1}},
                                                                  {{3, 1, 3, 3}, {1}},
                                                                  {{17}, {1}},
                                                                  {{2, 18}, {1}},
                                                                  {{1, 3, 20}, {1}},
                                                                  {{2, 200}, {1}},
                                                                  {{2, 17, 3, 4}, {1}}};

std::vector<ov::element::Type> is_precision = {
        ov::element::f32,
        ov::element::f16,
};

std::vector<ov::test::utils::ComparisonTypes> comparisonOpTypesIs = {ov::test::utils::ComparisonTypes::IS_NAN,
                                                                     ov::test::utils::ComparisonTypes::IS_INF,
                                                                     ov::test::utils::ComparisonTypes::IS_FINITE};

const auto ComparisonTestParamsIs = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_is_ops_static)),
        ::testing::ValuesIn(comparisonOpTypesIs), ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
        ::testing::ValuesIn(is_precision), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_IsOp, ComparisonLayerTestIsOp, ComparisonTestParamsIs,
                         ComparisonLayerTestIsOp::getTestCaseName);
}  // namespace
