//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include <openvino/opsets/opset1.hpp>

#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using DynamicReduceMeanParams = std::tuple<ov::test::InputShape,  // Input shape
                                           ov::element::Type>;    // Input precision

// Spatial ReduceMean over bounded H and W with a static [1, C, 1, 1] result.
class DynamicReduceMeanTest : public VpuOv2LayerTest, public testing::WithParamInterface<DynamicReduceMeanParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicReduceMeanParams>& obj) {
        const auto& [inShape, inType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape={" << inShape.first.to_string() << "}" << sep;
        result << "TargetShape={" << ov::test::utils::vec2str(inShape.second.front()) << "}" << sep;
        result << "InputType=" << inType;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& shape = targetInputStaticShapes[0];
        auto tensor = ov::Tensor(funcInputs[0].get_element_type(), shape);
        if (tensor.get_element_type() == ov::element::f16) {
            fillChannelRowDependent(tensor.data<ov::float16>(), shape);
        } else {
            fillChannelRowDependent(tensor.data<float>(), shape);
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }

private:
    // Channel- and row-dependent values make a wrong channel or row stride change the mean.
    template <typename T>
    static void fillChannelRowDependent(T* data, const ov::Shape& shape) {
        const auto channels = shape[1], height = shape[2], width = shape[3];
        for (size_t n = 0; n < shape[0]; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const float v = 1.0f + 0.5f * c + 2.0f * h / height + 0.25f * ((w % 4) / 4.0f);
                        data[((n * channels + c) * height + h) * width + w] = static_cast<T>(v);
                    }
                }
            }
        }
    }

public:
    void SetUp() override {
        const auto& [inShape, inType] = GetParam();
        init_input_shapes({inShape});

        const auto input = std::make_shared<ov::opset1::Parameter>(inType, inputDynamicShapes[0]);
        const auto axes = ov::opset1::Constant::create(ov::element::i64, {2}, {2, 3});
        const auto mean = std::make_shared<ov::opset1::ReduceMean>(input, axes, /*keep_dims=*/true);
        const auto result = std::make_shared<ov::opset1::Result>(mean);
        function =
                std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "DynamicReduceMean");
    }
};

TEST_P(DynamicReduceMeanTest, NPU4000_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    // NPU40XX does not default to FWLM_V1_PAGES, which the DMA kernel selection requires.
    configuration[ov::intel_npu::compilation_mode_params.name()] = "workload-management-mode=FWLM_V1_PAGES";
    run(Platform::NPU4000);
}

TEST_P(DynamicReduceMeanTest, NPU5010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(DynamicReduceMeanTest, NPU6010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU6010);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::test::InputShape boundedSpatialShape(const ov::Shape& target) {
    return {ov::PartialShape{1, 32, ov::Dimension(1, 540), ov::Dimension(1, 960)}, {target}};
}

const std::vector<ov::test::InputShape> largeInputShapes = {
        boundedSpatialShape({1, 32, 540, 960}),
        boundedSpatialShape({1, 32, 384, 640}),
};

const std::vector<ov::test::InputShape> smallInputShapes = {
        boundedSpatialShape({1, 32, 17, 31}),
        boundedSpatialShape({1, 32, 1, 1}),
};

const std::vector<ov::test::InputShape> allInputShapes = [] {
    auto shapes = largeInputShapes;
    shapes.insert(shapes.end(), smallInputShapes.begin(), smallInputShapes.end());
    return shapes;
}();

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReduceMean_HC_f16, DynamicReduceMeanTest,
                         ::testing::Combine(::testing::ValuesIn(allInputShapes), ::testing::Values(ov::element::f16)),
                         DynamicReduceMeanTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReduceMean_HC_f32, DynamicReduceMeanTest,
                         ::testing::Combine(::testing::ValuesIn(largeInputShapes), ::testing::Values(ov::element::f32)),
                         DynamicReduceMeanTest::getTestCaseName);

// H*W = 2^25: with these input values, a single FP32 running sum makes the mean about 4% low.
// One channel keeps the input at 64 MB.
const std::vector<ov::test::InputShape> longReductionShapes = {
        {ov::PartialShape{1, 1, ov::Dimension(1, 4096), ov::Dimension(1, 8192)}, {{1, 1, 4096, 8192}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReduceMean_HC_long_f16, DynamicReduceMeanTest,
                         ::testing::Combine(::testing::ValuesIn(longReductionShapes),
                                            ::testing::Values(ov::element::f16)),
                         DynamicReduceMeanTest::getTestCaseName);

}  // namespace
