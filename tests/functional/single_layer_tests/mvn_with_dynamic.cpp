//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/mvn.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

namespace ov::test::subgraph {

using MVNEpsMode = ov::op::MVNEpsMode;
using DynamicMVNParams = std::tuple<ov::test::InputShape, std::vector<int64_t>, bool, float, MVNEpsMode>;

class DynamicMVNLayerTest : public VpuOv2LayerTest, public ::testing::WithParamInterface<DynamicMVNParams> {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        for (const auto& shape : targetInputStaticShapes) {
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 10, -5, 1);
            inputs[function->get_parameters().at(0)] = tensor;
        }
    }

protected:
    void SetUp() override {
        auto params = GetParam();
        const auto& dataShape = std::get<0>(params);
        const auto& axes = std::get<1>(params);
        bool normalizeVariance = std::get<2>(params);
        float eps = std::get<3>(params);
        MVNEpsMode epsMode = std::get<4>(params);

        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);

        const auto mvn = std::make_shared<ov::opset6::MVN>(param, axesConst, normalizeVariance, eps, epsMode);
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(mvn->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicMVN");
    }
};

const std::vector<std::tuple<ov::test::InputShape, std::vector<int64_t>>> inShapesMvnDynamicUnknownTargetShape = {
        {ov::test::InputShape{{1, ov::Dimension(2, 16)}, {{1, 16}}}, {1}},
        {ov::test::InputShape{{1, ov::Dimension(1, 5), 32}, {{1, 5, 32}}}, {1, 2}},
        {ov::test::InputShape{{1, ov::Dimension(1, 5), 768, 1}, {{1, 5, 768, 1}}}, {1, 2, 3}},
        {ov::test::InputShape{{1, 5, ov::Dimension(1, 10), 16, 32}, {{1, 5, 10, 16, 32}}}, {2, 3, 4}}};

const std::vector<bool> normalizeVarianceValues = {true, false};
const std::vector<float> epsValues = {1e-9f, 1e-3f};
const std::vector<MVNEpsMode> epsModeValues = {MVNEpsMode::INSIDE_SQRT};

std::vector<DynamicMVNParams> generateDynamicMVNParams(
        const std::vector<std::tuple<ov::test::InputShape, std::vector<int64_t>>>& shapesAndAxes,
        const std::vector<bool>& normalizeVarianceValues, const std::vector<float>& epsValues,
        const std::vector<MVNEpsMode>& epsModeValues) {
    std::vector<DynamicMVNParams> result;
    for (const auto& [inputShape, axes] : shapesAndAxes) {
        for (bool normVar : normalizeVarianceValues) {
            for (float eps : epsValues) {
                for (MVNEpsMode epsMode : epsModeValues) {
                    result.emplace_back(inputShape, axes, normVar, eps, epsMode);
                }
            }
        }
    }
    return result;
}

TEST_P(DynamicMVNLayerTest, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicMVNLayerTest, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(DynamicMVN, DynamicMVNLayerTest,
                         ::testing::ValuesIn(generateDynamicMVNParams(inShapesMvnDynamicUnknownTargetShape,
                                                                      normalizeVarianceValues, epsValues,
                                                                      epsModeValues)));

}  // namespace ov::test::subgraph
