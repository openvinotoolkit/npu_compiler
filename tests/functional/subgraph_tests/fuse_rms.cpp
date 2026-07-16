//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/rms.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <algorithm>

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

template <class ShapeType>
using IODescription = std::tuple<ShapeType, std::optional<ov::Layout>>;

template <class ShapeType>
using IODescriptions = std::vector<IODescription<ShapeType>>;

using RMSNormDecompositionParams = std::tuple<IODescription<ov::Shape>,  // input shapes & optional layouts
                                              ov::element::Type,         // input precision
                                              bool,                      // useSelect (true = Select+Equal, false = Add)
                                              IODescription<ov::Shape>>;  // gamma shape (empty = same as input)

class FuseRMSTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RMSNormDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        IODescription<ov::Shape> input;
        ov::element::Type inputPrecision;
        bool useSelect;
        IODescription<ov::Shape> gammaOutput;
        std::tie(input, inputPrecision, useSelect, gammaOutput) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        auto [inputShape, inputLayout] = input;
        auto [gammaShape, gammaLayout] = gammaOutput;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape=" << inputShape << sep;
        if (inputLayout.has_value()) {
            result << "InputLayout=" << inputLayout->to_string() << sep;
        }
        result << "GammaShape=" << (gammaShape.empty() ? ov::Shape{inputShape.back()} : gammaShape) << sep;
        if (gammaLayout.has_value()) {
            result << "GammaLayout=" << gammaLayout->to_string() << sep;
        }
        result << "TestIdx=" << obj.index << sep;
        result << "useSelect=" << useSelect << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 1, 1, 10);
        auto* data = tensorData.data<float>();
        for (size_t i = 0; i < tensorData.get_size(); ++i) {
            data[i] *= 1e-2f;
        }

        if (!targetInputStaticShapes.empty() && !targetInputStaticShapes[0].empty()) {
            const size_t rowWidth = targetInputStaticShapes[0].back();
            const size_t rowCount = rowWidth == 0 ? 0 : tensorData.get_size() / rowWidth;
            const size_t zeroRows = rowCount > 1 ? rowCount / 2 : (rowCount == 1 ? 1 : 0);
            const size_t zeroElems = std::min(tensorData.get_size(), zeroRows * rowWidth);
            if (zeroElems > 0) {
                std::fill_n(data, zeroElems, 0.0f);
            }
        }

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Model> init_subgraph(const IODescriptions<ov::PartialShape>& input_info,
                                             const ov::Shape& target_shape, const ov::element::Type input_precision,
                                             bool useSelect, const IODescription<ov::Shape>& gamma_output) {
        ov::ParameterVector params;
        for (const auto& [input_shape, input_layout] : input_info) {
            auto param = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shape);
            if (input_layout.has_value()) {
                param->set_layout(input_layout.value());
            }
            params.push_back(std::move(param));
        }

        // x^2
        auto power_const = ov::op::v0::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::op::v0::Constant::create(input_precision, {}, {1e-5f});
        std::shared_ptr<ov::Node> mean_plus_eps = std::make_shared<ov::op::v1::Add>(mean, eps);
        if (useSelect) {
            // Pattern: ReduceMean -> Equal -> Select -> Sqrt
            //                 0  -> Equal  ->|
            //                        Const ->|
            auto zero_const = ov::op::v0::Constant::create(input_precision, {}, {0.f});
            auto equal_cond = std::make_shared<ov::op::v1::Equal>(mean, zero_const);
            mean_plus_eps = std::make_shared<ov::op::v1::Select>(equal_cond, eps, mean);
        }
        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(mean_plus_eps);
        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::op::v0::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        // empty gamma_shape means use input's last dimension
        // empty gamma_layout means use first input's layout
        auto [gamma_shape, gamma_layout] = gamma_output;
        ov::Shape actual_gamma_shape = gamma_shape.empty() ? ov::Shape{target_shape.back()} : gamma_shape;
        if (!gamma_layout.has_value()) {
            gamma_layout = std::get<1>(input_info[0]);
        }
        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, actual_gamma_shape);
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f16);
        std::shared_ptr<ov::op::v0::Result> res = std::make_shared<ov::op::v0::Result>(comp);
        if (gamma_layout.has_value()) {
            res->set_layout(gamma_layout.value());
        }
        return std::make_shared<ov::Model>(ov::OutputVector{res}, params, "RMSNormDecomposition");
    }
    void SetUp() override {
        IODescription<ov::Shape> input;
        ov::element::Type input_precision;
        bool useSelect;
        IODescription<ov::Shape> gamma_output;

        std::tie(input, input_precision, useSelect, gamma_output) = GetParam();
        const auto& [input_shapes, input_layout] = input;
        inType = outType = input_precision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shapes}));

        IODescriptions<ov::PartialShape> input_infos = {std::make_tuple(inputDynamicShapes.front(), input_layout)};
        function = init_subgraph(input_infos, input_shapes, input_precision, useSelect, gamma_output);
    }
};

TEST_P(FuseRMSTestCommon, NPU3720_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseRMSTestCommon, NPU4000_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseRMSTestCommon, NPU5010_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(FuseRMSTestCommon, NPU5020_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-159644
    setBatchCompilerMode("unroll");
    run(Platform::NPU5020);
}

namespace {
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
const std::vector<bool> use_select_options = {true, false};  // false = Add, true = Select+Equal

const IODescriptions<ov::Shape> input_shapes_basic = {
        {IODescription<ov::Shape>{{1, 2, 6}, {}}, IODescription<ov::Shape>{{2, 2, 6}, {"CHW"}}}};
const IODescriptions<ov::Shape> input_shapes = {{{32}, {"C"}},          {{3, 32}, {"HW"}},   {{1, 32, 16}, {}},
                                                {{1, 4, 16, 16}, {}},   {{1, 3, 4, 64}, {}}, {{1, 128, 1, 128}, {}},
                                                {{1, 128, 3, 256}, {}}, {{1, 77, 4096}, {}}};

const IODescriptions<ov::Shape> gamma_shapes_default = {{}};
const IODescriptions<ov::Shape> gamma_shapes_broadcast = {{{1, 1, 1}, {}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(use_select_options),
                                            ::testing::ValuesIn(gamma_shapes_default)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(use_select_options),
                                            ::testing::ValuesIn(gamma_shapes_default)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS_BroadcastGamma, FuseRMSTestCommon,
                         ::testing::Combine(::testing::Values(IODescription<ov::Shape>{{1, 4, 128}, {}}),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(use_select_options),
                                            ::testing::ValuesIn(gamma_shapes_broadcast)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS_3072, FuseRMSTestCommon,
                         ::testing::Combine(::testing::Values(IODescription<ov::Shape>{{1, 1, 64, 3072}, {}}),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(use_select_options),
                                            ::testing::ValuesIn(gamma_shapes_default)),
                         FuseRMSTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
