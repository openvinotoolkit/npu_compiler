//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1_decl.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"

#include "shared_test_classes/subgraph/ws_base.hpp"

namespace ov::test {

class Ws6DInputForInit : public WsBaseTest {
public:
    Ws6DInputForInit(): WsBaseTest("ws_6d_input_for_init") {
    }

private:
    void SetUp() override {
        const ov::Shape inputShape5D{3, 4, 5, 6, 7};
        init_input_shapes(static_shapes_to_test_representation({inputShape5D}));

        ov::ParameterVector params{std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes[0])};
        params[0]->set_friendly_name("input");
        params[0]->get_output_tensor(0).set_names({"input"});

        // Reshape parameter from 5D to 6D
        const auto paramTargetShapeConst = std::make_shared<ov::opset1::Constant>(
                ov::element::i64, ov::Shape{6}, std::vector<int64_t>{1, 3, 4, 5, 6, 7});
        const auto paramReshape = std::make_shared<ov::opset1::Reshape>(params[0], paramTargetShapeConst, false);

        // Create constant 6D tensor with f32 type for Add
        const ov::Shape constShape6D{1, 3, 4, 5, 6, 7};
        const size_t totalSize6D = ov::shape_size(constShape6D);
        std::vector<float> constData(totalSize6D);
        // avoid splatness
        for (size_t i = 0; i < totalSize6D; ++i) {
            constData[i] = static_cast<float>(i % 100) / 10.0f;
        }

        // TODO: E#215401 This constant is suitable for Init schedule in the WS pipeline; however, it is excluded
        // because the tensor rank exceeds 5D.
        const auto constInput6D = std::make_shared<ov::opset1::Constant>(ov::element::f32, constShape6D, constData);

        const auto add = std::make_shared<ov::op::v1::Add>(paramReshape, constInput6D);

        // ReduceMean to reduce from 6D to 4D (reduce along axes 0 and 1)
        const auto reduceAxes =
                std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1});
        const auto reduceMean = std::make_shared<ov::op::v1::ReduceMean>(add, reduceAxes, false);

        // Create constant 4D tensor with f32 type for Multiply
        const ov::Shape mulConstShape{4, 5, 6, 7};
        const size_t totalSize4D = ov::shape_size(mulConstShape);
        std::vector<float> mulConstData(totalSize4D);
        // avoid splatness
        for (size_t i = 0; i < totalSize4D; ++i) {
            mulConstData[i] = 1.0f + static_cast<float>(i % 50) / 25.0f;
        }
        const auto mulConst = std::make_shared<ov::opset1::Constant>(ov::element::f32, mulConstShape, mulConstData);

        // This node is needed to ensure that the Init schedule is not empty.
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(reduceMean, mulConst);

        const ov::ResultVector results{std::make_shared<ov::opset1::Result>(multiply)};
        results[0]->set_friendly_name("output");
        multiply->get_output_tensor(0).set_names({"output"});

        const auto model = std::make_shared<ov::Model>(results, params, "Ws6DInputForInit");

        // Note: Model serialization is required due to the nature of the Weight Separation (WS) feature, as it relies
        // on the WeightlessCacheAttribute. This attribute is populated during deserialization, which does not normally
        // occur in typical functional tests.
        ov::pass::Serialize(_xmlPath, _binPath).run_on_model(model);
        function = core->read_model(_xmlPath, _binPath);
    }
};

TEST_F(Ws6DInputForInit, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(Ws6DInputForInit, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(Ws6DInputForInit, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test
