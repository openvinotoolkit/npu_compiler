//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"

using namespace ov;
using namespace element;

namespace ov::test {

class LoopSubGraphTestCommon : public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class LoopSubGraphTestConstExecCond : public LoopSubGraphTestCommon {
public:
    void SetUp() override {
        // Setting up test data
        inType = ov::element::i32;
        std::vector<std::vector<size_t>> exInputShapes, bodyInputShapes;
        exInputShapes = {{1, 3, 2, 2}};
        bodyInputShapes = {{1, 3, 2, 2}};
        const ov::Shape weightsShape{1};

        init_input_shapes(static_shapes_to_test_representation({exInputShapes[0]}));
        ov::ParameterVector exParams, bodyInputParams;
        const auto const_trip_count = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {3});
        const auto const_ex_exec_cond = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{weightsShape}, {1});
        for (const auto& shape : exInputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        for (const auto& shape : bodyInputShapes) {
            bodyInputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }

        const auto const_0 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {1});
        const auto const_in_exec_cond = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{weightsShape}, {1});

        // Setting up body
        auto add_0 = std::make_shared<op::v1::Add>(bodyInputParams[0], const_0);
        auto body_result_0 = std::make_shared<op::v0::Result>(add_0);
        auto body_result_1 = std::make_shared<op::v0::Result>(const_in_exec_cond);
        auto body_module = std::make_shared<ov::Model>(OutputVector{body_result_0, body_result_1}, bodyInputParams);

        auto loop = std::make_shared<ov::op::v5::Loop>(const_trip_count, const_ex_exec_cond);
        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 1});
        loop->set_function(body_module);

        loop->set_merged_input(bodyInputParams[0], exParams[0], body_result_0);
        loop->get_iter_value(body_result_0);

        auto result0 = loop->output(0);
        function = std::make_shared<ov::Model>(OutputVector{result0}, exParams, "LoopTest");
    }
};

// Slice input, merged input and concat output cases
TEST_F(LoopSubGraphTestConstExecCond, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(LoopSubGraphTestConstExecCond, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(LoopSubGraphTestConstExecCond, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

class LoopSubGraphTestParamExecCond1 : public LoopSubGraphTestCommon {
public:
    void SetUp() override {
        // Setting up test data
        inType = ov::element::i32;
        std::vector<std::vector<size_t>> exInputShapes, bodyInputShapes;
        exInputShapes = {{1, 3, 2, 2}};
        bodyInputShapes = {{1, 3, 2, 2}, {1}};
        const ov::Shape weightsShape{1};

        init_input_shapes(static_shapes_to_test_representation({exInputShapes[0]}));
        ov::ParameterVector exParams, bodyInputParams;
        const auto const_trip_count = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {3});
        const auto const_ex_exec_cond = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{weightsShape}, {1});
        for (const auto& shape : exInputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        for (const auto& shape : bodyInputShapes) {
            bodyInputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }

        const auto const_0 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {10});

        // Setting up body
        auto add_0 = std::make_shared<op::v1::Add>(bodyInputParams[0], const_0);
        auto less_0 = std::make_shared<op::v1::Less>(bodyInputParams[1], const_0);
        auto body_result_0 = std::make_shared<op::v0::Result>(add_0);
        auto body_result_1 = std::make_shared<op::v0::Result>(less_0);
        auto body_module = std::make_shared<ov::Model>(OutputVector{body_result_0, body_result_1}, bodyInputParams);

        auto loop = std::make_shared<ov::op::v5::Loop>(const_trip_count, const_ex_exec_cond);
        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});
        loop->set_function(body_module);

        loop->set_merged_input(bodyInputParams[0], exParams[0], body_result_0);
        loop->get_iter_value(body_result_0);

        auto result0 = loop->output(0);
        function = std::make_shared<ov::Model>(OutputVector{result0}, exParams, "LoopTest");
    }
};

// Slice input, merged input and concat output cases
TEST_F(LoopSubGraphTestParamExecCond1, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(LoopSubGraphTestParamExecCond1, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_F(LoopSubGraphTestParamExecCond1, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

class LoopSubGraphTestParamExecCond2 : public LoopSubGraphTestCommon {
public:
    void SetUp() override {
        // Setting up test data
        inType = ov::element::i32;
        std::vector<std::vector<size_t>> exInputShapes, bodyInputShapes;
        exInputShapes = {{1, 5, 3, 2}};
        bodyInputShapes = {{1, 5, 3, 2}, {1}};
        const ov::Shape weightsShape{1};

        init_input_shapes(static_shapes_to_test_representation({exInputShapes[0]}));
        ov::ParameterVector exParams, bodyInputParams;
        const auto const_trip_count = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {7});
        const auto const_ex_exec_cond = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{weightsShape}, {1});
        for (const auto& shape : exInputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        for (const auto& shape : bodyInputShapes) {
            bodyInputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }

        const auto const_0 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{weightsShape}, {4});

        // Setting up body
        auto add_0 = std::make_shared<op::v1::Add>(bodyInputParams[0], const_0);
        auto less_0 = std::make_shared<op::v1::Less>(bodyInputParams[1], const_0);
        auto body_result_0 = std::make_shared<op::v0::Result>(add_0);
        auto body_result_1 = std::make_shared<op::v0::Result>(less_0);
        auto body_module = std::make_shared<ov::Model>(OutputVector{body_result_0, body_result_1}, bodyInputParams);

        auto loop = std::make_shared<ov::op::v5::Loop>(const_trip_count, const_ex_exec_cond);
        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});
        loop->set_function(body_module);

        loop->set_merged_input(bodyInputParams[0], exParams[0], body_result_0);
        loop->get_iter_value(body_result_0);

        auto result0 = loop->output(0);
        function = std::make_shared<ov::Model>(OutputVector{result0}, exParams, "LoopTest");
    }
};

// Slice input, merged input and concat output cases
TEST_F(LoopSubGraphTestParamExecCond2, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(LoopSubGraphTestParamExecCond2, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(LoopSubGraphTestParamExecCond2, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
