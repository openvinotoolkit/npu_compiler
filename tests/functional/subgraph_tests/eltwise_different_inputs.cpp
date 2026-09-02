//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/add.hpp"

namespace ov::test {

class EltwiseAddQuantizedSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ov::element::Type> {
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 56, 56};
        const ov::Shape weightsShape{1, 16, 56, 56};
        inType = outType = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape, weightsShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        params[0]->set_friendly_name("input1");
        params[1]->set_friendly_name("input2");

        const size_t dataLevels = 256;
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {}, {0.0},
                                                                {12.583984375}, {0.0}, {12.583984375});

        const size_t weightsLevels = 256;
        const auto weightsFq = ov::test::utils::make_fake_quantize(params[1], ov::element::f16, weightsLevels, {},
                                                                   {0.0}, {2.583984375}, {0.0}, {2.583984375});

        const auto addOp = std::make_shared<ov::op::v1::Add>(dataFq, weightsFq);

        const size_t outLevels = 256;
        const auto outputFq = ov::test::utils::make_fake_quantize(addOp, ov::element::f16, outLevels, {}, {0.0},
                                                                  {13.583984375}, {0.0}, {13.583984375});

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "EltwiseAddQuantized");
        abs_threshold = 0.06f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class EltwiseAddQuantizedFuseOutstandingQuantSubGraphTest_NPU3720 : public EltwiseAddQuantizedSubGraphTest_NPU3720 {};

TEST_P(EltwiseAddQuantizedSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseAddQuantizedFuseOutstandingQuantSubGraphTest_NPU3720, HW) {
    const float fqRange = 14, fqLevels = 256;
    abs_threshold = fqRange / fqLevels;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseAddQuantized, EltwiseAddQuantizedSubGraphTest_NPU3720,
                         ::testing::Values(ov::element::f16), EltwiseAddQuantizedSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseAddQuantized, EltwiseAddQuantizedFuseOutstandingQuantSubGraphTest_NPU3720,
                         ::testing::Values(ov::element::f16),
                         EltwiseAddQuantizedFuseOutstandingQuantSubGraphTest_NPU3720::getTestCaseName);

}  // namespace ov::test
