//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert_like.hpp"

using namespace ov;
using namespace element;

namespace ov::test {

using ConvertLikeLayerTestParams = std::tuple<ov::Shape, ov::Shape, ov::element::Type, ov::element::Type>;

class ConvertLikeLayerTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvertLikeLayerTestParams> {
protected:
    void SetUp() override {
        ov::Shape data_shape;
        ov::Shape like_shape;
        ov::element::Type data_type;
        ov::element::Type like_type;

        std::tie(data_shape, like_shape, data_type, like_type) = GetParam();

        init_input_shapes(static_shapes_to_test_representation({data_shape}));

        auto data = std::make_shared<op::v0::Parameter>(data_type, data_shape);
        auto like = std::make_shared<op::v0::Constant>(like_type, like_shape);

        auto ConvertLikeOp = std::make_shared<ov::op::v1::ConvertLike>(data, like);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ConvertLikeOp)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "ConvertLikeTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertLikeLayerTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(ConvertLikeLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvertLikeLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(ConvertLikeLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
const TypeVector inType{
        element::f16, element::f32, element::i32, element::i8, element::u8,
};

const std::vector<ov::Shape> data_Shapes = {{2, 2}};
const std::vector<ov::Shape> like_Shapes = {{2}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertLikeTest, ConvertLikeLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(data_Shapes), ::testing::ValuesIn(like_Shapes),
                                            ::testing::ValuesIn(inType), ::testing::ValuesIn(inType)),
                         ConvertLikeLayerTestCommon::getTestCaseName);
}  // namespace ov::test
