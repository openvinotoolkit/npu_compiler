//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/proposal.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/proposal.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ProposalLayerTestCommon : public ProposalLayerTest, virtual public VpuOv2LayerTest {
protected:
    void SetUp() override {
        ProposalLayerTest::SetUp();
        auto ops = function->get_ops();

        // should replace hardcoded f32 element type for Constant op
        for (const auto& op : ops) {
            auto proposal = std::dynamic_pointer_cast<ov::op::v0::Proposal>(op);
            if (proposal != nullptr) {
                std::vector<float> img_info = {225.0f, 225.0f, 1.0f};
                auto scores = proposal->input(0);
                auto boxes = proposal->input(1);
                auto image_shape =
                        std::make_shared<ov::op::v0::Constant>(scores.get_element_type(), ov::Shape{3}, img_info);
                proposal->set_arguments(ov::OutputVector{scores.get_source_output(), boxes.get_source_output(),
                                                         image_shape->get_default_output()});
                break;
            }
        }
        ov::NodeVector nodeVector;
        auto params = function->get_parameters();
        for (const auto& op : params) {
            nodeVector.push_back(std::dynamic_pointer_cast<ov::Node>(op));
        }
        std::vector<InputShape> inputShapes = static_shapes_to_test_representation(
                std::vector<ov::Shape>{nodeVector[0]->get_shape(), nodeVector[1]->get_shape()});
        init_input_shapes(inputShapes);
    }

    void validate() override {
        VpuOv2LayerTest::validate();
    }
    int outputSize = 0;
    // "IoU = intersection area / union area" of two boxes A, B
    // A, B: 4-dim array (x1, y1, x2, y2)
    template <class T>
    static T check_iou(const T* A, const T* B) {
        T c0 = T(0.0f);
        T c1 = T(1.0f);
        if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
            return c0;
        } else {
            // overlapped region (= box)
            const T x1 = std::max(A[0], B[0]);
            const T y1 = std::max(A[1], B[1]);
            const T x2 = std::min(A[2], B[2]);
            const T y2 = std::min(A[3], B[3]);

            // intersection area
            const T width = std::max(c0, x2 - x1 + c1);
            const T height = std::max(c0, y2 - y1 + c1);
            const T area = width * height;

            // area of A, B
            const T A_area = (A[2] - A[0] + c1) * (A[3] - A[1] + c1);
            const T B_area = (B[2] - B[0] + c1) * (B[3] - B[1] + c1);

            // IoU
            return area / (A_area + B_area - area);
        }
    }

    template <class T>
    void CompareIou(const T* expected, const T* actual, std::size_t size, T threshold) {
        const T c0 = T(0.0f);
        const float OVERLAP_ROI_COEF = 0.9f;
        const int OUTPUT_ROI_ELEMENT_SIZE = 5;
        auto num_gt = size / OUTPUT_ROI_ELEMENT_SIZE;
        // It will be recalculated base on real num_gt value
        unsigned int THRESHOLD_NUM_MATCH = num_gt * OVERLAP_ROI_COEF;  // Threshold for the number of matched roi
        T THRESHOLD_ROI_OVERLAP = T(0.9);                              // Threshold for ROI overlap restrictions
        int32_t count = 0;
        bool threshold_test_failed = false;
        const auto res = actual;
        const auto& ref = expected;
        for (int i = 0; i < (int)num_gt; i++) {
            T max_iou = c0;
            if (res[i * OUTPUT_ROI_ELEMENT_SIZE] < c0) {  // check if num of roy not finished as was not found
                // expected match just on the real size of the output
                num_gt = i - 1;
                break;
            }
            for (int j = 0; j < num_gt; j++) {
                // if reference finish list signal was found, not use anymore max end of roy size
                if (ref[j * OUTPUT_ROI_ELEMENT_SIZE] < c0) {
                    num_gt = j - 1;
                }
                auto cur_iou = check_iou(&res[i * OUTPUT_ROI_ELEMENT_SIZE + 1],
                                         &ref[j * OUTPUT_ROI_ELEMENT_SIZE + 1]);  // start index 1 to ignore score value
                if (cur_iou > max_iou) {
                    max_iou = cur_iou;
                }
            }
            if (max_iou > THRESHOLD_ROI_OVERLAP) {
                count++;
            }
        }
        THRESHOLD_NUM_MATCH = num_gt * OVERLAP_ROI_COEF;  // Threshold for the number of matched roi
        threshold_test_failed = (count < (int)THRESHOLD_NUM_MATCH) ? true : false;
        outputSize = num_gt;
        ASSERT_TRUE(!threshold_test_failed)
                << "Relative Proposal Iou comparison failed. "
                << "Number element inside output: " << num_gt << " Number match ROI found: " << count
                << " Threashold set to: " << THRESHOLD_NUM_MATCH << " Test Failed!";
    }

    template <class T>
    void CompareScores(const T* actual, std::size_t size) {
        bool scoreDecrease = true;
        int i = 1;
        for (i = 1; i < size; i++) {
            if (actual[i - 1] < actual[i]) {
                scoreDecrease = false;
                break;
            }
        }
        ASSERT_TRUE(scoreDecrease) << "Score decrease mismatch between position: " << (i - 1) << " and position: " << i
                                   << " val " << actual[i - 1] << " and val " << actual[i] << " Test failed.";
    }

    // LayerTestsCommon
    //  Compare base on reference from:
    //  openvino/src/tests_deprecated/functional/vpu/common/layers/myriad_layers_proposal_test.cpp
    // and from previous implementation:
    // See
    // ${NPU_FIRMWARE_SOURCES_PATH}/blob/develop/validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/leon/tests/exp_generate_proposals.cpp
    // Reference compare function from above link check just if from first 20 output ROI 18 of them can be found inside
    // reference with 70% overlap. I consider to extend this verification base on: the output can have less that 20 ROI,
    // if output have 1000 elements to check just first 20 I consider to be not enought; 70% error accepted I supose to
    // be to mutch. So I extend verification in this way: 90% of roy from output should be fund inside reference with an
    // overlap of 90%. This quarantee (I suppose) the the reference is corect, but can be imaginary situation when can
    // fail, even the reference is correct (base of significant number of threashold and computation made in floar for
    // reverence and in fp16 from npu).

    void compare(const std::vector<ov::Tensor>& expectedOutputs,
                 const std::vector<ov::Tensor>& actualOutputs) override {
        // box check
        const auto& expected = expectedOutputs[0];
        const auto& actual = actualOutputs[0];

        const auto* expectedBuffer = expected.data();
        const auto* actualBuffer = actual.data();

        const auto& precision = actual.get_element_type();
        auto size = actual.get_size();

        switch (precision) {
        case ov::element::bf16:
            CompareIou(reinterpret_cast<const ov::bfloat16*>(expectedBuffer),
                       reinterpret_cast<const ov::bfloat16*>(actualBuffer), size, ov::bfloat16(rel_threshold));
            break;
        case ov::element::f16:
            CompareIou(reinterpret_cast<const ov::float16*>(expectedBuffer),
                       reinterpret_cast<const ov::float16*>(actualBuffer), size, ov::float16(rel_threshold));
            break;
        case ov::element::f32:
            CompareIou<float>(reinterpret_cast<const float*>(expectedBuffer),
                              reinterpret_cast<const float*>(actualBuffer), size, rel_threshold);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
        }

        // score output is generated
        if (expectedOutputs.size() > 1) {
            // check if scores are decrescent value until the end of dynamic size
            const auto& scores = actualOutputs[1];
            const auto scoresBuffer = scores.data();
            const auto& precisionScore = scores.get_element_type();
            switch (precisionScore) {
            case ov::element::bf16:
                CompareScores(reinterpret_cast<const ov::bfloat16*>(scoresBuffer), outputSize);
                break;
            case ov::element::f16:
                CompareScores(reinterpret_cast<const ov::float16*>(scoresBuffer), outputSize);
                break;
            case ov::element::f32:
                CompareScores(reinterpret_cast<const float*>(scoresBuffer), outputSize);
                break;
            default:
                FAIL() << "Comparator for " << precisionScore << " precision isn't supported";
            }
        }
        return;
    }
};

TEST_P(ProposalLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ProposalLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ============= Proposal ============= */
const auto proposalParams0 = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(std::vector<size_t>{32}),
                           ::testing::ValuesIn(std::vector<size_t>{2147483647}),
                           ::testing::ValuesIn(std::vector<size_t>{100}),
                           ::testing::ValuesIn(std::vector<float>{0.69999998807907104f}),
                           ::testing::ValuesIn(std::vector<size_t>{1}),
                           ::testing::ValuesIn(std::vector<std::vector<float>>{{0.5f, 1.0f, 2.0f}}),
                           ::testing::ValuesIn(std::vector<std::vector<float>>{{0.25f, 0.5f, 1.0f, 2.0f}}),
                           ::testing::ValuesIn(std::vector<bool>{true}), ::testing::ValuesIn(std::vector<bool>{false}),
                           ::testing::ValuesIn(std::vector<std::string>{"tensowflow"})),
        ::testing::Values(ov::element::f16), ::testing::Values(test_utils::TARGET_DEVICE));

const auto proposalParams1 = ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(std::vector<size_t>{4}), ::testing::ValuesIn(std::vector<size_t>{6000}),
                           ::testing::ValuesIn(std::vector<size_t>{300}),
                           ::testing::ValuesIn(std::vector<float>{0.69999998807907104f}),
                           ::testing::ValuesIn(std::vector<size_t>{4}),
                           ::testing::ValuesIn(std::vector<std::vector<float>>{{0.5f}}),
                           ::testing::ValuesIn(std::vector<std::vector<float>>{{1.2f}}),
                           ::testing::ValuesIn(std::vector<bool>{true}), ::testing::ValuesIn(std::vector<bool>{false}),
                           ::testing::ValuesIn(std::vector<std::string>{""})),
        ::testing::Values(ov::element::f16), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_conformance_tf, ProposalLayerTestCommon, proposalParams0,
                         ProposalLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Proposal, ProposalLayerTestCommon, proposalParams1,
                         ProposalLayerTestCommon::getTestCaseName);

}  // namespace
