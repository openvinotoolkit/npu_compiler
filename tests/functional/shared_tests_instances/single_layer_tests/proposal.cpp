//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/proposal.hpp"

namespace LayerTestsDefinitions {

class KmbProposalLayerTest : public ProposalLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
protected:
    void Validate() override {
        LayerTestsUtils::KmbLayerTestsCommon::Validate();
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
    // ${VPU_FIRMWARE_SOURCES_PATH}/blob/develop/validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/leon/tests/exp_generate_proposals.cpp
    // Reference compare function from above link check just if from first 20 output ROI 18 of them can be found inside
    // reference with 70% overlap. I consider to extend this verification base on: the output can have less that 20 ROI,
    // if output have 1000 elements to check just first 20 I consider to be not enought; 70% error accepted I supose to
    // be to mutch. So I extend verification in this way: 90% of roy from output should be fund inside reference with an
    // overlap of 90%. This quarantee (I suppose) the the reference is corect, but can be imaginary situation when can
    // fail, even the reference is correct (base of significant number of threashold and computation made in floar for
    // reverence and in fp16 from vpu).
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override {
        // box check
        const auto& expected = expectedOutputs[0].second;
        const auto& actual = actualOutputs[0];

        const auto& expectedBuffer = expected.data();

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->rmap();
        const auto actualBuffer = lockedMemory.as<const std::uint8_t*>();

        const auto& precision = actual->getTensorDesc().getPrecision();
        auto size = actual->size();

        switch (precision) {
        case InferenceEngine::Precision::BF16:
            CompareIou(reinterpret_cast<const ngraph::bfloat16*>(expectedBuffer),
                       reinterpret_cast<const ngraph::bfloat16*>(actualBuffer), size, ngraph::bfloat16(threshold));
            break;
        case InferenceEngine::Precision::FP16:
            CompareIou(reinterpret_cast<const ngraph::float16*>(expectedBuffer),
                       reinterpret_cast<const ngraph::float16*>(actualBuffer), size, ngraph::float16(threshold));
            break;
        case InferenceEngine::Precision::FP32:
            CompareIou<float>(reinterpret_cast<const float*>(expectedBuffer),
                              reinterpret_cast<const float*>(actualBuffer), size, threshold);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
        }

        // score output is generated
        if (expectedOutputs.size() > 1) {
            // check if scores are decrescent value until the end of dynamic size
            const auto& scores = actualOutputs[1];
            auto memoryScore = InferenceEngine::as<InferenceEngine::MemoryBlob>(scores);
            IE_ASSERT(memoryScore);
            const auto lockedMemoryScore = memoryScore->rmap();
            const auto scoresBuffer = lockedMemoryScore.as<const std::uint8_t*>();
            const auto& precisionScore = scores->getTensorDesc().getPrecision();
            switch (precisionScore) {
            case InferenceEngine::Precision::BF16:
                CompareScores(reinterpret_cast<const ngraph::bfloat16*>(scoresBuffer), outputSize);
                break;
            case InferenceEngine::Precision::FP16:
                CompareScores(reinterpret_cast<const ngraph::float16*>(scoresBuffer), outputSize);
                break;
            case InferenceEngine::Precision::FP32:
                CompareScores(reinterpret_cast<const float*>(scoresBuffer), outputSize);
                break;
            default:
                FAIL() << "Comparator for " << precisionScore << " precision isn't supported";
            }
        }
        return;
    }
};

TEST_P(KmbProposalLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= Proposal ============= */
const std::vector<base_size_type> base_size_ = {16};
const std::vector<pre_nms_topn_type> pre_nms_topn_ = {100};
const std::vector<post_nms_topn_type> post_nms_topn_ = {100};
const std::vector<nms_thresh_type> nms_thresh_ = {0.7f};
const std::vector<min_size_type> min_size_ = {1};
const std::vector<ratio_type> ratio_ = {{1.0f, 2.0f}};
const std::vector<scale_type> scale_ = {{1.2f, 1.5f}};
const std::vector<clip_before_nms_type> clip_before_nms_ = {false};
const std::vector<clip_after_nms_type> clip_after_nms_ = {false};

// empty string corresponds to Caffe framework
const std::vector<framework_type> framework_ = {""};

const auto proposalParams = ::testing::Combine(::testing::ValuesIn(base_size_), ::testing::ValuesIn(pre_nms_topn_),
                                               ::testing::ValuesIn(post_nms_topn_), ::testing::ValuesIn(nms_thresh_),
                                               ::testing::ValuesIn(min_size_), ::testing::ValuesIn(ratio_),
                                               ::testing::ValuesIn(scale_), ::testing::ValuesIn(clip_before_nms_),
                                               ::testing::ValuesIn(clip_after_nms_), ::testing::ValuesIn(framework_));

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_tests, KmbProposalLayerTest,
                         ::testing::Combine(proposalParams,
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbProposalLayerTest::getTestCaseName);
// conformance "Proposal_108377"
INSTANTIATE_TEST_SUITE_P(
        smoke_Proposal_tests_108377, KmbProposalLayerTest,
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<base_size_type>{32}),
                                              ::testing::ValuesIn(std::vector<pre_nms_topn_type>{2147483647}),
                                              ::testing::ValuesIn(std::vector<post_nms_topn_type>{100}),
                                              ::testing::ValuesIn(std::vector<nms_thresh_type>{0.69999998807907104f}),
                                              ::testing::ValuesIn(std::vector<min_size_type>{1}),
                                              ::testing::ValuesIn(std::vector<ratio_type>{{0.5f, 1.0f, 2.0f}}),
                                              ::testing::ValuesIn(std::vector<scale_type>{{0.25f, 0.5f, 1.0f, 2.0f}}),
                                              ::testing::ValuesIn(std::vector<clip_before_nms_type>{true}),
                                              ::testing::ValuesIn(std::vector<clip_after_nms_type>{false}),
                                              ::testing::ValuesIn(std::vector<framework_type>{"tensorflow"})),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbProposalLayerTest::getTestCaseName);

// conformance "Proposal_129693"
INSTANTIATE_TEST_SUITE_P(
        smoke_Proposal_tests_129693, KmbProposalLayerTest,
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(std::vector<base_size_type>{16}),
                                              ::testing::ValuesIn(std::vector<pre_nms_topn_type>{6000}),
                                              ::testing::ValuesIn(std::vector<post_nms_topn_type>{300}),
                                              ::testing::ValuesIn(std::vector<nms_thresh_type>{0.69999998807907104f}),
                                              ::testing::ValuesIn(std::vector<min_size_type>{16}),
                                              ::testing::ValuesIn(std::vector<ratio_type>{{0.5f, 1.0f, 2.0f}}),
                                              ::testing::ValuesIn(std::vector<scale_type>{{8.0f, 16.0f, 32.0f}}),
                                              ::testing::ValuesIn(std::vector<clip_before_nms_type>{true}),
                                              ::testing::ValuesIn(std::vector<clip_after_nms_type>{false}),
                                              ::testing::ValuesIn(std::vector<framework_type>{""})),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbProposalLayerTest::getTestCaseName);
}  // namespace
