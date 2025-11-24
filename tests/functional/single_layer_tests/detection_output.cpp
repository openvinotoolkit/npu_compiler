//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <common/print_test_case_name.hpp>
#include <common/tensor_view.hpp>
#include <numeric>
#include <type_traits>
#include <vpu_ov2_layer_test.hpp>

#include "common/random_generator.hpp"
#include "pretty_test_arguments.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/detection_output.hpp"

namespace ov::test {

enum class CodeType { CENTER_SIZE, CORNER_SIZE, CORNER };

std::string codeTypeToString(CodeType codeType) {
    switch (codeType) {
    case CodeType::CENTER_SIZE:
        return "caffe.PriorBoxParameter.CENTER_SIZE";
    case CodeType::CORNER_SIZE:
        return "caffe.PriorBoxParameter.CORNER_SIZE";
    case CodeType::CORNER:
        return "caffe.PriorBoxParameter.CORNER";
    }
    VPUX_THROW("Unsupported CodeType");
}

// We can't get the name of Enum field
static void PrintTo(CodeType param, ::std::ostream* os) {
    const auto fullName = codeTypeToString(param);
    const auto garbageSize = std::string_view(fullName).find_last_of(".") + 1;
    const auto goodSize = fullName.size() - garbageSize;
    const auto importantPart = fullName.substr(garbageSize, goodSize);

    *os << "CodeType: " << ::testing::PrintToString(importantPart);
}

PRETTY_PARAM(VarianceEncodedInTarget, bool);
PRETTY_PARAM(ShareLocation, bool);
PRETTY_PARAM(Normalized, bool);
PRETTY_PARAM(InputHeight, int);
PRETTY_PARAM(InputWidth, int);
PRETTY_PARAM(NumClasses, int);
PRETTY_PARAM(BackgroundLabelId, int);
PRETTY_PARAM(TopK, int);
PRETTY_PARAM(KeepTopK, int);
PRETTY_PARAM(NmsThreshold, float);
PRETTY_PARAM(ConfidenceThreshold, float);
PRETTY_PARAM(ClipAfterNms, bool);
PRETTY_PARAM(ClipBeforeNms, bool);
PRETTY_PARAM(DecreaseLabelId, bool);
PRETTY_PARAM(ObjectnessScore, float);

PRETTY_PARAM(HasAdditionalInputs, bool);
PRETTY_PARAM(NumPriors, int);
PRETTY_PARAM(NumBatches, int);
PRETTY_PARAM(PriorBatchSizeOne, bool);

PRETTY_PARAM(NumDetectedClasses, int);
PRETTY_PARAM(NumDetectionsPerClass, int);

PRETTY_PARAM(Device, std::string);

class DetectionOutputAttributesBuilder {
public:
#define str(x) #x
#define BUILD_PARAM(Type, Field)                                                    \
    DetectionOutputAttributesBuilder& set##Type(Type value) {                       \
        params.Field = value;                                                       \
        VPUX_THROW_WHEN(filled.count(str(Type)) != 0, str(Type) " is already set"); \
        filled.insert(str(Type));                                                   \
        return *this;                                                               \
    }

    BUILD_PARAM(NumClasses, num_classes);
    BUILD_PARAM(BackgroundLabelId, background_label_id);
    BUILD_PARAM(TopK, top_k);
    BUILD_PARAM(VarianceEncodedInTarget, variance_encoded_in_target);
    BUILD_PARAM(ShareLocation, share_location);
    BUILD_PARAM(NmsThreshold, nms_threshold);
    BUILD_PARAM(ConfidenceThreshold, confidence_threshold);
    BUILD_PARAM(ClipAfterNms, clip_after_nms);
    BUILD_PARAM(ClipBeforeNms, clip_before_nms);
    BUILD_PARAM(DecreaseLabelId, decrease_label_id);
    BUILD_PARAM(Normalized, normalized);
    BUILD_PARAM(InputHeight, input_height);
    BUILD_PARAM(InputWidth, input_width);
    BUILD_PARAM(ObjectnessScore, objectness_score);
    DetectionOutputAttributesBuilder& setKeepTopK(KeepTopK value) {
        params.keep_top_k = std::vector<int>{value};
        VPUX_THROW_WHEN(filled.count("KeepTopK") != 0, "KeepTopK is already set");
        filled.insert("KeepTopK");
        return *this;
    }
    DetectionOutputAttributesBuilder& setCodeType(CodeType value) {
        params.code_type = codeTypeToString(value);
        VPUX_THROW_WHEN(filled.count("CodeType") != 0, "CodeType is already set");
        filled.insert("CodeType");
        return *this;
    }
#undef str
#undef BUILD_PARAM

    ov::op::v0::DetectionOutput::Attributes build() {
        VPUX_THROW_UNLESS(filled.size() == 16, "Not all DetectionOutput attributes were set");
        filled.clear();
        return params;
    }

private:
    std::set<std::string> filled;
    ov::op::v0::DetectionOutput::Attributes params;
};

struct NormalizedBox {
    float xmin{};
    float ymin{};
    float xmax{};
    float ymax{};
};

struct DenormalizedBox {
    float padding{};
    float xmin{};
    float ymin{};
    float xmax{};
    float ymax{};
};

template <typename Distribution>
auto generateBox(Distribution& distr, float minSize) {
    auto& generator = RandomGenerator::get();

    float x1 = distr(generator);
    float x2 = distr(generator);
    float y1 = distr(generator);
    float y2 = distr(generator);

    float xmin = std::min(x1, x2);
    float xmax = std::max(x1, x2);
    float ymin = std::min(y1, y2);
    float ymax = std::max(y1, y2);

    return NormalizedBox{xmin, ymin, xmax + minSize, ymax + minSize};
}

ov::Tensor generateNormalizedPriorBoxTensor(NumBatches numBatches, PriorBatchSizeOne priorBatchSizeOne,
                                            VarianceEncodedInTarget varianceEncodedInTarget, NumPriors numPriors) {
    const auto height = varianceEncodedInTarget ? 1 : 2;
    const auto normalizedBoxSize = sizeof(NormalizedBox) / sizeof(float);
    const auto numPriorBatches = priorBatchSizeOne ? 1 : static_cast<int>(numBatches);

    const auto tensor = ov::Tensor{ov::element::f32, makeShape(numPriorBatches, height, numPriors * normalizedBoxSize)};
    const auto normalizedPriors = TensorView<NormalizedBox, 3>(tensor);

    const auto priorMinSize = 0.3f;
    auto priorsDistr = std::uniform_real_distribution<float>(-0.45f, 1.45f - priorMinSize);  // ssd_mobilenet_v1_coco

    for (int b = 0; b < numPriorBatches; b++) {
        for (int p = 0; p < numPriors; p++) {
            normalizedPriors.at(b, 0, p) = generateBox(priorsDistr, priorMinSize);
        }
    }

    if (!varianceEncodedInTarget) {
        const auto varianceMinSize = 0.1f;
        auto varianceDistr = std::uniform_real_distribution<float>(0.7f, 1.2f - varianceMinSize);  // values near 1.0f

        for (int b = 0; b < numPriorBatches; b++) {
            for (int p = 0; p < numPriors; p++) {
                normalizedPriors.at(b, 1, p) = generateBox(varianceDistr, varianceMinSize);
            }
        }
    }

    return tensor;
}

NormalizedBox encodeBoxLogit(NormalizedBox decodedBox, NormalizedBox priorBox, NormalizedBox varianceBox,
                             CodeType codeType) {
    const float priorWidth = priorBox.xmax - priorBox.xmin;
    const float priorHeight = priorBox.ymax - priorBox.ymin;
    const float priorCenterX = (priorBox.xmin + priorBox.xmax) / 2.0f;
    const float priorCenterY = (priorBox.ymin + priorBox.ymax) / 2.0f;

    switch (codeType) {
    case CodeType::CENTER_SIZE: {
        const float decodedBoxWidth = decodedBox.xmax - decodedBox.xmin;
        const float decodedBoxHeight = decodedBox.ymax - decodedBox.ymin;
        const float decodedBoxCenterX = (decodedBox.xmin + decodedBox.xmax) / 2.0f;
        const float decodedBoxCenterY = (decodedBox.ymin + decodedBox.ymax) / 2.0f;

        return NormalizedBox{(decodedBoxCenterX - priorCenterX) / (varianceBox.xmin * priorWidth),   // x min
                             (decodedBoxCenterY - priorCenterY) / (varianceBox.ymin * priorHeight),  // y min
                             std::log(decodedBoxWidth / priorWidth) / varianceBox.xmax,              // x max
                             std::log(decodedBoxHeight / priorHeight) / varianceBox.ymax};           // y max
    }
    case CodeType::CORNER: {
        return NormalizedBox{(decodedBox.xmin - priorBox.xmin) / varianceBox.xmin,   // x min
                             (decodedBox.ymin - priorBox.ymin) / varianceBox.ymin,   // y min
                             (decodedBox.xmax - priorBox.xmax) / varianceBox.xmax,   // x max
                             (decodedBox.ymax - priorBox.ymax) / varianceBox.ymax};  // y max
    }
    case CodeType::CORNER_SIZE: {
        return NormalizedBox{(decodedBox.xmin - priorBox.xmin) / (varianceBox.xmin * priorWidth),    // x min
                             (decodedBox.ymin - priorBox.ymin) / (varianceBox.ymin * priorHeight),   // y min
                             (decodedBox.xmax - priorBox.xmax) / (varianceBox.xmax * priorWidth),    // x max
                             (decodedBox.ymax - priorBox.ymax) / (varianceBox.ymax * priorHeight)};  // y max
    }
    }
    VPUX_THROW("CodeType '{0}' is not supported", codeTypeToString(codeType));
}

ov::Tensor encodeBoxLogits(TensorView<NormalizedBox, 3> priorBoxes, NumPriors numPriors, NumClasses numClasses,
                           NumBatches numBatches, ShareLocation shareLocation, CodeType codeType) {
    const auto numLocClasses = shareLocation ? 1 : static_cast<int>(numClasses);
    const auto varianceEncodedInTarget = (priorBoxes.getShape()[1] == 1);
    const auto priorBatchSizeOne = (priorBoxes.getShape()[0] == 1);

    const auto boxSize = sizeof(NormalizedBox) / sizeof(float);
    const auto tensor = ov::Tensor{ov::element::f32, makeShape(numBatches, numPriors * numLocClasses * boxSize)};
    const auto boxLogits = TensorView<NormalizedBox, 3>(tensor, makeShape(numBatches, numPriors, numLocClasses));

    const auto decodedMinSize = 0.05f;
    auto decodedDistr = std::uniform_real_distribution<float>(0.0f, 1.0f - decodedMinSize);

    for (int b = 0; b < numBatches; b++) {
        const auto priorBatch = b * !priorBatchSizeOne;
        for (int p = 0; p < numPriors; p++) {
            const auto priorBox = priorBoxes.at(priorBatch, 0, p);
            const auto varianceBox =
                    varianceEncodedInTarget ? NormalizedBox{1.0f, 1.0f, 1.0f, 1.0f} : priorBoxes.at(priorBatch, 1, p);

            for (int c = 0; c < numLocClasses; c++) {
                auto decoded = generateBox(decodedDistr, decodedMinSize);
                boxLogits.at(b, p, c) = encodeBoxLogit(decoded, priorBox, varianceBox, codeType);
            }
        }
    }

    return tensor;
}

std::vector<int> generateDetectedClassesIndices(int numClasses, int maxDetectedClasses) {
    const auto numDetectedClasses = std::min<int>(numClasses, maxDetectedClasses);

    auto indices = std::vector<int>(numClasses);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), RandomGenerator::get());

    indices.resize(numDetectedClasses);
    std::sort(indices.begin(), indices.end());

    return indices;
}

ov::Tensor generateClassPredictions(NumBatches numBatches, NumPriors numPriors, NumClasses numClasses,
                                    int maxDetectedClasses, int maxDetectionsPerClass,
                                    ConfidenceThreshold confidenceThreshold) {
    const auto tensor = ov::Tensor{ov::element::f32, makeShape(numBatches, numPriors * numClasses)};
    const auto classPredictions = TensorView<float, 3>(tensor, makeShape(numBatches, numPriors, numClasses));

    std::fill_n(classPredictions.data(), classPredictions.size(), 0.0f);

    const auto detectedClassIndices = generateDetectedClassesIndices(numClasses, maxDetectedClasses);
    const auto numDetectionsPerClass = std::min<int>(maxDetectionsPerClass, numPriors);

    const auto genConfidenceUniform = [&](std::vector<float>& confidence) {
        const auto threshold = confidenceThreshold + 0.001;  // to not have a box with confidence == confidenceThreshold
        const auto step = (1.0f - threshold) / numDetectionsPerClass;

        for (int i = 0; i < numDetectionsPerClass; i++) {
            confidence[i] = threshold + step * i;
        }
        for (int i = numDetectionsPerClass; i < numPriors; i++) {
            confidence[i] = 0.0f;
        }

        std::shuffle(confidence.begin(), confidence.end(), RandomGenerator::get());
    };

    auto confidence = std::vector<float>(numPriors);
    const auto numDetections = static_cast<int>(detectedClassIndices.size());
    for (int b = 0; b < numBatches; b++) {
        for (int i = 0; i < numDetections; i++) {
            const auto classIndex = detectedClassIndices[i];

            genConfidenceUniform(confidence);

            VPUX_THROW_UNLESS(confidence.size() == numPriors, "Confidence values have unexpected size");
            for (int p = 0; p < numPriors; p++) {
                classPredictions.at(b, p, classIndex) = confidence[p];
            }
        }
    }

    return tensor;
}

ov::Tensor denormalizePriorBoxTensor(TensorView<NormalizedBox, 3> normalizedPriorBox, InputWidth inputWidth,
                                     InputHeight inputHeight) {
    const auto priorsShape = normalizedPriorBox.getShape();
    VPUX_THROW_UNLESS(priorsShape[1] == 1,
                      "Normalized == false and VarianceEncodedInTarget == false is not supported.");

    const auto boxSize = sizeof(DenormalizedBox) / sizeof(float);
    const auto tensor =
            ov::Tensor{ov::element::f32, makeShape(priorsShape[0], priorsShape[1], priorsShape[2] * boxSize)};
    const auto denormalizedPriorBox = TensorView<DenormalizedBox, 3>(tensor, priorsShape);

    const auto denormalizePriorBox = [&](const NormalizedBox& normalized) {
        DenormalizedBox denormalized;

        denormalized.padding = 0.0f;
        denormalized.xmin = normalized.xmin * inputWidth;
        denormalized.ymin = normalized.ymin * inputHeight;
        denormalized.xmax = normalized.xmax * inputWidth;
        denormalized.ymax = normalized.ymax * inputHeight;

        return denormalized;
    };
    std::transform(normalizedPriorBox.data(), normalizedPriorBox.data() + normalizedPriorBox.size(),
                   denormalizedPriorBox.data(), denormalizePriorBox);

    return tensor;
}

float jaccardOverlap(const NormalizedBox& box1, const NormalizedBox& box2) {
    float intersectXmin = std::max(box1.xmin, box2.xmin);
    float intersectYmin = std::max(box1.ymin, box2.ymin);
    float intersectXmax = std::min(box1.xmax, box2.xmax);
    float intersectYmax = std::min(box1.ymax, box2.ymax);

    float intersectWidth = std::max(intersectXmax - intersectXmin, 0.0f);
    float intersectHeight = std::max(intersectYmax - intersectYmin, 0.0f);
    float intersectSize = intersectWidth * intersectHeight;

    if (intersectSize <= 0) {
        return 0;
    }

    float bbox1Size = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
    float bbox2Size = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);

    return intersectSize / (bbox1Size + bbox2Size - intersectSize);
}

// remove boxes that has intersection area near nmsThreshold
void removeUnwantedClassPredictions(TensorView<float, 3> classPredictions, TensorView<NormalizedBox, 3> decodedBoxes,
                                    NumClasses numClasses, ShareLocation shareLocation, DecreaseLabelId decreaseLabelId,
                                    NmsThreshold nmsThreshold, ConfidenceThreshold confidenceThreshold) {
    VPUX_THROW_UNLESS(decreaseLabelId == false, "DecreaseLabelId == true (MxNet NMS) is not supported");

    const auto rangeNearNmsThreshold = 0.1f;

    const auto shape = classPredictions.getShape();
    const auto numBatches = shape[0];
    const auto numPriors = shape[1];

    struct BoxConfidence {
        float confidence;
        int priorId;
        NormalizedBox box;
    };

    auto detections = std::vector<BoxConfidence>();
    auto goodBoxes = std::vector<NormalizedBox>();
    detections.reserve(numPriors);
    goodBoxes.reserve(numPriors);

    for (int b = 0; b < numBatches; b++) {
        for (int c = 0; c < numClasses; c++) {
            detections.clear();
            for (int p = 0; p < numPriors; p++) {
                const auto box = decodedBoxes.at(b, c * !shareLocation, p);
                const auto conf = classPredictions.at(b, p, c);
                detections.push_back(BoxConfidence{conf, p, box});
            }

            const auto greaterConf = [](const BoxConfidence& lhs, const BoxConfidence& rhs) {
                return lhs.confidence > rhs.confidence;
            };
            std::sort(detections.begin(), detections.end(), greaterConf);

            const auto nonConfidentBox = [&](const BoxConfidence& boxConf) {
                return boxConf.confidence < confidenceThreshold;
            };
            const auto lastBoxIt = std::find_if(detections.begin(), detections.end(), nonConfidentBox);
            const auto confidentBoxesCount = std::distance(detections.begin(), lastBoxIt);

            detections.resize(confidentBoxesCount);

            goodBoxes.clear();
            for (const auto& detection : detections) {
                const auto suspectBox = detection.box;

                const auto intersectsNearNmsThreshold = [&](const NormalizedBox& goodBox) {
                    const auto overlap = jaccardOverlap(suspectBox, goodBox);
                    return (std::fabs(overlap - nmsThreshold) < rangeNearNmsThreshold);
                };
                const auto isBorderlineBox =
                        std::any_of(goodBoxes.begin(), goodBoxes.end(), intersectsNearNmsThreshold);

                if (isBorderlineBox) {
                    // remove suspectBox from consideration by zeroing its confidence
                    const auto prior = detection.priorId;
                    classPredictions.at(b, prior, c) = 0.0f;
                } else {
                    goodBoxes.push_back(suspectBox);
                }
            }
        }
    }
}

NormalizedBox decodeBox(NormalizedBox bbox, NormalizedBox priorBox, NormalizedBox varianceBox, CodeType codeType) {
    switch (codeType) {
    case CodeType::CENTER_SIZE: {
        float priorWidth = priorBox.xmax - priorBox.xmin;
        float priorHeight = priorBox.ymax - priorBox.ymin;
        float priorCenterX = (priorBox.xmin + priorBox.xmax) / 2.f;
        float priorCenterY = (priorBox.ymin + priorBox.ymax) / 2.f;

        float decodeBboxCenterX = varianceBox.xmin * bbox.xmin * priorWidth + priorCenterX;
        float decodeBboxCenterY = varianceBox.ymin * bbox.ymin * priorHeight + priorCenterY;
        float decodeBboxWidth = std::exp(varianceBox.xmax * bbox.xmax) * priorWidth;
        float decodeBboxHeight = std::exp(varianceBox.ymax * bbox.ymax) * priorHeight;

        return NormalizedBox{
                decodeBboxCenterX - decodeBboxWidth / 2.f,   // xmin
                decodeBboxCenterY - decodeBboxHeight / 2.f,  // ymin
                decodeBboxCenterX + decodeBboxWidth / 2.f,   // xmax
                decodeBboxCenterY + decodeBboxHeight / 2.f   // ymax
        };
    }
    case CodeType::CORNER: {
        return NormalizedBox{
                priorBox.xmin + varianceBox.xmin * bbox.xmin,  // xmin
                priorBox.ymin + varianceBox.ymin * bbox.ymin,  // ymin
                priorBox.xmax + varianceBox.xmax * bbox.xmax,  // xmax
                priorBox.ymax + varianceBox.ymax * bbox.ymax   // ymax
        };
    }
    case CodeType::CORNER_SIZE: {
        float priorWidth = priorBox.xmax - priorBox.xmin;
        float priorHeight = priorBox.ymax - priorBox.ymin;

        return NormalizedBox{
                priorBox.xmin + varianceBox.xmin * bbox.xmin * priorWidth,   // xmin
                priorBox.ymin + varianceBox.ymin * bbox.ymin * priorHeight,  // ymin
                priorBox.xmax + varianceBox.xmax * bbox.xmax * priorWidth,   // xmax
                priorBox.ymax + varianceBox.ymax * bbox.ymax * priorHeight   // ymax
        };
    }
    }
    VPUX_THROW("Unsupported CodeType");
}

ov::Tensor decodeBoxes(TensorView<NormalizedBox, 3> boxLogits, TensorView<NormalizedBox, 3> normalizedPriorBox,
                       CodeType codeType) {
    const auto logitsShape = boxLogits.getShape();
    const auto numBatches = logitsShape[0];
    const auto numPriors = logitsShape[1];
    const auto numLocClasses = logitsShape[2];

    const auto boxSize = sizeof(NormalizedBox) / sizeof(float);
    const auto tensor = ov::Tensor(ov::element::f32, makeShape(numBatches, numLocClasses, numPriors * boxSize));
    const auto decodedBoxes = TensorView<NormalizedBox, 3>(tensor);

    const auto varianceEncodedInTarget = (normalizedPriorBox.getShape()[1] == 1);

    for (int b = 0; b < numBatches; b++) {
        for (int p = 0; p < numPriors; p++) {
            const auto priorBox = normalizedPriorBox.at(b, 0, p);
            const auto varianceBox =
                    varianceEncodedInTarget ? NormalizedBox{1.0f, 1.0f, 1.0f, 1.0f} : normalizedPriorBox.at(b, 1, p);
            for (int c = 0; c < numLocClasses; c++) {
                const auto boxLogit = boxLogits.at(b, p, c);

                // store in more convenient "layout" (batch, classId, priorId)
                decodedBoxes.at(b, c, p) = decodeBox(boxLogit, priorBox, varianceBox, codeType);
            }
        }
    }

    return tensor;
}

template <typename Threshold>
struct TolerantFloat {
    operator float() const {
        return value;
    }

    bool operator<(const TolerantFloat& rhs) const {
        const auto threshold = static_cast<float>(Threshold::num) / Threshold::den;
        return (std::fabs(value - rhs.value) > threshold) && (value < rhs.value);
    }

    bool operator==(const TolerantFloat& rhs) const {
        return !(value < rhs.value) && !(rhs.value < value);
    }

    float value = 0.0f;
};

struct Detection {
    using ConfidenceTolerance = std::ratio<1, 1000>;
    using CoordinateTolerance = std::ratio<5, 100>;

    float batch;
    float classId;
    TolerantFloat<ConfidenceTolerance> confidence;
    TolerantFloat<CoordinateTolerance> x0;
    TolerantFloat<CoordinateTolerance> y0;
    TolerantFloat<CoordinateTolerance> x1;
    TolerantFloat<CoordinateTolerance> y1;
};

bool operator<(const Detection& lhs, const Detection& rhs) {
    return std::tie(lhs.batch, lhs.classId, lhs.confidence, lhs.x0, lhs.y0, lhs.x1, lhs.y1) <
           std::tie(rhs.batch, rhs.classId, rhs.confidence, rhs.x0, rhs.y0, rhs.x1, rhs.y1);
}

bool operator==(const Detection& lhs, const Detection& rhs) {
    return !(lhs < rhs) && !(rhs < lhs);
}

std::ostream& operator<<(std::ostream& os, const Detection& detection) {
    os << "[" << static_cast<int>(detection.batch) << " : " << static_cast<int>(detection.classId) << " : "
       << std::fixed << std::setprecision(4) << static_cast<float>(detection.confidence) << " | "
       << static_cast<float>(detection.x0) << " " << static_cast<float>(detection.y0) << " "
       << static_cast<float>(detection.x1) << " " << static_cast<float>(detection.y1) << "]";
    return os;
}

std::vector<Detection> detectionDifference(const std::vector<Detection>& lhs, const std::vector<Detection>& rhs) {
    auto difference = std::vector<Detection>();
    difference.reserve(std::max(lhs.size(), rhs.size()));
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(difference));
    difference.shrink_to_fit();
    return difference;
}

void printDetections(const std::vector<Detection>& detections, int printErrors = 50) {
    for (const auto& detection : detections) {
        std::cout << detection << "\n";
        if (!--printErrors) {
            std::cout << "[...]\n";
            break;
        }
    }
}

using NormalizationParams = std::tuple<Normalized, InputHeight, InputWidth, VarianceEncodedInTarget>;
using TensorShapeParams = std::tuple<NumPriors, NumClasses, NumBatches, PriorBatchSizeOne, BackgroundLabelId>;
using DetectionOutputAttributes = std::tuple<ShareLocation, TopK, KeepTopK, CodeType, NmsThreshold, ConfidenceThreshold,
                                             ClipAfterNms, ClipBeforeNms, DecreaseLabelId>;
using AdditionalInputsParams = std::tuple<HasAdditionalInputs, ObjectnessScore>;
using MetaParams = std::tuple<NumDetectedClasses, NumDetectionsPerClass>;

using DetectionOutputParams = std::tuple<NormalizationParams, TensorShapeParams, DetectionOutputAttributes,
                                         AdditionalInputsParams, MetaParams, Device>;

class DetectionOutputLayerTestCommon :
        public testing::WithParamInterface<DetectionOutputParams>,
        public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 3, "Only 3 inputs are supported");

        const auto detectionOutputAttrs = std::get<DetectionOutputAttributes>(GetParam());
        const auto codeType = std::get<CodeType>(detectionOutputAttrs);
        const auto nmsThreshold = std::get<NmsThreshold>(detectionOutputAttrs);

        const auto normalizationParams = std::get<NormalizationParams>(GetParam());
        const auto normalized = std::get<Normalized>(normalizationParams);
        const auto inputHeight = std::get<InputHeight>(normalizationParams);
        const auto inputWidth = std::get<InputWidth>(normalizationParams);
        const auto varianceEncodedInTarget = std::get<VarianceEncodedInTarget>(normalizationParams);

        const auto tensorSizeParams = std::get<TensorShapeParams>(GetParam());
        const auto numPriors = std::get<NumPriors>(tensorSizeParams);
        const auto numClasses = std::get<NumClasses>(tensorSizeParams);
        const auto numBatches = std::get<NumBatches>(tensorSizeParams);
        const auto priorBatchSizeOne = std::get<PriorBatchSizeOne>(tensorSizeParams);

        const auto normalizedPriorBoxTensor =
                generateNormalizedPriorBoxTensor(numBatches, priorBatchSizeOne, varianceEncodedInTarget, numPriors);

        const auto detectionOutputAttributes = std::get<DetectionOutputAttributes>(GetParam());
        const auto shareLocation = std::get<ShareLocation>(detectionOutputAttributes);
        const auto decreaseLabelId = std::get<DecreaseLabelId>(detectionOutputAttributes);
        const auto confidenceThreshold = std::get<ConfidenceThreshold>(detectionOutputAttributes);

        const auto numLocClasses = shareLocation ? 1 : static_cast<int>(numClasses);
        const auto normalizedPriorBoxes = TensorView<NormalizedBox, 3>(normalizedPriorBoxTensor);
        const auto boxLogitsTensor =
                encodeBoxLogits(normalizedPriorBoxes, numPriors, numClasses, numBatches, shareLocation, codeType);

        const auto metaParams = std::get<MetaParams>(GetParam());
        const auto numDetectedClasses = std::get<NumDetectedClasses>(metaParams);
        const auto numDetectionsPerClass = std::get<NumDetectionsPerClass>(metaParams);
        auto classPredictionsTensor = generateClassPredictions(numBatches, numPriors, numClasses, numDetectedClasses,
                                                               numDetectionsPerClass, confidenceThreshold);

        const auto boxLogits =
                TensorView<NormalizedBox, 3>(boxLogitsTensor, makeShape(numBatches, numPriors, numLocClasses));
        const auto decodedBoxesTensor = decodeBoxes(boxLogits, normalizedPriorBoxes, codeType);

        auto classPredictions =
                TensorView<float, 3>(classPredictionsTensor, makeShape(numBatches, numPriors, numClasses));

        const auto decodedBoxes = TensorView<NormalizedBox, 3>(decodedBoxesTensor);
        removeUnwantedClassPredictions(classPredictions, decodedBoxes, numClasses, shareLocation, decreaseLabelId,
                                       nmsThreshold, confidenceThreshold);

        const auto& priorBoxTensor = [&] {
            if (normalized) {
                return normalizedPriorBoxTensor;
            }
            return denormalizePriorBoxTensor(normalizedPriorBoxes, inputWidth, inputHeight);
        }();

        VPUX_THROW_UNLESS(boxLogitsTensor.get_shape() == inputShapes[0], "BoxLogits has shape {0}, but expected {1}",
                          boxLogitsTensor.get_shape().to_string(), inputShapes[0].to_string());
        VPUX_THROW_UNLESS(classPredictionsTensor.get_shape() == inputShapes[1],
                          "ClassPredictions has shape {0}, but expected {1}",
                          classPredictionsTensor.get_shape().to_string(), inputShapes[1].to_string());
        VPUX_THROW_UNLESS(priorBoxTensor.get_shape() == inputShapes[2], "PriorBox has shape {0}, but expected {1}",
                          priorBoxTensor.get_shape().to_string(), inputShapes[2].to_string());

        inputs = {
                {funcInputs[0].get_node_shared_ptr(), boxLogitsTensor},
                {funcInputs[1].get_node_shared_ptr(), classPredictionsTensor},
                {funcInputs[2].get_node_shared_ptr(), priorBoxTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        auto expected = expectedTensors[0];
        auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const auto countDetections = [](const ov::Tensor& tensor) {
            const auto detections = reinterpret_cast<const Detection*>(tensor.data());
            VPUX_THROW_UNLESS(tensor.get_shape().back() == 7, "DetectionOutput result tensor must have width == 7");
            const auto maxDetections = tensor.get_size() / 7;
            for (size_t i = 0; i < maxDetections; i++) {
                if (detections[i].batch == -1.0f) {
                    return i;
                }
            }
            return maxDetections;
        };

        const auto actSize = countDetections(actual);
        const auto expSize = countDetections(expected);
        const auto detectionSize = 7;

        VPUX_THROW_UNLESS(actSize <= (actual.get_size() / detectionSize), "Too many boxes detected");
        VPUX_THROW_UNLESS(expSize <= (expected.get_size() / detectionSize), "Too many boxes in reference detected");

        auto output = std::vector<Detection>(reinterpret_cast<const Detection*>(actual.data()),
                                             reinterpret_cast<const Detection*>(actual.data()) + actSize);
        auto reference = std::vector<Detection>(reinterpret_cast<const Detection*>(expected.data()),
                                                reinterpret_cast<const Detection*>(expected.data()) + expSize);

        std::sort(output.begin(), output.end());
        std::sort(reference.begin(), reference.end());

        const auto diff0 = detectionDifference(output, reference);
        if (!diff0.empty()) {
            std::cout << "Detections in output but not in reference:\n";
            printDetections(diff0);
        }

        const auto diff1 = detectionDifference(reference, output);
        if (!diff1.empty()) {
            std::cout << "Detections in reference but not in output:\n";
            printDetections(diff1);
        }

        ASSERT_EQ(output.size(), reference.size());
        ASSERT_EQ(diff0.size(), 0);
        ASSERT_EQ(diff1.size(), 0);
    }

    void SetUp() override {
        const auto& [normalizationParams, tensorSizeParams, detectionOutputAttributes, additionalInputsParams,
                     metaParams, device] = GetParam();

        targetDevice = device;
        inType = ov::element::Type_t::f32;
        outType = ov::element::Type_t::f32;

        const auto& [normalized, inputHeight, inputWidth, varianceEncodedInTarget] = normalizationParams;
        const auto& [numPriors, numClasses, numBatches, priorBatchSizeOne, backgroundLabelId] = tensorSizeParams;
        const auto& [shareLocation, topK, keepTopK, codeType, nmsThreshold, confidenceThreshold, clipAfterNms,
                     clipBeforeNms, decreaseLabelId] = detectionOutputAttributes;
        const auto& [hasAdditionalInputs, objectnessScore] = additionalInputsParams;

        attrs = DetectionOutputAttributesBuilder()
                        .setBackgroundLabelId(backgroundLabelId)
                        .setClipAfterNms(clipAfterNms)
                        .setClipBeforeNms(clipBeforeNms)
                        .setCodeType(codeType)
                        .setConfidenceThreshold(confidenceThreshold)
                        .setDecreaseLabelId(decreaseLabelId)
                        .setInputHeight(inputHeight)
                        .setInputWidth(inputWidth)
                        .setKeepTopK(keepTopK)
                        .setNmsThreshold(nmsThreshold)
                        .setNormalized(normalized)
                        .setNumClasses(numClasses)
                        .setObjectnessScore(objectnessScore)
                        .setShareLocation(shareLocation)
                        .setTopK(topK)
                        .setVarianceEncodedInTarget(varianceEncodedInTarget)
                        .build();

        VPUX_THROW_UNLESS(hasAdditionalInputs == false, "DetectionOutput test does not support additional inputs");

        const auto numLocClasses = shareLocation ? 1 : static_cast<int>(numClasses);

        auto boxLogitsShape = generateTestShape(numBatches, numPriors * numLocClasses * 4);
        auto classConfidenceShape = generateTestShape(numBatches, numPriors * numClasses);

        const auto priorBatch = priorBatchSizeOne ? 1 : static_cast<int>(numBatches);
        const auto priorHeight = varianceEncodedInTarget ? 1 : 2;
        const auto priorBoxSize = normalized ? 4 : 5;

        auto priorBoxesShape = generateTestShape(priorBatch, priorHeight, numPriors * priorBoxSize);

        init_input_shapes({boxLogitsShape, classConfidenceShape, priorBoxesShape});

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        ASSERT_EQ(3, params.size());
        auto detOut = std::make_shared<ov::op::v0::DetectionOutput>(params[0], params[1], params[2], attrs);
        auto results = ov::ResultVector{std::make_shared<ov::op::v0::Result>(detOut)};
        function = std::make_shared<ov::Model>(results, params, "DetectionOutput");
    }

private:
    ov::op::v0::DetectionOutput::Attributes attrs;
};

//
// Test parameters
//

const auto normalizationParams = std::vector<NormalizationParams>{
        {Normalized(true), InputHeight(0), InputWidth(0), VarianceEncodedInTarget(true)},
        {Normalized(true), InputHeight(0), InputWidth(0), VarianceEncodedInTarget(false)},
        {Normalized(false), InputHeight(420), InputWidth(1000), VarianceEncodedInTarget(true)},
};

// Example on how to extract NumPriors and NumClasses from input tensors shape:
// Inputs: BoxLogits, ClassPredictions, PriorBoxes
// PriorBoxSize = Normalized ? 4 : 5
// NumPriors = PriorBoxesShape[2] / PriorBoxSize
// NumClasses = ClassPredictions[1] / NumPriors
//
// Example face-detection-adas-0001: {1, 40448}, {1, 20224}, {1, 2, 40448}
// PriorBoxSize = 4
// NumPriors = {1, 2, 40448}[2] / 4 = 10112
// NumClasses = {1, 20224}[1] / 10112 = 2
//
const auto tensorShapeParams = std::vector<TensorShapeParams>{
        // face-detection-adas-0001
        {NumPriors(10112), NumClasses(2), NumBatches(1), PriorBatchSizeOne(true), BackgroundLabelId(0)},
};

const auto detectionOutputAttributes = ::testing::Combine(             //
        ::testing::Values(ShareLocation(true), ShareLocation(false)),  //
        ::testing::Values(TopK(200)),                                  //
        ::testing::Values(KeepTopK(400)),                              //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),            //
        ::testing::Values(NmsThreshold(0.45f)),                        //
        ::testing::Values(ConfidenceThreshold(0.001f)),                //
        ::testing::Values(ClipAfterNms(true), ClipAfterNms(false)),    //
        ::testing::Values(ClipBeforeNms(true), ClipBeforeNms(false)),  //
        ::testing::Values(DecreaseLabelId(false))                      //
);

const auto additionalInputsParams = std::vector<AdditionalInputsParams>{
        {HasAdditionalInputs(false), ObjectnessScore(0.0f)},
};

const auto metaParams = std::vector<MetaParams>{{NumDetectedClasses(5), NumDetectionsPerClass(20)}};

//
// Platform test definition
//

TEST_P(DetectionOutputLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DetectionOutputLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

//
// 3 Inputs tests, all permutations
//

const auto detectionOutputParams =
        ::testing::Combine(::testing::ValuesIn(normalizationParams), ::testing::ValuesIn(tensorShapeParams),
                           detectionOutputAttributes, ::testing::ValuesIn(additionalInputsParams),
                           ::testing::ValuesIn(metaParams), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput, DetectionOutputLayerTestCommon, detectionOutputParams,
                         PrintTestCaseName());

//
// ssdlite_mobilenet_v2
//

const auto normalizedWithNonEncodedVariance =
        NormalizationParams{Normalized(true), InputHeight(0), InputWidth(0), VarianceEncodedInTarget(false)};

const auto ssdliteMobilenetV2ShapeParams = TensorShapeParams{NumPriors(1917), NumClasses(91), NumBatches(1),
                                                             PriorBatchSizeOne(true), BackgroundLabelId(0)};

const auto ssdliteMobilenetV2Attributes = ::testing::Combine(  //
        ::testing::Values(ShareLocation(true)),                //
        ::testing::Values(TopK(100)),                          //
        ::testing::Values(KeepTopK(100)),                      //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),    //
        ::testing::Values(NmsThreshold(0.6f)),                 //
        ::testing::Values(ConfidenceThreshold(0.3f)),          //
        ::testing::Values(ClipAfterNms(true)),                 //
        ::testing::Values(ClipBeforeNms(false)),               //
        ::testing::Values(DecreaseLabelId(false))              //
);

const auto ssdliteMobilenetV2Params = ::testing::Combine(
        ::testing::Values(normalizedWithNonEncodedVariance), ::testing::Values(ssdliteMobilenetV2ShapeParams),
        ssdliteMobilenetV2Attributes, ::testing::ValuesIn(additionalInputsParams), ::testing::ValuesIn(metaParams),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_ssdlite_mobilenet_v2, DetectionOutputLayerTestCommon,
                         ssdliteMobilenetV2Params, PrintTestCaseName());

//
// person-detection-0100
//

const auto personDetection0100ShapeParams =
        TensorShapeParams{NumPriors(1344), NumClasses(2), NumBatches(1), PriorBatchSizeOne(true), BackgroundLabelId(0)};

// Increased number of confident classes generated by the test
const auto metaParamsPersonDetection0100 = std::vector<MetaParams>{{NumDetectedClasses(2), NumDetectionsPerClass(300)}};

const auto personDetection0100Attributes = ::testing::Combine(  //
        ::testing::Values(ShareLocation(true)),                 //
        ::testing::Values(TopK(200)),                           //
        ::testing::Values(KeepTopK(200)),                       //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),     //
        ::testing::Values(NmsThreshold(0.45f)),                 //
        ::testing::Values(ConfidenceThreshold(0.02f)),          //
        ::testing::Values(ClipAfterNms(false)),                 //
        ::testing::Values(ClipBeforeNms(false)),                //
        ::testing::Values(DecreaseLabelId(false))               //
);

const auto personDetection0100Params = ::testing::Combine(
        ::testing::Values(normalizedWithNonEncodedVariance), ::testing::Values(personDetection0100ShapeParams),
        personDetection0100Attributes, ::testing::ValuesIn(additionalInputsParams),
        ::testing::ValuesIn(metaParamsPersonDetection0100), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_person_detection_0100, DetectionOutputLayerTestCommon,
                         personDetection0100Params, PrintTestCaseName());

//
// efficientdet-d0
//

const auto efficientdetD0ShapeParams = TensorShapeParams{NumPriors(49104), NumClasses(90), NumBatches(1),
                                                         PriorBatchSizeOne(true), BackgroundLabelId(91)};

const auto efficientdetD0Attributes = ::testing::Combine(    //
        ::testing::Values(ShareLocation(true)),              //
        ::testing::Values(TopK(100)),                        //
        ::testing::Values(KeepTopK(100)),                    //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),  //
        ::testing::Values(NmsThreshold(0.6f)),               //
        ::testing::Values(ConfidenceThreshold(0.2f)),        //
        ::testing::Values(ClipAfterNms(false)),              //
        ::testing::Values(ClipBeforeNms(false)),             //
        ::testing::Values(DecreaseLabelId(false))            //
);

const auto efficientdetD0Params = ::testing::Combine(
        ::testing::Values(normalizedWithNonEncodedVariance), ::testing::Values(efficientdetD0ShapeParams),
        efficientdetD0Attributes, ::testing::ValuesIn(additionalInputsParams), ::testing::ValuesIn(metaParams),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_efficientdet_d0, DetectionOutputLayerTestCommon,
                         efficientdetD0Params, PrintTestCaseName());

//
// efficientdet-d1
//

const auto efficientdetD1ShapeParams = TensorShapeParams{NumPriors(76725), NumClasses(90), NumBatches(1),
                                                         PriorBatchSizeOne(true), BackgroundLabelId(91)};

const auto efficientdetD1Params = ::testing::Combine(
        ::testing::Values(normalizedWithNonEncodedVariance), ::testing::Values(efficientdetD1ShapeParams),
        efficientdetD0Attributes, ::testing::ValuesIn(additionalInputsParams), ::testing::ValuesIn(metaParams),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_efficientdet_d1, DetectionOutputLayerTestCommon,
                         efficientdetD1Params, PrintTestCaseName());

//
// faster_rcnn_inception_resnet_v2_atrous_oid_v4
//

const auto normalizedWithEncodedVariance =
        NormalizationParams{Normalized(true), InputHeight(0), InputWidth(0), VarianceEncodedInTarget(true)};

const auto fasterRcnnShapeParams = TensorShapeParams{NumPriors(100), NumClasses(602), NumBatches(1),
                                                     PriorBatchSizeOne(true), BackgroundLabelId(0)};

const auto fasterRcnnAttributes = ::testing::Combine(        //
        ::testing::Values(ShareLocation(false)),             //
        ::testing::Values(TopK(100)),                        //
        ::testing::Values(KeepTopK(100)),                    //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),  //
        ::testing::Values(NmsThreshold(0.6f)),               //
        ::testing::Values(ConfidenceThreshold(0.3f)),        //
        ::testing::Values(ClipAfterNms(false)),              //
        ::testing::Values(ClipBeforeNms(true)),              //
        ::testing::Values(DecreaseLabelId(false))            //
);

const auto fasterRcnnParams =
        ::testing::Combine(::testing::Values(normalizedWithEncodedVariance), ::testing::Values(fasterRcnnShapeParams),
                           fasterRcnnAttributes, ::testing::ValuesIn(additionalInputsParams),
                           ::testing::ValuesIn(metaParams), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_faster_rcnn, DetectionOutputLayerTestCommon, fasterRcnnParams,
                         PrintTestCaseName());

//
// faster_rcnn_resnet
//

const auto fasterRcnnResnetNormalization =
        NormalizationParams{Normalized(false), InputHeight(600), InputWidth(1000), VarianceEncodedInTarget(true)};

const auto fasterRcnnResnetShapeParams =
        TensorShapeParams{NumPriors(300), NumClasses(21), NumBatches(1), PriorBatchSizeOne(true), BackgroundLabelId(0)};

const auto fasterRcnnResnetAttributes = ::testing::Combine(  //
        ::testing::Values(ShareLocation(false)),             //
        ::testing::Values(TopK(400)),                        //
        ::testing::Values(KeepTopK(200)),                    //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),  //
        ::testing::Values(NmsThreshold(0.3f)),               //
        ::testing::Values(ConfidenceThreshold(0.5f)),        // actual confidence threshold is 0 for the network
        ::testing::Values(ClipAfterNms(false)),              //
        ::testing::Values(ClipBeforeNms(false)),             //
        ::testing::Values(DecreaseLabelId(false))            //
);

const auto fasterRcnnResnetParams = ::testing::Combine(
        ::testing::Values(fasterRcnnResnetNormalization), ::testing::Values(fasterRcnnResnetShapeParams),
        fasterRcnnResnetAttributes, ::testing::ValuesIn(additionalInputsParams), ::testing::ValuesIn(metaParams),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_smoke_DetectionOutput_faster_rcnn_resnet, DetectionOutputLayerTestCommon,
                         fasterRcnnResnetParams, PrintTestCaseName());

// ssd-resnet101-fpn-oid

const auto ssdResnet101ShapeParams = TensorShapeParams{NumPriors(131040), NumClasses(602), NumBatches(1),
                                                       PriorBatchSizeOne(true), BackgroundLabelId(0)};

const auto ssdResnet101Attributes = ::testing::Combine(      //
        ::testing::Values(ShareLocation(true)),              //
        ::testing::Values(TopK(100)),                        //
        ::testing::Values(KeepTopK(100)),                    //
        ::testing::Values(CodeType(CodeType::CENTER_SIZE)),  //
        ::testing::Values(NmsThreshold(0.6f)),               //
        ::testing::Values(ConfidenceThreshold(0.3f)),        //
        ::testing::Values(ClipAfterNms(true)),               //
        ::testing::Values(ClipBeforeNms(false)),             //
        ::testing::Values(DecreaseLabelId(false))            //
);

const auto ssdResnet101Params = ::testing::Combine(
        ::testing::Values(normalizedWithNonEncodedVariance), ::testing::Values(ssdResnet101ShapeParams),
        ssdResnet101Attributes, ::testing::ValuesIn(additionalInputsParams), ::testing::ValuesIn(metaParams),
        ::testing::Values(test_utils::TARGET_DEVICE));

// Unexpected application crash with code: 14
// Inputs: {1, 524160}, {1, 78886080}, {1, 2, 524160}
INSTANTIATE_TEST_SUITE_P(DISABLED_precommit_smoke_DetectionOutput_ssd_resnet101, DetectionOutputLayerTestCommon,
                         ssdResnet101Params, PrintTestCaseName());

}  // namespace ov::test
