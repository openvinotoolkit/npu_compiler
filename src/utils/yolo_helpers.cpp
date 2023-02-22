//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "yolo_helpers.hpp"

#include "vpux/utils/IE/blob.hpp"

#include <cmath>

namespace ie = InferenceEngine;

static int entryIndex(int lw, int lh, int lcoords, int lclasses, int lnum, int batch, int location, int entry) {
    int n = location / (lw * lh);
    int loc = location % (lw * lh);
    int loutputs = lh * lw * lnum * (lclasses + lcoords + 1);
    return batch * loutputs + n * lw * lh * (lcoords + lclasses + 1) + entry * lw * lh + loc;
}

static utils::Box getRegionBox(float* x, const std::vector<float>& biases, int n, int index, int i, int j, int w, int h,
                               int stride) {
    utils::Box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = std::exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = std::exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;

    return b;
}

static utils::Box getRegionBoxV3V4(const std::vector<float>& predictions, const std::vector<float>& anchor,
                                   int anchor_offset, int n, int box_index, int col, int row, int imw, int imh,
                                   int side, const std::function<float(const float)>& transformationFunc) {
    utils::Box b;
    b.x = (col + transformationFunc(predictions[box_index + 0 * side * side])) / side;
    b.y = (row + transformationFunc(predictions[box_index + 1 * side * side])) / side;
    b.w = std::exp(predictions[box_index + 2 * side * side]) * anchor[anchor_offset + 2 * n] / imw;
    b.h = std::exp(predictions[box_index + 3 * side * side]) * anchor[anchor_offset + 2 * n + 1] / imh;

    return b;
}

static void correctRegionBoxes(std::vector<utils::Box>& boxes, int n, int w, int h, int netw, int neth, int relative) {
    int new_w = 0;
    int new_h = 0;
    if ((static_cast<float>(netw) / w) < (static_cast<float>(neth) / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }

    IE_ASSERT(static_cast<int>(boxes.size()) >= n);
    for (int i = 0; i < n; ++i) {
        utils::Box b = boxes[i];
        b.x = (b.x - (netw - new_w) / 2. / netw) / (static_cast<float>(new_w) / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (static_cast<float>(new_h) / neth);
        b.w *= static_cast<float>(netw) / new_w;
        b.h *= static_cast<float>(neth) / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

static void getRegionBoxesV3V4(const std::vector<std::vector<float>>& predictions, int w, int h, int lclasses,
                               int lcoords, int lnum, const std::vector<float>& anchors,
                               std::vector<std::vector<size_t>>& blobWH, float thresh,
                               std::vector<std::vector<float>>& probs, std::vector<utils::Box>& boxes,
                               const std::function<float(const float)>& transformationFunc,
                               const std::function<int(const size_t, const int)>& anchorFunc) {
    for (size_t iout = 0; iout < predictions.size(); ++iout) {
        int lw = blobWH[iout][0];
        int lh = blobWH[iout][1];
        int anchorOffset = anchorFunc(lnum, iout);

        for (int i = 0; i < lw * lh; ++i) {
            int row = i / lw;
            int col = i % lw;

            for (int n = 0; n < lnum; ++n) {
                int obj_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
                float scale = predictions[iout][obj_index];
                float max = 0;
                for (int j = 0; j < lclasses; ++j) {
                    int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                    float prob = transformationFunc(scale) * transformationFunc(predictions[iout][class_index]);
                    if (max < prob)
                        max = prob;
                }
                if (max < thresh)
                    continue;

                int box_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);

                boxes.push_back(getRegionBoxV3V4(predictions[iout], anchors, anchorOffset, n, box_index, col, row, w, h,
                                                 lw, transformationFunc));

                std::vector<float> prob(lclasses + 1, 0.0);
                for (int j = 0; j < lclasses; ++j) {
                    int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                    float probability = transformationFunc(scale) * transformationFunc(predictions[iout][class_index]);
                    prob[j] = probability > thresh ? probability : 0;
                }
                prob[lclasses] = max;
                probs.push_back(prob);
            }
        }
    }
}

static void getRegionBoxes(std::vector<float>& predictions, int lw, int lh, int lcoords, int lclasses, int lnum, int w,
                           int h, int netw, int neth, float thresh, std::vector<std::vector<float>>& probs,
                           std::vector<utils::Box>& boxes, int relative, const std::vector<float>& anchors) {
    for (int i = 0; i < lw * lh; ++i) {
        int row = i / lw;
        int col = i % lw;
        for (int n = 0; n < lnum; ++n) {
            int index = n * lw * lh + i;
            int obj_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
            int box_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);
            float scale = predictions[obj_index];

            boxes[index] = getRegionBox(predictions.data(), anchors, n, box_index, col, row, lw, lh, lw * lh);

            float max = 0;
            for (int j = 0; j < lclasses; ++j) {
                int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                float prob = scale * predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if (prob > max)
                    max = prob;
            }
            probs[index][lclasses] = max;
        }
    }
    correctRegionBoxes(boxes, lw * lh * lnum, w, h, netw, neth, relative);
}

struct sortableYoloBBox {
    int index;
    int cclass;
    std::vector<std::vector<float>> probs;
    sortableYoloBBox(int index, float cclass, std::vector<std::vector<float>>& probs)
            : index(index), cclass(cclass), probs(probs){};
};

static float overlap(float x1, float w1, float x2, float w2) {
    const float l1 = x1 - w1 / 2;
    const float l2 = x2 - w2 / 2;
    const float left = l1 > l2 ? l1 : l2;

    const float r1 = x1 + w1 / 2;
    const float r2 = x2 + w2 / 2;
    const float right = r1 < r2 ? r1 : r2;

    return right - left;
}

static float boxIntersection(const utils::Box& a, const utils::Box& b) {
    const float w = overlap(a.x, a.w, b.x, b.w);
    const float h = overlap(a.y, a.h, b.y, b.h);

    if (w < 0 || h < 0) {
        return 0.0f;
    }

    return w * h;
}

static float boxUnion(const utils::Box& a, const utils::Box& b) {
    const float i = boxIntersection(a, b);
    return a.w * a.h + b.w * b.h - i;
}

float utils::boxIntersectionOverUnion(const utils::Box& a, const utils::Box& b) {
    return boxIntersection(a, b) / boxUnion(a, b);
}

static void doNonMaximumSupressionSort(std::vector<utils::Box>& boxes, std::vector<std::vector<float>>& probs,
                                       int total, int classes, float thresh) {
    std::vector<sortableYoloBBox> boxCandidates;

    for (int i = 0; i < total; ++i) {
        sortableYoloBBox candidate(i, 0, probs);
        boxCandidates.push_back(candidate);
    }

    for (int k = 0; k < classes; ++k) {
        for (int i = 0; i < total; ++i) {
            boxCandidates[i].cclass = k;
        }
        std::sort(boxCandidates.begin(), boxCandidates.end(), [](const sortableYoloBBox& a, const sortableYoloBBox& b) {
            float diff = a.probs[a.index][b.cclass] - b.probs[b.index][b.cclass];
            return diff > 0;
        });
        for (int i = 0; i < total; ++i) {
            if (probs[boxCandidates[i].index][k] == 0)
                continue;
            utils::Box a = boxes[boxCandidates[i].index];
            for (int j = i + 1; j < total; ++j) {
                utils::Box b = boxes[boxCandidates[j].index];
                if (utils::boxIntersectionOverUnion(a, b) > thresh) {
                    probs[boxCandidates[j].index][k] = 0;
                }
            }
        }
    }
}

static int maxIndex(std::vector<float>& a, int n) {
    if (n <= 0) {
        return -1;
    }
    int max_i = 0;
    float max = a[0];
    for (int i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

static float clampToImageSize(const float& valueToClamp, const float& low, const float& high) {
    float result = valueToClamp;
    if (valueToClamp > high) {
        result = high;
    } else if (valueToClamp < low) {
        result = low;
    } else {
        result = valueToClamp;
    }

    return result;
}

static void getDetections(int imw, int imh, int num, float thresh, utils::Box* boxes,
                          std::vector<std::vector<float>>& probs, int classes,
                          std::vector<utils::BoundingBox>& detect_result) {
    for (int i = 0; i < num; ++i) {
        int idxClass = maxIndex(probs[i], classes);
        float prob = probs[i][idxClass];

        if (prob > thresh) {
            utils::Box b = boxes[i];

            float left = (b.x - b.w / 2.) * imw;
            float right = (b.x + b.w / 2.) * imw;
            float top = (b.y - b.h / 2.) * imh;
            float bot = (b.y + b.h / 2.) * imh;
            float clampedLeft = clampToImageSize(left, 0, imw);
            float clampedRight = clampToImageSize(right, 0, imw);
            float clampedTop = clampToImageSize(top, 0, imh);
            float clampedBottom = clampToImageSize(bot, 0, imh);

            utils::BoundingBox bx(idxClass, clampedLeft, clampedTop, clampedRight, clampedBottom, prob);
            detect_result.push_back(bx);
        }
    }
}

static std::vector<utils::BoundingBox> yolov2BoxExtractor(float threshold, std::vector<float>& net_out, int imgWidth,
                                                          int imgHeight, int class_num, bool isTiny) {
    int classes = class_num;
    int coords = 4;
    int num = 5;
    std::vector<utils::BoundingBox> boxes_result;

    std::vector<float> TINY_YOLOV2_ANCHORS = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
    std::vector<float> YOLOV2_ANCHORS = {1.3221,  1.73145, 3.19275, 4.00944, 5.05587,
                                         8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
    std::vector<float> YOLOV2_ANCHORS_80_CLASSES = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                                                    5.47434, 7.88282,  3.52778, 9.77052, 9.16828};

    int imw = 416;
    int imh = 416;

    int lw = 13;
    int lh = 13;
    float nms = 0.4f;

    std::vector<utils::Box> boxes(lw * lh * num);
    std::vector<std::vector<float>> probs(lw * lh * num, std::vector<float>(classes + 1, 0.0));

    // TODO refactoring ticket S#37819
    std::vector<float> anchors;
    if (isTiny) {
        anchors = TINY_YOLOV2_ANCHORS;
    } else {
        anchors = YOLOV2_ANCHORS;
        if (class_num == 80) {
            anchors = YOLOV2_ANCHORS_80_CLASSES;
        }
    }

    getRegionBoxes(net_out, lw, lh, coords, classes, num, imgWidth, imgHeight, imw, imh, threshold, probs, boxes, 1,
                   anchors);

    doNonMaximumSupressionSort(boxes, probs, lw * lh * num, classes, nms);
    getDetections(imgWidth, imgHeight, lw * lh * num, threshold, boxes.data(), probs, classes, boxes_result);

    return boxes_result;
}

static std::vector<utils::BoundingBox> yolov3v4BoxExtractor(
        std::vector<std::vector<float>>& net_out, int imgW, int imgH, int classes, int coords, int num,
        const std::vector<float>& anchors, std::vector<std::vector<size_t>>& blobWH, float threshold, float nms,
        const std::function<float(const float)>& transformationFunc,
        const std::function<int(const size_t, const int)>& anchorFunc) {
    std::vector<utils::BoundingBox> boxes_result;
    std::vector<utils::Box> boxes;
    std::vector<std::vector<float>> probs;

    getRegionBoxesV3V4(net_out, imgW, imgH, classes, coords, num, anchors, blobWH, threshold, probs, boxes,
                       transformationFunc, anchorFunc);
    doNonMaximumSupressionSort(boxes, probs, probs.size(), classes, nms);
    getDetections(imgW, imgH, probs.size(), threshold, boxes.data(), probs, classes, boxes_result);

    return boxes_result;
}

static std::vector<utils::BoundingBox> SSDBoxExtractor(float threshold, std::vector<float>& net_out, size_t imgWidth,
                                                       size_t imgHeight) {
    std::vector<utils::BoundingBox> boxes_result;

    if (net_out.empty()) {
        return boxes_result;
    }
    size_t oneDetectionSize = 7;

    IE_ASSERT(net_out.size() % oneDetectionSize == 0);

    for (size_t i = 0; i < net_out.size() / oneDetectionSize; i++) {
        if (net_out[i * oneDetectionSize + 2] > threshold) {
            boxes_result.emplace_back(net_out[i * oneDetectionSize + 1], net_out[i * oneDetectionSize + 3] * imgWidth,
                                      net_out[i * oneDetectionSize + 4] * imgHeight,
                                      net_out[i * oneDetectionSize + 5] * imgWidth,
                                      net_out[i * oneDetectionSize + 6] * imgHeight, net_out[i * oneDetectionSize + 2]);
        }
    }

    return boxes_result;
}

std::vector<utils::BoundingBox> utils::parseYoloOutput(const ie::Blob::Ptr& blob, size_t imgWidth, size_t imgHeight,
                                                       float confThresh, bool isTiny) {
    auto ptr = blob->cbuffer().as<float*>();
    IE_ASSERT(ptr != nullptr);

    std::vector<float> results(blob->size());
    for (size_t i = 0; i < blob->size(); i++) {
        results[i] = ptr[i];
    }

    std::vector<utils::BoundingBox> out;
    int classes = 20;
    out = yolov2BoxExtractor(confThresh, results, imgWidth, imgHeight, classes, isTiny);

    return out;
}

std::vector<utils::BoundingBox> utils::parseYoloV3Output(const ie::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
                                                         int classes, int coords, int num,
                                                         const std::vector<float>& anchors, float confThresh,
                                                         InferenceEngine::Layout layout) {
    auto funcV3 = [](const float x) -> float {
        return x;
    };

    auto anchorOffsetV3 = [&](const size_t iout, const int lnum) -> int {
        return (blobs.size() - 1) * (lnum * (blobs.size() - 1 - iout));
    };

    auto result = parseYoloV3V4Output(blobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh, layout,
                                      funcV3, anchorOffsetV3);
    return result;
}

std::vector<utils::BoundingBox> utils::parseYoloV4Output(const ie::BlobMap& blobs, size_t imgWidth, size_t imgHeight,
                                                         int classes, int coords, int num,
                                                         const std::vector<float>& anchors, float confThresh,
                                                         InferenceEngine::Layout layout) {
    auto funcV4 = [](const float x) -> float {
        return 1 / (1 + std::exp(-x));
    };

    auto anchorOffsetV4 = [&](const size_t iout, const int lnum) -> int {
        return lnum * 2 * iout;
    };

    auto result = parseYoloV3V4Output(blobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh, layout,
                                      funcV4, anchorOffsetV4);
    return result;
}

std::vector<utils::BoundingBox> utils::parseYoloV3V4Output(
        const ie::BlobMap& blobs, size_t imgWidth, size_t imgHeight, int classes, int coords, int num,
        const std::vector<float>& anchors, float confThresh, InferenceEngine::Layout layout,
        const std::function<float(const float)>& transformationFunc,
        const std::function<int(const size_t, const int)>& anchorFunc) {
    std::vector<std::vector<float>> results;
    std::vector<std::vector<size_t>> blobWH;
    for (auto blob : blobs) {
        auto blobFP32 = vpux::toFP32(ie::as<ie::MemoryBlob>(blob.second));
        auto ptr = blobFP32->cbuffer().as<float*>();
        IE_ASSERT(ptr != nullptr);

        size_t C = blobFP32->getTensorDesc().getDims()[1];
        size_t H = blobFP32->getTensorDesc().getDims()[2];
        size_t W = blobFP32->getTensorDesc().getDims()[3];

        std::vector<float> result(blobFP32->size());
        if (layout == InferenceEngine::NCHW) {
            for (size_t j = 0; j < blobFP32->size(); j++) {
                result[j] = ptr[j];
            }
        } else if (layout == InferenceEngine::NHWC) {
            // TODO may be using copyBlob is good decision but can't find a way how include it
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        result[c * H * W + h * W + w] = ptr[h * W * C + w * C + c];
                    }
                }
            }
        }
        results.push_back(result);
        blobWH.push_back(std::vector<size_t>{W, H});
    }

    return yolov3v4BoxExtractor(results, imgWidth, imgHeight, classes, coords, num, anchors, blobWH, confThresh, 0.4f,
                                transformationFunc, anchorFunc);
}

std::vector<utils::BoundingBox> utils::parseSSDOutput(const InferenceEngine::Blob::Ptr& blob, size_t imgWidth,
                                                      size_t imgHeight, float confThresh) {
    auto blobPtr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    IE_ASSERT(blobPtr != nullptr);

    auto minputHolder = blobPtr->wmap();
    auto dataPtr = minputHolder.as<float*>();
    std::vector<float> results(blob->size());
    std::copy_n(dataPtr, blob->size(), results.begin());

    std::vector<utils::BoundingBox> out;
    out = SSDBoxExtractor(confThresh, results, imgWidth, imgHeight);
    return out;
}

void utils::printDetectionBBoxOutputs(std::vector<utils::BoundingBox>& actualOutput, std::ostringstream& outputStream,
                                      const std::vector<std::string>& labels) {
    outputStream << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        outputStream << i << " : ";
        if (static_cast<int>(labels.size()) < bb.idx) {
            outputStream << bb.idx;
        } else {
            outputStream << labels.at(bb.idx);
        }
        outputStream << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " " << bb.bottom
                     << ")] : " << bb.prob * 100 << "%" << std::endl;
    }
}
