//
// Copyright 2016-2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <frontend_mcm.hpp>
#include "kmb_config.h"
#include <utils/dims_parser.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <limits>
#include <functional>

#include <ie_layers_internal.hpp>
#include "ie_blob.h"

#include <precision_utils.h>

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/tensor/quantization_params.hpp"

namespace vpu {

namespace KmbPlugin {

namespace {

static std::unordered_map<int, char> DIM_NAMES({
    {3, 'W'},
    {2, 'H'},
    {1, 'C'},
    {0, 'N'}
});

}  // namespace

template<typename ResultType>
std::vector<ResultType> packBlobToVector(
        ie::Blob::Ptr blobPtr,
        size_t expectedSize) {
    IE_ASSERT(blobPtr != nullptr);

    std::vector<ResultType> blobData(expectedSize, 0);

    // TODO: Make the ASSERT on equality after correction of blob creation in tests
    IE_ASSERT(expectedSize <= blobPtr->size());

    ie::Precision blobPrecision = blobPtr->getTensorDesc().getPrecision();

    // TODO: add proper layout handling. for now, weights are assumed to have OIYX
    if (blobPrecision == ie::Precision::FP16) {
        const auto* blobDataFP16 = blobPtr->cbuffer().as<const fp16_t*>();
        IE_ASSERT(blobDataFP16 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = ie::PrecisionUtils::f16tof32(blobDataFP16[pos]);
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::FP32) {
        const auto* blobDataFP32 = blobPtr->cbuffer().as<const float*>();
        IE_ASSERT(blobDataFP32 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataFP32[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I8) {
        const auto* blobDataI8 = blobPtr->cbuffer().as<const int8_t*>();
        IE_ASSERT(blobDataI8 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI8[pos];
            blobData[pos] = val;
        }
    } else if (blobPrecision == ie::Precision::I32) {
        const auto* blobDataI32 = blobPtr->cbuffer().as<const int32_t*>();
        IE_ASSERT(blobDataI32 != nullptr);

        for (size_t pos = 0; pos < expectedSize; pos++) {
            ResultType val = blobDataI32[pos];
            blobData[pos] = val;
        }
    } else {
        THROW_IE_EXCEPTION << "precision '" << blobPrecision << "' is not supported";
    }

    return blobData;
}

mv::QuantizationParams createQuantParams(const ie::CNNLayerPtr& layer, std::string bName) {
    mv::QuantizationParams quantParams = {{}, {}, {}, {}};
    double inf = std::numeric_limits<double>::infinity();
    ie::Blob::Ptr scaleBlob;
    auto blob = layer->blobs.find(bName);

    if (blob != layer->blobs.end()) {
        scaleBlob = blob->second;
        auto scale = packBlobToVector<double>(scaleBlob, scaleBlob->size());
        quantParams = {{mv::utils::generateSequence<int64_t>(scale.size(), 0, 0)},
                       {scale},
                       {mv::utils::generateSequence<double>(scale.size(), -inf, 0)},
                       {mv::utils::generateSequence<double>(scale.size(), inf, 0)}};
    }

    return quantParams;
}

void getOutputScale(const ie::CNNLayerPtr& layer, mv::QuantizationParams &quantParams) {
    std::vector<double> oiScaleData, wScaleData;
    IE_ASSERT(layer->blobs["weights"] != nullptr);
    IE_ASSERT(layer->blobs["weights"]->getTensorDesc().getPrecision() == ie::Precision::I8);
    // quantized layer shall contain mandatory dequantize scale and optional requantize scale
    // extract dequantize scale
    IE_ASSERT(layer->blobs["w-scale"] != nullptr);
    auto blob = layer->blobs.find("w-scale");
    if (blob != layer->blobs.end()) {
        wScaleData = packBlobToVector<double>(blob->second, blob->second->size());
    }
    // placeholder for resulted output scale
    std::vector<double> oScaleDataVector(wScaleData.size(), 0);
    if (layer->outData[0]->getPrecision() == ie::Precision::I8 ||
        layer->outData[0]->getPrecision() == ie::Precision::U8) {
        // next layer is quantized therefore extract requantize scale oi-scale
        // resulted scale will be w-scale/oi-scale
        blob = layer->blobs.find("oi-scale");
        if (blob != layer->blobs.end()) {
            oiScaleData = packBlobToVector<double>(blob->second, blob->second->size());
        }
        for (size_t c = 0; c < wScaleData.size(); c++) {
            oScaleDataVector[c] = (wScaleData[c] / oiScaleData[c]);
        }
    } else {
        oScaleDataVector = wScaleData;
    }
    double inf = std::numeric_limits<double>::infinity();
    quantParams =  {{mv::utils::generateSequence<int64_t>(oScaleDataVector.size(), 0, 0)},
                    {oScaleDataVector},
                    {mv::utils::generateSequence<double>(oScaleDataVector.size(), -inf, 0)},
                    {mv::utils::generateSequence<double>(oScaleDataVector.size(), inf, 0)}};
}

mv::DType convert_data_type(ie::Precision iePrecision) {
    mv::DType mvType;
    switch (iePrecision) {
    case ie::Precision::I8:
        mvType = mv::DType("Int8");
        break;
    case ie::Precision::U8:
        mvType = mv::DType("UInt8");
        break;
    case ie::Precision::I32:
        mvType = mv::DType("Int32");
        break;
    case ie::Precision::FP16:
        mvType = mv::DType("Float16");
        break;
    case ie::Precision::FP32:
        mvType = mv::DType("Float32");
        break;
    default:
        VPU_THROW_EXCEPTION << "Data type handling is not implemented" << iePrecision.name();
    }
    return mvType;
}

void FrontEndMcm::parseConvolution(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    bool is_quantized = false;
    bool with_bias = false;
    //
    // Extract parameters
    //
    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    int kernelSizeX = convLayer->_kernel_x;
    int kernelSizeY = convLayer->_kernel_y;

    int kernelStrideX = convLayer->_stride_x;
    int kernelStrideY = convLayer->_stride_y;

    auto paddings = getPaddings(*convLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = convLayer->_dilation_x;
    int dilationY = convLayer->_dilation_y;
    if (dilationX != dilationY) {
        VPU_THROW_EXCEPTION << "kmb Convolution supports only equal dilationX and dilationY";
    }

    size_t groupSize = convLayer->_group;

    // Quantization parameters
    mv::QuantizationParams weightsQuantParams = {{}, {}, {}, {}};
    mv::QuantizationParams inputQuantParams   = {{}, {}, {}, {}};
    mv::QuantizationParams outputQuantParams  = {{}, {}, {}, {}};

    if (layer->precision == ie::Precision::I8) {
        // Quantized layer
        double inf = std::numeric_limits<double>::infinity();
        inputQuantParams   = {{0}, {1}, {-inf}, {inf}};
        weightsQuantParams = {{0}, {1}, {-inf}, {inf}};
        getOutputScale(layer, outputQuantParams);
        is_quantized = true;
    }

    auto layerOutput = layer->outData[0];

    IE_ASSERT(layerOutput != nullptr);
    auto outDesc = layerOutput->getTensorDesc();
    mv::Data::TensorIterator mvConv;
    mv::Data::TensorIterator mvConvOnly;
    mv::Data::TensorIterator mvWeights;
    mv::Data::TensorIterator mvBiases;

    ie::Blob::Ptr biasBlob = nullptr;
    mv::Shape biasesShape {1};
    auto bias = layer->blobs.find("biases");
    if (bias != layer->blobs.end()) {
        with_bias = true;
        biasBlob = bias->second;
        biasesShape[0] = biasBlob->size();
    }

    _logger->debug("Convolution orig: '%s' from '%s' ", convLayer->name, input->getMcmNode()->getName());
    size_t inputGroupSize, outputGroupSize, stub;
    parseDims(input->desc(), stub, inputGroupSize, stub, stub);
    parseDims(outDesc, stub, outputGroupSize, stub, stub);
    if (groupSize > 1 && groupSize == inputGroupSize && groupSize != outputGroupSize) {
        auto weightsShape = {static_cast<std::size_t>(kernelSizeX),
                             static_cast<std::size_t>(kernelSizeY),
                             inputGroupSize,
                             1lu};
        int weightsSize = kernelSizeX * kernelSizeY * inputGroupSize;
        if (is_quantized) {
            // TODO: create per layer test
            auto weightsData = packBlobToVector<int64_t>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constantInt(weightsData,
                                              weightsShape,
                                              mv::DType("Int8"),
                                              mv::Order(mv::Order::getColMajorID(4)));
            mvWeights->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
            if (with_bias) {
                auto biasesData = packBlobToVector<int64_t>(biasBlob, biasBlob->size());
                mvBiases = _modelMcm.constantInt(
                    biasesData,
                    biasesShape,
                    // TODO: Biases data type should be discussed with mcmCompiler team
                    mv::DType("UInt8"), mv::Order::getColMajorID(1));
                mvBiases->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
            }
        } else {
            auto weightsData = packBlobToVector<double>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constant(weightsData,
                                           weightsShape,
                                           mv::DType("Float16"),
                                           mv::Order(mv::Order::getColMajorID(4)));
            if (with_bias) {
                auto biasesData = packBlobToVector<double>(biasBlob, biasBlob->size());
                mvBiases = _modelMcm.constant(
                    biasesData,
                    biasesShape,
                    mv::DType("Float16"), mv::Order::getColMajorID(1));
            }
        }

        mvConv = _modelMcm.depthwiseConv(input->getMcmNode(),
                                         mvWeights,
                                        {static_cast<uint16_t>(kernelStrideX),
                                         static_cast<uint16_t>(kernelStrideY)},
                                        {static_cast<uint16_t>(padLeft),
                                         static_cast<uint16_t>(padRight),
                                         static_cast<uint16_t>(padTop),
                                         static_cast<uint16_t>(padBottom)},
                                         static_cast<unsigned>(dilationX),
                                         inputQuantParams,
                                         convLayer->name);
    } else {
        size_t inputC, outputC, stub;
        parseDims(input->desc(), stub, inputC, stub, stub, 1);
        parseDims(outDesc, stub, outputC, stub, stub, 1);
        auto weightsShape = {static_cast<std::size_t>(kernelSizeX),
                             static_cast<std::size_t>(kernelSizeY),
                             inputC,
                             outputC / groupSize};
        int weightsSize = std::accumulate(weightsShape.begin(), weightsShape.end(), 1, std::multiplies<int>());
        if (is_quantized) {
            auto weightsData = packBlobToVector<int64_t>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constantInt(weightsData,
                                              weightsShape,
                                              mv::DType("Int8"),
                                              mv::Order("NCWH"));
            mvWeights->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
            if (with_bias) {
                auto biasesData = packBlobToVector<int64_t>(biasBlob, biasBlob->size());
                mvBiases = _modelMcm.constantInt(
                    biasesData,
                    biasesShape,
                    // TODO: Biases data type should be discussed with mcmCompiler team
                    mv::DType("UInt8"), mv::Order::getColMajorID(1));
                mvBiases->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
            }
        } else {
            auto weightsData = packBlobToVector<double>(layer->blobs["weights"], weightsSize);
            mvWeights = _modelMcm.constant(weightsData,
                                           weightsShape,
                                           mv::DType("Float16"),
                                           mv::Order("NCWH"));
            if (with_bias) {
                auto biasesData = packBlobToVector<double>(biasBlob, biasBlob->size());
                mvBiases = _modelMcm.constant(
                    biasesData,
                    biasesShape,
                    mv::DType("Float16"), mv::Order::getColMajorID(1));
            }
        }

        mvConv = _modelMcm.conv(input->getMcmNode(),
                                mvWeights,
                               {static_cast<uint16_t>(kernelStrideX),
                                static_cast<uint16_t>(kernelStrideY)},
                               {static_cast<uint16_t>(padLeft),
                                static_cast<uint16_t>(padRight),
                                static_cast<uint16_t>(padTop),
                                static_cast<uint16_t>(padBottom)},
                                static_cast<unsigned>(dilationX),
                                static_cast<unsigned>(groupSize),
                                inputQuantParams,
                                convLayer->name);
    }
    if (is_quantized) {
        mvConv->set<mv::QuantizationParams>("quantParams", outputQuantParams);
    }

    if (with_bias) {
        mvConvOnly = mvConv;
        mvConv = _modelMcm.bias(mvConvOnly, mvBiases, outputQuantParams, convLayer->name + ":bias");
        _logger->debug("conv orig: '%s' Bias part (%s) added to mcmModel", convLayer->name, mvConv->getName());
    }

    mvConv->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    auto conv = std::make_shared<McmNodeObject>(mvConv, outDesc);
    _nodes.push_back(conv);

    bindData(conv, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvConv->getName());
}

void FrontEndMcm::parsePooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    int kernelSizeX = poolLayer->_kernel_x;
    int kernelSizeY = poolLayer->_kernel_y;

    int kernelStrideX = poolLayer->_stride_x;
    int kernelStrideY = poolLayer->_stride_y;

    auto paddings = getPaddings(*poolLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    auto poolType = poolLayer->_type;

    _logger->debug("Pooling orig: '%s' from '%s'", poolLayer->name, inputs[0]->getMcmNode()->getName());
    mv::Data::TensorIterator mvPooling;
    if (poolType == ie::PoolingLayer::AVG) {
        mvPooling = _modelMcm.averagePool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX),
             static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX),
             static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft),
             static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop),
             static_cast<uint16_t>(padBottom)},
            true, "", "floor", {{}, {}, {}, {}},
            poolLayer->name);
    } else {
        mvPooling = _modelMcm.maxPool(inputs[0]->getMcmNode(),
            {static_cast<uint16_t>(kernelSizeX),
             static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX),
             static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft),
             static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop),
             static_cast<uint16_t>(padBottom)},
            true, "", "floor", {{}, {}, {}, {}},
            poolLayer->name);
    }

    mvPooling->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto pool = std::make_shared<McmNodeObject>(mvPooling, layerOutput->getTensorDesc());

    _nodes.push_back(pool);
    bindData(pool, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvPooling->getName());
}

void FrontEndMcm::parseFullyConnected(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto FClayer = std::dynamic_pointer_cast<ie::FullyConnectedLayer>(layer);
    IE_ASSERT(layer != nullptr);

    bool is_quantized = false;
    bool with_bias = false;
    // Quantization parameters
    mv::QuantizationParams weightsQuantParams = {{}, {}, {}, {}};
    mv::QuantizationParams inputQuantParams   = {{}, {}, {}, {}};
    mv::QuantizationParams outputQuantParams  = {{}, {}, {}, {}};

    if (layer->precision == ie::Precision::I8) {
        // Quantized layer
        double inf = std::numeric_limits<double>::infinity();
        inputQuantParams   = {{0}, {1}, {-inf}, {inf}};
        weightsQuantParams = {{0}, {1}, {-inf}, {inf}};
        getOutputScale(layer, outputQuantParams);
        is_quantized = true;
    }

    auto input = inputs[0];

    //
    // Create const datas
    //

    ie::Blob::Ptr biasBlob = nullptr;
    mv::Shape biasesShape {1};
    mv::Data::TensorIterator mvBiases;

    auto bias = layer->blobs.find("biases");
    if (bias != layer->blobs.end()) {
        with_bias = true;
        biasBlob = bias->second;
        biasesShape[0] = biasBlob->size();
    }

    mv::Data::TensorIterator mvWeights;
    size_t dimC, dimY, dimX, stub;
    parseDims(input->desc(), stub, dimC, dimY, dimX, 1);
    int weightsSize = static_cast<int>(dimX * dimY * dimC * FClayer->_out_num);
    if (is_quantized) {
        std::vector<int64_t> weightsData = packBlobToVector<int64_t>(FClayer->blobs["weights"], weightsSize);
        mvWeights = _modelMcm.constantInt(weightsData,
                                              {inputs[0]->getMcmNode()->getShape().totalSize(), static_cast<std::size_t>(FClayer->_out_num)},
                                               mv::DType("Int8"), mv::Order(mv::Order::getColMajorID(2)));

        mvWeights->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
        if (with_bias) {
            auto biasesData = packBlobToVector<int64_t>(biasBlob, biasBlob->size());
            mvBiases = _modelMcm.constantInt(
                biasesData,
                biasesShape,
                // TODO: Biases data type should be discussed with mcmCompiler team
                mv::DType("UInt8"), mv::Order::getColMajorID(1));
            mvBiases->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
        }
    } else {
        std::vector<double> weightsData = packBlobToVector<double>(FClayer->blobs["weights"], weightsSize);

        mvWeights = _modelMcm.constant(weightsData,
                                           {inputs[0]->getMcmNode()->getShape().totalSize(), static_cast<std::size_t>(FClayer->_out_num)},
                                            mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));
        if (with_bias) {
            auto biasesData = packBlobToVector<double>(biasBlob, biasBlob->size());
            mvBiases = _modelMcm.constant(
                biasesData,
                biasesShape,
                mv::DType("Float16"), mv::Order::getColMajorID(1));
        }
    }

    auto layerOutput = FClayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    _logger->debug("FullyConnected orig: '%s' from '%s'",
            FClayer->name, input->getMcmNode()->getName());
    auto mvFullyConnected = _modelMcm.fullyConnected(input->getMcmNode(), mvWeights, inputQuantParams, FClayer->name);

    if (is_quantized) {
        mvFullyConnected->set<mv::QuantizationParams>("quantParams", outputQuantParams);
    }

    if (with_bias) {
        auto mvFCOnly = mvFullyConnected;
        mvFullyConnected = _modelMcm.bias(mvFCOnly, mvBiases, outputQuantParams, FClayer->name + ":bias");
        _logger->debug("FullyConnected orig: '%s' Bias part (%s) added to mcmModel", FClayer->name, mvFullyConnected->getName());
    }

    mvFullyConnected->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    auto fullyConnected = std::make_shared<McmNodeObject>(mvFullyConnected, layerOutput->getTensorDesc());

    _nodes.push_back(fullyConnected);
    bindData(fullyConnected, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvFullyConnected->getName());
}

namespace {

union FloatUint {
    float flt;
    uint32_t unt;
};

}  // namespace

void FrontEndMcm::parseReLU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto reluLayer = std::dynamic_pointer_cast<ie::ReLULayer>(layer);
    IE_ASSERT(reluLayer != nullptr);

    _logger->debug("ReLU orig: '%s' from '%s'", reluLayer->name, inputs[0]->getMcmNode()->getName());

    float negativeSlope = reluLayer->negative_slope;
    mv::Data::TensorIterator mvRelu;
    if (std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon()) {
        mvRelu = _modelMcm.relu(inputs[0]->getMcmNode(), {{}, {}, {}, {}}, reluLayer->name);
    } else {
        // TODO FIXME: unsigned int alpha should be fixed or clarified
        mvRelu = _modelMcm.leakyRelu(inputs[0]->getMcmNode(),
                negativeSlope,
                reluLayer->name);
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto relu = std::make_shared<McmNodeObject>(mvRelu, layerOutput->getTensorDesc());

    _nodes.push_back(relu);
    bindData(relu, layerOutput);
    _logger->debug("parsed to mcmModel as '%s", mvRelu->getName());
}

void FrontEndMcm::parseSoftMax(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto softMaxLayer = std::dynamic_pointer_cast<ie::SoftMaxLayer>(layer);
    IE_ASSERT(softMaxLayer != nullptr);

    IE_ASSERT(static_cast<size_t>(softMaxLayer->axis) < inputs[0]->desc().getDims().size());

    _logger->debug("Softmax orig: '%s' from '%s'", softMaxLayer->name, inputs[0]->getMcmNode()->getName());
    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[softMaxLayer->axis];
    auto mvSoftmax = _modelMcm.softmax(inputs[0]->getMcmNode(), mcmAxis, {{}, {}, {}, {}}, softMaxLayer->name);

    auto layerOutput = softMaxLayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto softmax = std::make_shared<McmNodeObject>(mvSoftmax, layerOutput->getTensorDesc());

    _nodes.push_back(softmax);
    bindData(softmax, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvSoftmax->getName());
}

void FrontEndMcm::parseNorm(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto normLayer = std::dynamic_pointer_cast<ie::NormLayer>(layer);
    IE_ASSERT(normLayer != nullptr);

    _logger->debug("LRN orig: '%s' from '%s'", normLayer->name, inputs[0]->getMcmNode()->getName());
    auto mvLRN = _modelMcm.localResponseNormalization(inputs[0]->getMcmNode(),
            normLayer->_size, normLayer->_k, normLayer->name);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto LRN = std::make_shared<McmNodeObject>(mvLRN, layerOutput->getTensorDesc());

    _nodes.push_back(LRN);
    bindData(LRN, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvLRN->getName());

    // TODO: add parsing following parameters
    // stage->attrs().set<float>("alpha", layer->_alpha);
    // stage->attrs().set<float>("beta", layer->_beta);
}

void FrontEndMcm::parseScale(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto scaleLayer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(layer);
    IE_ASSERT(scaleLayer != nullptr);
    IE_ASSERT(scaleLayer->_weights != nullptr);

    if (scaleLayer->_broadcast != 0) {
        VPU_THROW_EXCEPTION <<
            "Layer " << scaleLayer->name << " doesn't support broadcast param";
    }

    auto input = inputs[0];

    size_t dimC, stub;
    parseDims(input->desc(), stub, dimC, stub, stub);
    int weightsSize = static_cast<int>(dimC);
    auto weightsData = packBlobToVector<double>(scaleLayer->_weights, weightsSize);

    mv::Shape weightsShape = { dimC };
    auto mvWeights = _modelMcm.constant(
            weightsData,
            weightsShape,
            mv::DType("Float16"), mv::Order("W"));

    _logger->debug("ScaleShift orig: '%s' from '%s' ", scaleLayer->name, input->getMcmNode()->getName());
    auto mvScale = _modelMcm.scale(input->getMcmNode(), mvWeights, {{}, {}, {}, {}}, scaleLayer->name);
    auto mvScaleShift = mvScale;

    _logger->debug("ScaleShift orig: '%s' Scale part (%s) added to mcmModel", scaleLayer->name, mvScaleShift->getName());

    if (scaleLayer->_biases != nullptr) {
        size_t C, stub;
        parseDims(input->desc(), stub, C, stub, stub);
        int biasesSize = static_cast<int>(dimC);
        auto biasData = packBlobToVector<double>(scaleLayer->_biases, biasesSize);

        auto mvBias = _modelMcm.constant(
                biasData,
                weightsShape,
                mv::DType("Float16"), mv::Order("W"));
        mvScaleShift = _modelMcm.bias(mvScale, mvBias, {{}, {}, {}, {}}, scaleLayer->name + ":bias");
        _logger->debug("ScaleShift orig: '%s' Bias part (%s) added to mcmModel", scaleLayer->name, mvScaleShift->getName());
    }

    auto layerOutput = scaleLayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    mvScaleShift->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    auto scaleShift = std::make_shared<McmNodeObject>(mvScaleShift, layerOutput->getTensorDesc());
    _nodes.push_back(scaleShift);
    bindData(scaleShift, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvScaleShift->getName());
}

void FrontEndMcm::parsePermute(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto ieOrder = layer->GetParamAsInts("order");

    std::string newOrder;

//  4d NCHW inputs are supported
    for (size_t i = 0; i < ieOrder.size(); i++) {
        newOrder += DIM_NAMES[ieOrder[ieOrder.size() - 1 - i]];
    }

    _logger->debug("Permute orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvPerm = _modelMcm.permute(inputs[0]->getMcmNode(), mv::Order(newOrder), layer->name);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto perm = std::make_shared<McmNodeObject>(mvPerm, layerOutput->getTensorDesc());

    _nodes.push_back(perm);
    bindData(perm, layerOutput);
    _logger->debug("parsed to mcmModel as '%s", mvPerm->getName());
}

void FrontEndMcm::parseEltwise(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    if (inputs.size() != 2) {
        VPU_THROW_EXCEPTION << "Eltwise with 2 inputs is only supported by kmbPlugin";
    }

    auto eltwiseLayer = std::dynamic_pointer_cast<ie::EltwiseLayer>(layer);
    IE_ASSERT(eltwiseLayer != nullptr);

    bool isSumStage = false;
    auto addCoefficient0 = 1.0f;
    auto addCoefficient1 = 1.0f;

    // Quantization parameters
    mv::QuantizationParams secondInputQuantParams  = {{}, {}, {}, {}};

    if (layer->precision == ie::Precision::I8) {
        // Quantized layer
        ie::Blob::Ptr eScaleBlob;
        IE_ASSERT(layer->blobs["eltwise-sum-scale"] != nullptr);

        auto es = layer->blobs.find("eltwise-sum-scale");
        if ((layer->outData[0]->getPrecision() == ie::Precision::I8 || layer->outData[0]->getPrecision() == ie::Precision::U8)
                && es == layer->blobs.end()) {
            THROW_IE_EXCEPTION << "Internal error of graph quantization - mismatch of intermediate scales and next layer type for convolution "
                    << layer->name;
        }

        secondInputQuantParams  = createQuantParams(layer, "eltwise-sum-scale");
    }

    switch (eltwiseLayer->_operation) {
    case ie::EltwiseLayer::eOperation::Sum:
    case ie::EltwiseLayer::eOperation::Sub:
        addCoefficient1 = -1.0f;
        if (eltwiseLayer->coeff.size() > 0) {
            addCoefficient0 *= eltwiseLayer->coeff[0];
        }
        if (eltwiseLayer->coeff.size() > 1) {
            addCoefficient1 *= eltwiseLayer->coeff[1];
        }
        if (std::abs(addCoefficient0) != 1.0f || std::abs(addCoefficient1) != 1.0f ||
                (addCoefficient0 == -1.0f && addCoefficient1 == -1.0f)) {
            VPU_THROW_EXCEPTION << eltwiseLayer->name <<
                    " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
        }
            isSumStage = true;
        break;
    case ie::EltwiseLayer::eOperation::Prod:
        break;
    default:
        VPU_THROW_EXCEPTION << "Eltwise operation" << eltwiseLayer->_operation << " is not supported";
    }

    if (!isSumStage && !eltwiseLayer->coeff.empty()) {
        VPU_THROW_EXCEPTION << layer->name << " coefficients (1 and -1) are only supported for Sum/Sub operations.";
    }

    _logger->debug("eltwise orig: '%s' from '%s' and '%s'",
            eltwiseLayer->name, inputs[0]->getMcmNode()->getName(), inputs[1]->getMcmNode()->getName());
    mv::Data::TensorIterator mvEltwise;
    if (addCoefficient0 == 1.0f && addCoefficient1 == 1.0f) {
        mvEltwise = _modelMcm.add(inputs[0]->getMcmNode(), inputs[1]->getMcmNode(), secondInputQuantParams, eltwiseLayer->name);
    } else {
        if (addCoefficient0 == 1.0f && addCoefficient1 == -1.0f) {
            mvEltwise = _modelMcm.subtract(inputs[0]->getMcmNode(), inputs[1]->getMcmNode(), secondInputQuantParams, eltwiseLayer->name);
        } else {
            mvEltwise = _modelMcm.subtract(inputs[1]->getMcmNode(), inputs[0]->getMcmNode(), secondInputQuantParams, eltwiseLayer->name);
        }
    }

    mvEltwise->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto eltwise = std::make_shared<McmNodeObject>(mvEltwise, layerOutput->getTensorDesc());

    _nodes.push_back(eltwise);
    bindData(eltwise, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvEltwise->getName());
}

void FrontEndMcm::parseBias(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    mv::Data::TensorIterator mvBias;
    if (inputs.size() == 1) {
        auto input = inputs[0];
         _logger->debug("ScaleShift(Bias) orig: '%s' from '%s' ", layer->name, input->getMcmNode()->getName());
        size_t dimC, stub;
        parseDims(input->desc(), stub, dimC, stub, stub);
        mv::Shape biasShape = { dimC };
        int biasesSize = dimC;
        auto biases = layer->blobs["biases"];

        auto weights = layer->blobs["weights"];
        auto biasData = packBlobToVector<double>(biases, biasesSize);

        auto mvBiasValues = _modelMcm.constant(
                biasData,
                biasShape,
                mv::DType("Float16"), mv::Order("W"));
        mvBias = _modelMcm.bias(input->getMcmNode(), mvBiasValues, {{}, {}, {}, {}}, layer->name);
    } else if (inputs.size() == 2) {
        auto input = inputs[0];
        auto input1 = inputs[1];
        _logger->debug("ScaleShift(Bias) orig: '%s' from '%s', '%s' ", layer->name,
                        input->getMcmNode()->getName(), input1->getMcmNode()->getName());
        mvBias = _modelMcm.bias(input->getMcmNode(), input1->getMcmNode(), {{}, {}, {}, {}}, layer->name);
    } else {
        VPU_THROW_EXCEPTION << "Bias layer does not support " << inputs.size() << " inputs";
    }

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto bias = std::make_shared<McmNodeObject>(mvBias, layerOutput->getTensorDesc());
    _nodes.push_back(bias);
    bindData(bias, layerOutput);
    _logger->debug("parsed to mcmModel as '%s'", mvBias->getName());
}

void FrontEndMcm::parseClamp(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(inputs.size() == 1);

    auto clampLayer = std::dynamic_pointer_cast<ie::ClampLayer>(layer);
    IE_ASSERT(clampLayer != nullptr);

    _logger->debug("Clamp orig: '%s' from '%s'", clampLayer->name, inputs[0]->getMcmNode()->getName());
    auto mvClamp = _modelMcm.clamp(inputs[0]->getMcmNode(), clampLayer->min_value, clampLayer->max_value, clampLayer->name);

    auto layerOutput = clampLayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto clamp = std::make_shared<McmNodeObject>(mvClamp, layerOutput->getTensorDesc());

    _nodes.push_back(clamp);
    bindData(clamp, layerOutput);
    _logger->debug("parsed to mcmModel as '%s", mvClamp->getName());
}

void FrontEndMcm::parseReshape(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    // Because mcmCompiler supports only "dense" layouts
    // for example NC should be represented as NCHW with dims NC11
    // Formation of a newShape, "dense" shape with 1, substituted in the places of non-existent measurements
    // TODO: Tests on parsing/compilation of different cases of reshape should be added: Jira: CVS-20409
    // McmCompiler accept only input in WHCN format
    mv::Shape newShape(getWHCN(layerOutput->getTensorDesc()).getDims());

    _logger->debug("Reshape orig: '%s' from '%s'", layer->name, inputs[0]->getMcmNode()->getName());
    auto mvReshape = _modelMcm.reshape(inputs[0]->getMcmNode(), newShape, "", layer->name);

    auto reshape = std::make_shared<McmNodeObject>(mvReshape, layerOutput->getTensorDesc());

    _nodes.push_back(reshape);
    bindData(reshape, layerOutput);
    _logger->debug("parsed to mcmModel as '%s", mvReshape->getName());
}

void FrontEndMcm::parseConcat(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);

    IE_ASSERT(!inputs.empty());

    auto clampLayer = std::dynamic_pointer_cast<ie::ConcatLayer>(layer);
    IE_ASSERT(clampLayer != nullptr);
    IE_ASSERT(clampLayer->_axis < inputs[0]->desc().getDims().size());

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[clampLayer->_axis];
    std::vector<mv::Data::TensorIterator> concatInputs;

    for (const auto & input : inputs) {
        concatInputs.push_back(input->getMcmNode());
    }

    _logger->debug("Concat orig: '%s' from '%s'", clampLayer->name, inputs[0]->getMcmNode()->getName());
    auto mvConcat = _modelMcm.concat(concatInputs, mcmAxis, {{}, {}, {}, {}}, clampLayer->name + ":step0");

    auto layerOutput = clampLayer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    auto concat = std::make_shared<McmNodeObject>(mvConcat, layerOutput->getTensorDesc());

    _nodes.push_back(concat);
    bindData(concat, layerOutput);
    _logger->debug("parsed to mcmModel as '%s", mvConcat->getName());
}

void FrontEndMcm::parseArgMax(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing ArgMax layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ArgMax layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseGRN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing GRN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "GRN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMVN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing MVN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "MVN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePower(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Power layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Power layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDetectionOutput(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing DetectionOutput layer %s", layer->name);
    VPU_THROW_EXCEPTION << "DetectionOutput layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseSigmoid(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Sigmoid layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Sigmoid layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTanH(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing TanH layer %s", layer->name);
    VPU_THROW_EXCEPTION << "TanH layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePReLU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing PReLU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseBatchNorm(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing PReLU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDeconvolution(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    // TODO: Leyer can be with bias
    _logger->debug("Parsing Deconvolution( layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Deconvolution layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCopy(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Copy layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Copy layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseELU(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing ELU layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ELU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCrop(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Crop layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Crop layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTile(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Tile layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Tile layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseNormalize(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Normalize layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Normalize layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseRegionYolo(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing RegionYolo layer %s", layer->name);
    VPU_THROW_EXCEPTION << "RegionYolo layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseReorgYolo(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing ReorgYolo layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ReorgYolo layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCTCDecoder(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing CTCDecoder layer %s", layer->name);
    VPU_THROW_EXCEPTION << "CTCDecoder layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseInterp(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Interp layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Interp layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseProposal(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Proposal layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Proposal layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseROIPooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing ROIPooling layer %s", layer->name);
    VPU_THROW_EXCEPTION << "ROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePSROIPooling(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing PSROIPooling layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PSROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCustom(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Custom layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Custom layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMTCNN(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing MTCNN layer %s", layer->name);
    VPU_THROW_EXCEPTION << "MTCNN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePad(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Pad layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Pad layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseResample(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Resample layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Resample layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseLSTMCell(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing LSTMCell layer %s", layer->name);
    VPU_THROW_EXCEPTION << "LSTMCell layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBox(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing PriorBox layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PriorBox layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBoxClustered(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing PriorBoxClustered layer %s", layer->name);
    VPU_THROW_EXCEPTION << "PriorBoxClustered layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseSplit(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(modelMcm);
    UNUSED(inputs);
    _logger->debug("Parsing Split layer %s", layer->name);
    VPU_THROW_EXCEPTION << "Split layer is not supported by kmbPlugin";
}

void FrontEndMcm::checkNetwork(const ie::CNNNetwork& network) {
    auto networkPrecision = network.getPrecision();
    if (networkPrecision != ie::Precision::FP16) {
        VPU_THROW_EXCEPTION << "Unsupported network precision : " << networkPrecision;
    }

    _parsedNetwork.networkInputs = network.getInputsInfo();
    _parsedNetwork.networkOutputs = network.getOutputsInfo();

    if (_parsedNetwork.networkInputs.empty()) {
        VPU_THROW_EXCEPTION << "No inputs detected in network " << network.getName();
    }
    if (_parsedNetwork.networkOutputs.empty()) {
        VPU_THROW_EXCEPTION << "No outputs detected in network " << network.getName();
    }

    for (const auto& netInput : _parsedNetwork.networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputPrecision = inputInfo->getPrecision();

        if (inputPrecision != ie::Precision::U8 &&
            inputPrecision != ie::Precision::FP16 &&
            inputPrecision != ie::Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported input precision: " << inputPrecision.name() << "!";
        }
    }

    for (const auto& netOutput : _parsedNetwork.networkOutputs) {
        auto outputData = netOutput.second;
        IE_ASSERT(outputData != nullptr);

        auto outputPrecision = outputData->getPrecision();

        if (outputPrecision != ie::Precision::FP16 &&
            outputPrecision != ie::Precision::FP32) {
            THROW_IE_EXCEPTION << "[PARAMETER_MISMATCH] Unsupported output precision: " << outputPrecision.name() << "!";
        }
    }
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
