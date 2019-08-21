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

std::unordered_map<int, char> DIM_NAMES({
    {3, 'W'},
    {2, 'H'},
    {1, 'C'},
    {0, 'N'}
});

constexpr char FINISH_PARSING_STR[] = "Parsed to mcmModel as '%s";

void logParsingStartHelper(Logger::Ptr logger, const ie::CNNLayerPtr& layer, const McmNodeVector& inputs) {
    logger->debug("Start parsing '%s' layer: '%s'", layer->type, layer->name);

    if (inputs.empty()) {
        logger->debug("Layer has no input");
    } else {
        for (size_t i = 0; i < inputs.size(); ++i)
            logger->debug("Layer input %d: '%s'", i, inputs[0]->getMcmNode()->getName());
    }
}

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

namespace {
void getOutputScale(const ie::CNNLayerPtr& layer, mv::QuantizationParams &quantParams,
        const Logger::Ptr& logger) {  // TODO: Refactor logging: JIRA: CVS-21492
    std::vector<double> oiScaleData, wScaleData;
    IE_ASSERT(layer->blobs["weights"] != nullptr);

// TODO: Check correctness of weights presicion
// IE_ASSERT(layer->blobs["weights"]->getTensorDesc().getPrecision() == ie::Precision::I8);
    if (layer->blobs["weights"]->getTensorDesc().getPrecision() == ie::Precision::I8) {
        logger->warning("Weights of uantized layer %s have %d precision (!= ie::Precision::I8)' ",
                layer->name, layer->blobs["weights"]->getTensorDesc().getPrecision());
    }
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
}  // namespace

void FrontEndMcm::parseInputData() {
    _logger->debug("Try to parse network input");

    IE_ASSERT(_parsedNetwork.networkInputs.size() == 1);

    for (const auto& inputInfo : _parsedNetwork.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& dataDesc = ieData->getTensorDesc();
        mv::Shape inputShape(getWHCN(dataDesc).getDims());
        auto mvInput = _modelMcm.input(inputShape, convert_data_type(dataDesc.getPrecision()), mv::Order("NCHW"), {{}, {}, {}, {}}, netInput->name());
        bindOutput(mvInput, ieData);
        _logger->debug("Network input '%s'(orig: '%s') parsed to mcmModel", mvInput->getName(), netInput->name());
    }

#if 0  // TODO: rewrite logic to mcm compiler if this part is needed
    //
// Parse constant data
//

    auto kmbData = model->addInputData(ieData->getName(), kmbDesc);
    bindData(kmbData, ieData);

    auto kmbData = model->addConstData(
        ieData->getName(),
        kmbDesc,
        ieBlobContent(ieBlob));

    // User might ask to return the output from Const layer.
    if (auto kmbOutData = getVpuData(ieData)) {
        IE_ASSERT(kmbOutData->usage() == DataUsage::Output);

        _stageBuilder->addCopyStage(
            model,
            formatString("%s@return-const", kmbData->name()),
            nullptr,
            kmbData,
            kmbOutData);
    }

    bindData(kmbData, ieData);
#endif
}

void FrontEndMcm::parseOutputData() {
    _logger->debug("Try to parse network output");

    for (const auto& outputInfo : _parsedNetwork.networkOutputs) {
        auto ieData = outputInfo.second;

        IE_ASSERT(ieData != nullptr);

        auto lastLayerOut = getMcmData(ieData);
        IE_ASSERT(lastLayerOut != nullptr);
        auto name = lastLayerOut->getMcmNode()->getName();

        auto mvOutput = _modelMcm.output(lastLayerOut->getMcmNode());
        _output = std::make_shared<McmNodeObject>(mvOutput, lastLayerOut->desc());
        _nodes.push_back(_output);
    }
}

void FrontEndMcm::parseConvolution(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    bool is_quantized = false;
    bool with_bias = false;
    //
    // Extract parameters
    //
    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

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
        getOutputScale(layer, outputQuantParams, _logger);
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
                    mv::DType("Int32"), mv::Order::getColMajorID(1));
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
                                         outputQuantParams,
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
                    mv::DType("Int32"), mv::Order::getColMajorID(1));
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
                                outputQuantParams,
                                convLayer->name);
    }
    if (is_quantized) {
        mvConv->set<mv::QuantizationParams>("quantParams", outputQuantParams);
    }

    if (with_bias) {
        mvConvOnly = mvConv;
        mvConv = _modelMcm.bias(mvConvOnly, mvBiases, outputQuantParams, convLayer->name + ":bias");
        _logger->debug("'%s' layer '%s': Bias part (%s) added to mcmModel", convLayer->type, convLayer->name, mvConv->getName());
    }

    mvConv->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    bindOutput(mvConv, layerOutput);
    _logger->debug(FINISH_PARSING_STR, mvConv->getName());
}

void FrontEndMcm::parsePooling(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto input = inputs[0];
    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

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

    bindOutput(mvPooling, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvPooling->getName());
}

void FrontEndMcm::parseFullyConnected(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto FClayer = std::dynamic_pointer_cast<ie::FullyConnectedLayer>(layer);
    IE_ASSERT(layer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

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
        getOutputScale(layer, outputQuantParams, _logger);
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
                mv::DType("Int32"), mv::Order::getColMajorID(1));
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

    auto mvFullyConnected = _modelMcm.fullyConnected(input->getMcmNode(), mvWeights, outputQuantParams, FClayer->name);

    if (is_quantized) {
        mvFullyConnected->set<mv::QuantizationParams>("quantParams", outputQuantParams);
    }

    if (with_bias) {
        auto mvFCOnly = mvFullyConnected;
        mvFullyConnected = _modelMcm.bias(mvFCOnly, mvBiases, outputQuantParams, FClayer->name + ":bias");
        _logger->debug("'%s' layer '%s': Bias part (%s) added to mcmModel", FClayer->type, FClayer->name, mvFullyConnected->getName());
    }

    mvFullyConnected->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    bindOutput(mvFullyConnected, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvFullyConnected->getName());
}

void FrontEndMcm::parseReLU(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto reluLayer = std::dynamic_pointer_cast<ie::ReLULayer>(layer);
    IE_ASSERT(reluLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

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

    bindOutput(mvRelu, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvRelu->getName());
}

void FrontEndMcm::parseSoftMax(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto softMaxLayer = std::dynamic_pointer_cast<ie::SoftMaxLayer>(layer);
    IE_ASSERT(softMaxLayer != nullptr);

    IE_ASSERT(static_cast<size_t>(softMaxLayer->axis) < inputs[0]->desc().getDims().size());

    logParsingStartHelper(_logger, layer, inputs);

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[softMaxLayer->axis];
    auto mvSoftmax = _modelMcm.softmax(inputs[0]->getMcmNode(), mcmAxis, {{}, {}, {}, {}}, softMaxLayer->name);

    bindOutput(mvSoftmax, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvSoftmax->getName());
}

void FrontEndMcm::parseNorm(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto normLayer = std::dynamic_pointer_cast<ie::NormLayer>(layer);
    IE_ASSERT(normLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    auto mvLRN = _modelMcm.localResponseNormalization(inputs[0]->getMcmNode(),
            normLayer->_size, normLayer->_k, normLayer->name);

    bindOutput(mvLRN, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvLRN->getName());

    // TODO: add parsing following parameters
    // stage->attrs().set<float>("alpha", layer->_alpha);
    // stage->attrs().set<float>("beta", layer->_beta);
}

void FrontEndMcm::parseScale(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto scaleLayer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(layer);
    IE_ASSERT(scaleLayer != nullptr);
    IE_ASSERT(scaleLayer->_weights != nullptr);

    if (scaleLayer->_broadcast != 0) {
        VPU_THROW_EXCEPTION <<
            "Layer " << scaleLayer->name << " doesn't support broadcast param";
    }

    logParsingStartHelper(_logger, layer, inputs);

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

    auto mvScale = _modelMcm.scale(input->getMcmNode(), mvWeights, {{}, {}, {}, {}}, scaleLayer->name);
    auto mvScaleShift = mvScale;

    _logger->debug("'%s' layer '%s': Scale part (%s) added to mcmModel", scaleLayer->type, scaleLayer->name, mvScaleShift->getName());

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
        _logger->debug("'%s' layer '%s': Bias part (%s) added to mcmModel", scaleLayer->type, scaleLayer->name, mvScaleShift->getName());
    }

    bindOutput(mvScaleShift, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvScaleShift->getName());
}

void FrontEndMcm::parsePermute(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto ieOrder = layer->GetParamAsInts("order");

    std::string newOrder;

//  4d NCHW inputs are supported
    for (size_t i = 0; i < ieOrder.size(); i++) {
        newOrder += DIM_NAMES[ieOrder[ieOrder.size() - 1 - i]];
    }

    auto mvPerm = _modelMcm.permute(inputs[0]->getMcmNode(), mv::Order(newOrder), layer->name);
    bindOutput(mvPerm, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvPerm->getName());
}

void FrontEndMcm::parseEltwise(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    auto eltwiseLayer = std::dynamic_pointer_cast<ie::EltwiseLayer>(layer);
    IE_ASSERT(eltwiseLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    // Quantization parameters
    mv::QuantizationParams secondInputQuantParams  = {{}, {}, {}, {}};
    mv::QuantizationParams outputQuantParams = {{}, {}, {}, {}};

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

        const double inf = std::numeric_limits<double>::infinity();
        outputQuantParams   = {{0}, {1}, {-inf}, {inf}};
        // TODO: insert DW convolution or ScaleShift before non conv input with eltwise-sum-scale
        secondInputQuantParams  = createQuantParams(layer, "eltwise-sum-scale");
    }

    mv::Data::TensorIterator mvEltwise;
    std::vector< mv::Data::TensorIterator > mvInputs;
    for (const auto& input : inputs) {
        mvInputs.push_back(input->getMcmNode());
    }

    auto addCoefficient0 = 1.0f;
    auto addCoefficient1 = 1.0f;
    switch (eltwiseLayer->_operation) {
    case ie::EltwiseLayer::eOperation::Sub:
        if (inputs.size() > 2) {
            VPU_THROW_EXCEPTION << eltwiseLayer->name <<
                    "Eltwise Sub operations with with more than 2 operands is not supported by kmbPlugin";
        }
        addCoefficient1 = -1.0f;
        // fall through
    case ie::EltwiseLayer::eOperation::Sum:
        for (size_t i = 0; i < eltwiseLayer->coeff.size(); ++i) {
            if (std::abs(eltwiseLayer->coeff[i]) != 1.0f) {
                VPU_THROW_EXCEPTION << eltwiseLayer->name <<
                        " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
            }
        }
        if (inputs.size() == 2) {
            if (eltwiseLayer->coeff.size() > 0) {
                addCoefficient0 *= eltwiseLayer->coeff[0];
            }
            if (eltwiseLayer->coeff.size() > 1) {
                addCoefficient1 *= eltwiseLayer->coeff[1];
            }
            if (addCoefficient0 == 1.0f && addCoefficient1 == 1.0f) {
                mvEltwise = _modelMcm.add(mvInputs, secondInputQuantParams, eltwiseLayer->name);
            } else if (addCoefficient0 * addCoefficient1 < 0.f) {
                if (addCoefficient0 == 1.0f && addCoefficient1 == -1.0f) {
                    mvEltwise = _modelMcm.subtract(mvInputs, secondInputQuantParams, eltwiseLayer->name);
                } else {
                    mvEltwise = _modelMcm.subtract(mvInputs, secondInputQuantParams, eltwiseLayer->name);
                }
            } else {
                VPU_THROW_EXCEPTION << eltwiseLayer->name <<
                        " Eltwise Sum/Sub operations with such coefficients is not supported by kmbPlugin";
            }
        } else {
            mvEltwise = _modelMcm.add(mvInputs, secondInputQuantParams, eltwiseLayer->name);
        }

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
        break;
    default:
        VPU_THROW_EXCEPTION << "Eltwise operation" << eltwiseLayer->_operation << " is not supported";
    }

    mvEltwise->set<mv::DType>("dType", convert_data_type(layer->outData[0]->getPrecision()));

    bindOutput(mvEltwise, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvEltwise->getName());
}

void FrontEndMcm::parseBias(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    mv::Data::TensorIterator mvBias;
    if (inputs.size() == 1) {
        logParsingStartHelper(_logger, layer, inputs);

        auto input = inputs[0];
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
        logParsingStartHelper(_logger, layer, inputs);

        auto input = inputs[0];
        auto input1 = inputs[1];
        mvBias = _modelMcm.bias(input->getMcmNode(), input1->getMcmNode(), {{}, {}, {}, {}}, layer->name);
    } else {
        VPU_THROW_EXCEPTION << "Bias layer does not support " << inputs.size() << " inputs";
    }

    bindOutput(mvBias, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvBias->getName());
}

void FrontEndMcm::parseClamp(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    auto clampLayer = std::dynamic_pointer_cast<ie::ClampLayer>(layer);
    IE_ASSERT(clampLayer != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    auto mvClamp = _modelMcm.clamp(inputs[0]->getMcmNode(), clampLayer->min_value, clampLayer->max_value, clampLayer->name);
    bindOutput(mvClamp, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvClamp->getName());
}

void FrontEndMcm::parseReshape(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    auto layerOutput = layer->outData[0];
    IE_ASSERT(layerOutput != nullptr);

    logParsingStartHelper(_logger, layer, inputs);

    // Because mcmCompiler supports only "dense" layouts
    // for example NC should be represented as NCHW with dims NC11
    // Formation of a newShape, "dense" shape with 1, substituted in the places of non-existent measurements
    // TODO: Tests on parsing/compilation of different cases of reshape should be added: Jira: CVS-20409
    // McmCompiler accept only input in WHCN format
    mv::Shape newShape(getWHCN(layerOutput->getTensorDesc()).getDims());

    auto mvReshape = _modelMcm.reshape(inputs[0]->getMcmNode(), newShape, "", layer->name);
    bindOutput(mvReshape, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvReshape->getName());
}

void FrontEndMcm::parseConcat(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(!inputs.empty());

    auto clampLayer = std::dynamic_pointer_cast<ie::ConcatLayer>(layer);
    IE_ASSERT(clampLayer != nullptr);
    IE_ASSERT(clampLayer->_axis < inputs[0]->desc().getDims().size());

    logParsingStartHelper(_logger, layer, inputs);

    std::string mcmAxis;
    mcmAxis = mcmAxis + DIM_NAMES[clampLayer->_axis];
    std::vector<mv::Data::TensorIterator> concatInputs;

    for (const auto & input : inputs) {
        concatInputs.push_back(input->getMcmNode());
    }

    auto mvConcat = _modelMcm.concat(concatInputs, mcmAxis, {{}, {}, {}, {}}, clampLayer->name + ":step0");
    bindOutput(mvConcat, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, mvConcat->getName());
}

void FrontEndMcm::parseRegionYolo(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto coords = layer->GetParamAsUInt("coords");
    auto classes = layer->GetParamAsUInt("classes");
    auto do_softmax = layer->GetParamAsBool("do_softmax");
    auto num = layer->GetParamAsUInt("num");

    auto region = _modelMcm.regionYolo(inputs[0]->getMcmNode(), coords, classes, do_softmax, num, {}, layer->name);
    bindOutput(region, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, region->getName());
}

void FrontEndMcm::parseReorgYolo(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    IE_ASSERT(inputs.size() == 1);

    logParsingStartHelper(_logger, layer, inputs);

    auto stride = layer->GetParamAsUInt("stride");

    auto reorg = _modelMcm.reorgYolo(inputs[0]->getMcmNode(), stride, layer->name);
    bindOutput(reorg, layer->outData[0]);

    _logger->debug(FINISH_PARSING_STR, reorg->getName());
}

void FrontEndMcm::parseArgMax(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ArgMax layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseGRN(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "GRN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMVN(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "MVN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePower(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Power layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDetectionOutput(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "DetectionOutput layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseSigmoid(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Sigmoid layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTanH(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "TanH layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePReLU(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseBatchNorm(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PReLU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseDeconvolution(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    // TODO: Leyer can be with bias
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Deconvolution layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCopy(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Copy layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseELU(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ELU layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCrop(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Crop layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseTile(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Tile layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseNormalize(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Normalize layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCTCDecoder(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "CTCDecoder layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseInterp(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Interp layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseProposal(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Proposal layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseROIPooling(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "ROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePSROIPooling(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PSROIPooling layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseCustom(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Custom layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseMTCNN(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "MTCNN layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePad(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Pad layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseResample(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Resample layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseLSTMCell(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "LSTMCell layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBox(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PriorBox layer is not supported by kmbPlugin";
}

void FrontEndMcm::parsePriorBoxClustered(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "PriorBoxClustered layer is not supported by kmbPlugin";
}

void FrontEndMcm::parseSplit(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs) {
    UNUSED(inputs);
    UNUSED(layer);
    VPU_THROW_EXCEPTION << "Split layer is not supported by kmbPlugin";
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
