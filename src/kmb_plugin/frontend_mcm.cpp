//
// Copyright 2018 Intel Corporation.
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

#include <memory>
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <graph_tools.hpp>
#include <ie_profiling.hpp>

#include <utils/dims_parser.hpp>
#include "low_precision_transformations/transformer.hpp"
#include <ie_util_internal.hpp>
#include <graph_transformer.h>


#ifdef ENABLE_MCM_COMPILER
namespace vpu {

namespace KmbPlugin {

namespace {

typedef void (FrontEndMcm::*parser_t)(
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs);

ie::details::caseless_map<std::string, parser_t> g_mcm_parsers = {
    {"Convolution",        &FrontEndMcm::parseConvolution},
    {"Pooling",            &FrontEndMcm::parsePooling},
    {"ReLU",               &FrontEndMcm::parseReLU},
    {"Clamp",              &FrontEndMcm::parseClamp},
    {"FullyConnected",     &FrontEndMcm::parseFullyConnected},
    {"SoftMax",            &FrontEndMcm::parseSoftMax},
    {"GRN",                &FrontEndMcm::parseGRN},
    {"MVN",                &FrontEndMcm::parseMVN},
    {"Norm",               &FrontEndMcm::parseNorm},
    {"Concat",             &FrontEndMcm::parseConcat},
    {"Eltwise",            &FrontEndMcm::parseEltwise},
    {"Split",              &FrontEndMcm::parseSplit},
    {"Sigmoid",            &FrontEndMcm::parseSigmoid},
    {"TanH",               &FrontEndMcm::parseTanH},
    {"PReLU",              &FrontEndMcm::parsePReLU},
    {"Bias",               &FrontEndMcm::parseBias},
    // Caffe Slice is transformed to Split by IE
    {"Slice",              &FrontEndMcm::parseSplit},
    {"BatchNormalization", &FrontEndMcm::parseBatchNorm},
    {"ScaleShift",         &FrontEndMcm::parseScale},
    {"Deconvolution",      &FrontEndMcm::parseDeconvolution},
    {"Power",              &FrontEndMcm::parsePower},
    {"Copy",               &FrontEndMcm::parseCopy},
    {"Reshape",            &FrontEndMcm::parseReshape},
    {"Squeeze",            &FrontEndMcm::parseReshape},
    {"Unsqueeze",          &FrontEndMcm::parseReshape},
    {"ELU",                &FrontEndMcm::parseELU},
    // Flatten is represented as Reshape in KMB model
    {"Flatten",            &FrontEndMcm::parseReshape},
    {"Crop",               &FrontEndMcm::parseCrop},
    {"Tile",               &FrontEndMcm::parseTile},
    {"Normalize",          &FrontEndMcm::parseNormalize},
    {"PriorBox",           &FrontEndMcm::parsePriorBox},
    {"PriorBoxClustered",  &FrontEndMcm::parsePriorBoxClustered},
    {"Permute",            &FrontEndMcm::parsePermute},
    {"DetectionOutput",    &FrontEndMcm::parseDetectionOutput},
    {"RegionYolo",         &FrontEndMcm::parseRegionYolo},
    {"ReorgYolo",          &FrontEndMcm::parseReorgYolo},
    {"CTCGreedyDecoder",   &FrontEndMcm::parseCTCDecoder},
    {"Proposal",           &FrontEndMcm::parseProposal},
    {"ROIPooling",         &FrontEndMcm::parseROIPooling},
    {"PSROIPooling",       &FrontEndMcm::parsePSROIPooling},
    {"Interp",             &FrontEndMcm::parseInterp},
    {"Custom",             &FrontEndMcm::parseCustom},
    {"MTCNN",              &FrontEndMcm::parseMTCNN},
    {"LSTMCell",           &FrontEndMcm::parseLSTMCell},
    {"Pad",                &FrontEndMcm::parsePad},
    {"Resample",           &FrontEndMcm::parseResample},
    {"ArgMax",             &FrontEndMcm::parseArgMax},
};

}  // namespace

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

void FrontEndMcm::buildInitialModel(ie::ICNNNetwork& network) {
    runCommonPasses(network);

    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        _logger->debug("Try to parse layer %s", layer->name);

        McmNodeVector inputs;
        if (layer->type == "FakeQuantize") {
            continue;
        }

        if (layer->type == "ScaleShift") {
            auto layerInput = layer->insData[0].lock();
            IE_ASSERT(layerInput != nullptr);

            auto prevLayer = layerInput->getCreatorLayer().lock();
            if (prevLayer != nullptr) {
                if (prevLayer->type == "Const") {
                    continue;
                }
            }
        }

        getInputData(layer, inputs);
        auto it = g_mcm_parsers.find(layer->type);
        if (it == g_mcm_parsers.end()) {
            VPU_THROW_EXCEPTION
                    << "Cannot convert layer \""
                    << layer->name
                    << "\" due to unsupported layer type \""
                    << layer->type
                    << "\"";
        }

        auto parser = it->second;
        IE_ASSERT(parser != nullptr);

        (this->*parser)(layer, inputs);
    }

    parseOutputData();
}

std::set<std::string> FrontEndMcm::checkSupportedLayers(ie::ICNNNetwork& network) {
    runCommonPasses(network);

    std::set<std::string> layerNames;

    for (const auto& layer : _parsedNetwork.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        _logger->debug("Try to parse layer %s", layer->name);

        McmNodeVector inputs;
        getInputData(layer, inputs);

        auto it = g_mcm_parsers.find(layer->type);
        if (it != g_mcm_parsers.end()) {
            try {
                // If we can create and have not thrown exception, then layer is supported.
                auto parser = it->second;
                IE_ASSERT(parser != nullptr);

                (this->*parser)(layer, inputs);

                layerNames.insert(layer->name);
            } catch (const ie::details::InferenceEngineException&) {
                // TODO: continue instead?
                break;
            }
        }
    }

    return layerNames;
}

void FrontEndMcm::parseNetworkDFS(const ie::ICNNNetwork& network, ParsedNetwork& parsedNetwork) {
    IE_PROFILING_AUTO_SCOPE(parseNetworkDFS);

    ie::details::CaselessEq<std::string> cmp;

    //
    // Collect all network input data.
    //

    network.getInputsInfo(parsedNetwork.networkInputs);
    network.getOutputsInfo(parsedNetwork.networkOutputs);

    std::unordered_set<ie::DataPtr> allInputDatas;
    for (const auto& netInput : parsedNetwork.networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputData = inputInfo->getInputData();
        IE_ASSERT(inputData != nullptr);

        allInputDatas.insert(inputData);
    }

    //
    // Collect all network const data.
    //

    for (const auto& layer : ie::CNNNetGetAllInputLayers(network)) {
        IE_ASSERT(layer != nullptr);

        if (!cmp(layer->type, "Const"))
            continue;

        if (layer->outData.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of outputs "
                    << layer->outData.size();
        }

        if (layer->blobs.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of blobs "
                    << layer->blobs.size();
        }

        auto constData = layer->outData[0];
        IE_ASSERT(constData != nullptr);

        auto constBlob = layer->blobs.begin()->second;
        IE_ASSERT(constBlob != nullptr);

        parsedNetwork.constDatas[constData] = constBlob;

        allInputDatas.insert(constData);
    }

    //
    // Collect initial layers.
    //

    std::unordered_set<ie::CNNLayerPtr> visitedInitialLayers;
    SmallVector<ie::CNNLayerPtr> initialLayers;

    for (const auto& inputData : allInputDatas) {
        for (const auto& consumer : inputData->getInputTo()) {
            auto initialLayer = consumer.second;
            IE_ASSERT(initialLayer != nullptr);

            if (visitedInitialLayers.count(initialLayer) > 0)
                continue;

            bool allInputsAvailable = true;
            for (const auto& in : initialLayer->insData) {
                auto input = in.lock();
                IE_ASSERT(input != nullptr);

                if (allInputDatas.count(input) == 0) {
                    allInputsAvailable = false;
                    break;
                }
            }

            if (allInputsAvailable) {
                visitedInitialLayers.insert(initialLayer);
                initialLayers.emplace_back(std::move(initialLayer));
            }
        }
    }

    IE_ASSERT(!initialLayers.empty());

    //
    // Run recursive DFS algorithm.
    //

    std::sort(initialLayers.begin(), initialLayers.end(),
              [](const ie::CNNLayerPtr& left, const ie::CNNLayerPtr& right) {
                  ie::details::CaselessLess<std::string> cmp;
                  return cmp(left->name, right->name);
              });

    InferenceEngine::CNNNetForestDFS(initialLayers, [&parsedNetwork](const ie::CNNLayerPtr& layer) {
        parsedNetwork.orderedLayers.emplace_back(layer);
    }, false);

    std::reverse(parsedNetwork.orderedLayers.begin(), parsedNetwork.orderedLayers.end());
}

void FrontEndMcm::applyQuantizationTransformations(ie::CNNNetwork& network) {
    InferenceEngine::details::LowPrecisionTransformer transformer(InferenceEngine::details::LowPrecisionTransformer::getAllTransformations(true));
    transformer.transform(network);

    for (auto layer : network) {
        if (layer->type == "FakeQuantize") {
            auto quantizeLayer = std::dynamic_pointer_cast<InferenceEngine::QuantizeLayer>(layer);
            InferenceEngine::DataPtr layerInput = quantizeLayer->insData[0].lock();
            auto prevLayer = layerInput->getCreatorLayer().lock();
            if (prevLayer->type != "Convolution") {
                continue;
            }
            auto initAxisIdx = [&](size_t edgeIdx) {
                auto edge = quantizeLayer->insData[edgeIdx].lock();

                size_t axisIdx = 0;
                int numberOfNonUnit = 0;
                for (unsigned i = 0; i < edge->getDims().size(); i++) {
                    if (edge->getDims()[i] > 1) {
                        axisIdx = i;
                        numberOfNonUnit++;
                    }
                }
                if (numberOfNonUnit > 1) {
                    THROW_IE_EXCEPTION << "Quantize layer " << quantizeLayer->name
                                       << " supports only per-tensor and per-channel quantizations";
                }

                return axisIdx;
            };

            int levels = -1;

            size_t inputLowAxis = 0;
            size_t inputHighAxis = 0;
            size_t outputLowAxis = 0;
            size_t outputHighAxis = 0;

            inputLowAxis = initAxisIdx(1);
            inputHighAxis = initAxisIdx(2);
            outputLowAxis = initAxisIdx(3);
            outputHighAxis = initAxisIdx(4);

            size_t axis = 0;

            axis = std::max(std::max(std::max(inputLowAxis, inputHighAxis), outputLowAxis), outputHighAxis);

            if ((inputLowAxis != 0 && inputLowAxis != axis) ||
                (inputHighAxis != 0 && inputHighAxis != axis) ||
                (outputLowAxis != 0 && outputLowAxis != axis) ||
                (outputHighAxis != 0 && outputHighAxis != axis)) {
                THROW_IE_EXCEPTION << "Quantize layer " << layer->name << " has inconsistent input dims";
            }

            // WA to enable jit implementation
            if (axis == 0) {
                if (quantizeLayer->insData[1].lock()->getDims()[inputLowAxis] == 1 &&
                    quantizeLayer->insData[2].lock()->getDims()[inputHighAxis] == 1 &&
                    quantizeLayer->insData[3].lock()->getDims()[outputLowAxis] == 1 &&
                    quantizeLayer->insData[4].lock()->getDims()[outputHighAxis] == 1) {
                    axis = 1;
                }
            }

            size_t axisRealSize = layer->insData[0].lock()->getDims()[axis];

            InferenceEngine::DataPtr inputLow = layer->insData[1].lock();
            bool isInputLowBroadcasted = inputLow->getDims()[inputLowAxis] != axisRealSize;

            InferenceEngine::DataPtr inputHigh = layer->insData[2].lock();
            bool isInputHighBroadcasted = inputHigh->getDims()[inputHighAxis] != axisRealSize;

            InferenceEngine::DataPtr outputLow = layer->insData[3].lock();
            bool isOutputLowBroadcasted = outputLow->getDims()[outputLowAxis] != axisRealSize;

            InferenceEngine::DataPtr outputHigh = layer->insData[4].lock();
            bool isOutputHighBroadcasted = outputHigh->getDims()[outputHighAxis] != axisRealSize;

            auto outputHighCreator = outputHigh->getCreatorLayer().lock()->blobs["custom"];
            auto outputHighBlob = dynamic_cast<InferenceEngine::TBlob<float> *>(outputHighCreator.get());
            auto outputHighData = outputHighBlob->buffer().as<float *>();

            auto outputLowCreator = outputLow->getCreatorLayer().lock()->blobs["custom"];
            auto outputLowBlob = dynamic_cast<InferenceEngine::TBlob<float> *>(outputLowCreator.get());
            auto outputLowData = outputLowBlob->buffer().as<float *>();

            auto inputHighCreator = inputHigh->getCreatorLayer().lock()->blobs["custom"];
            auto inputHighBlob = dynamic_cast<InferenceEngine::TBlob<float> *>(inputHighCreator.get());
            auto inputHighData = inputHighBlob->buffer().as<float *>();

            auto inputLowCreator = inputLow->getCreatorLayer().lock()->blobs["custom"];
            auto inputLowBlob = dynamic_cast<InferenceEngine::TBlob<float> *>(inputLowCreator.get());
            auto inputLowData = inputLowBlob->buffer().as<float *>();

            auto dataDescFP = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                        {axisRealSize},
                                                        InferenceEngine::Layout::C);
            auto dataDescInt = InferenceEngine::TensorDesc(InferenceEngine::Precision::I32,
                                            {axisRealSize},
                                            InferenceEngine::Layout::C);

            auto newActivationInputScale = InferenceEngine::make_shared_blob<float>(dataDescFP);
            auto newActivationOutScale = InferenceEngine::make_shared_blob<float>(dataDescFP);
            auto newActivationInputShift = InferenceEngine::make_shared_blob<int32_t>(dataDescInt);
            auto newActivationOutShift = InferenceEngine::make_shared_blob<int32_t>(dataDescInt);

            newActivationInputScale->allocate();
            newActivationOutScale->allocate();
            newActivationOutShift->allocate();
            newActivationInputShift->allocate();

            auto inputScale = newActivationInputScale->buffer().as<float *>();
            auto outputScale = newActivationOutScale->buffer().as<float *>();
            auto inputShift = newActivationInputShift->buffer().as<int32_t *>();
            auto outputShift = newActivationOutShift->buffer().as<int32_t *>();

            for (unsigned i = 0; i < axisRealSize; i++) {
                float il = inputLowData[isInputLowBroadcasted ? 0 : i];
                float ih = inputHighData[isInputHighBroadcasted ? 0 : i];
                float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
                float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

                inputScale[i] = static_cast<float>((levels - 1) / (ih - il));
                inputShift[i] = static_cast<int32_t>(-il * (levels - 1) / (ih - il));
                outputScale[i] = static_cast<float>((oh - ol) / (levels - 1));
                outputShift[i] = static_cast<int32_t>(ol);
            }

            prevLayer->blobs["newActivationInputScale"] = newActivationInputScale;
            prevLayer->blobs["newActivationOutScale"] = newActivationOutScale;
            prevLayer->blobs["newActivationInputShift"] = newActivationInputShift;
            prevLayer->blobs["newActivationOutShift"] = newActivationOutShift;
        }
    }
}

void FrontEndMcm::runCommonPasses(ie::ICNNNetwork& network) {
    auto cnnNet = ie::CNNNetwork(std::shared_ptr<ie::ICNNNetwork>(&network, [](ie::ICNNNetwork*){}));
    applyQuantizationTransformations(cnnNet);
    parseNetworkDFS(cnnNet, _parsedNetwork);
    parseInputData();
}

McmNode FrontEndMcm::getMcmData(const ie::DataPtr& ieData) {
    IE_ASSERT(ieData != nullptr);
    auto it = _ieToMcmMap.find(ieData);
    if (it == _ieToMcmMap.end()) {
        return nullptr;
    }

    return it->second;
}

void FrontEndMcm::bindData(const McmNode& data, const ie::DataPtr& ieData) {
    IE_ASSERT(_modelMcm.isValid(data->getMcmNode()));
    IE_ASSERT(_modelMcm.isValid(_modelMcm.getSourceOp(data->getMcmNode())));
    _ieToMcmMap[ieData] = data;
    data->setOrigData(ieData);
}

void FrontEndMcm::bindOutput(mv::Data::TensorIterator node, ie::DataPtr& layerOutput) {
    IE_ASSERT(layerOutput != nullptr);
    auto layer = std::make_shared<McmNodeObject>(node, layerOutput->getTensorDesc());
    _nodes.push_back(layer);
    bindData(layer, layerOutput);
}

void FrontEndMcm::getInputData(
        const ie::CNNLayerPtr& layer,
        McmNodeVector& inputs) {
    IE_ASSERT(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);

        auto prevLayer = layerInput->getCreatorLayer().lock();
        if (prevLayer != nullptr) {
            // WA for ScaleShift on Weights, should be remove
            if ((prevLayer->type == "Const") || (layer->type == "Const") || (prevLayer->type == "ScaleShift" && i != 0)) {
                continue;
            }

            if (prevLayer->type == "FakeQuantize")  {
                auto prevLayerInput =  prevLayer->insData[0].lock();
                layerInput = prevLayerInput;
            }
        }

        inputs[i] = getMcmData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }
}

}  // namespace KmbPlugin

}  // namespace vpu
#endif
