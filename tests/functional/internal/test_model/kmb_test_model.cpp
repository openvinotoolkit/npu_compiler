//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "kmb_test_model.hpp"
#include <file_utils.h>
#include "kmb_test_utils.hpp"

#include <blob_factory.hpp>
#include <queue>

namespace {

std::string createNameWithPort(const std::string& origName, const int& numberOfPorts, const int& portId) {
    std::string finalName = origName;
    if (numberOfPorts > 1) {
        finalName += "." + std::to_string(portId);
    }
    return finalName;
}

}  // namespace

TestNetwork::TestNetwork(const TestNetwork& other)
        : _refFuncs(other._refFuncs),
          _inputPrecisions(other._inputPrecisions),
          _inputLayouts(other._inputLayouts),
          _outputPrecisions(other._outputPrecisions),
          _outputLayouts(other._outputLayouts),
          _compileConfig(other._compileConfig) {
    std::vector<NodePtr> rootNodes;
    for (const auto& p : other._nodes) {
        const auto& node = p.second;
        if (node->get_users().empty()) {
            rootNodes.push_back(node);
        }
    }

    const auto sortedNodes = ngraph::topological_sort(rootNodes);

    for (const auto& node : sortedNodes) {
        const auto& name = node->get_friendly_name();
        IE_ASSERT(_nodes.find(name) == _nodes.end());

        OutputVector newInputs;
        for (const auto& input : node->inputs()) {
            const auto& port = input.get_source_output();
            const auto curInputNode = getLayer(port.get_node()->get_friendly_name());
            newInputs.emplace_back(curInputNode, port.get_index());
        }

        const auto newNode = node->copy_with_new_inputs(newInputs);
        newNode->set_friendly_name(name);

        _nodes.insert({name, newNode});
    }

    for (const auto& origParam : other._params) {
        const auto newParam =
                std::dynamic_pointer_cast<ngraph::op::Parameter>(getLayer(origParam->get_friendly_name()));
        IE_ASSERT(newParam != nullptr);
        _params.push_back(newParam);
    }

    for (const auto& origResult : other._results) {
        const auto newResult = std::dynamic_pointer_cast<ngraph::op::Result>(getLayer(origResult->get_friendly_name()));
        IE_ASSERT(newResult != nullptr);
        _results.push_back(newResult);
    }
}

TestNetwork& TestNetwork::operator=(const TestNetwork& other) {
    if (&other != this) {
        TestNetwork temp(other);
        swap(temp);
    }
    return *this;
}

void TestNetwork::swap(TestNetwork& other) {
    std::swap(_nodes, other._nodes);
    std::swap(_refFuncs, other._refFuncs);
    std::swap(_inputPrecisions, other._inputPrecisions);
    std::swap(_inputLayouts, other._inputLayouts);
    std::swap(_outputPrecisions, other._outputPrecisions);
    std::swap(_outputLayouts, other._outputLayouts);
    std::swap(_params, other._params);
    std::swap(_results, other._results);
    std::swap(_func, other._func);
    std::swap(_compileConfig, other._compileConfig);
}

TestNetwork& TestNetwork::addNetInput(const std::string& name, const SizeVector& dims, const Precision& precision) {
    IE_ASSERT(_func == nullptr);
    IE_ASSERT(_nodes.find(name) == _nodes.end());

    const auto type = precisionToType(precision);
    const auto shape = ngraph::Shape(dims);
    const auto inputNode = std::make_shared<ngraph::op::Parameter>(type, shape);
    inputNode->set_friendly_name(name);

    _nodes.insert({name, inputNode});
    _params.push_back(inputNode);

    return *this;
}

TestNetwork& TestNetwork::setUserInput(const std::string& name, const Precision& precision, const Layout& layout) {
    IE_ASSERT(_func == nullptr);

    _inputPrecisions[name] = precision;
    _inputLayouts[name] = layout;

    return *this;
}

TestNetwork& TestNetwork::addNetOutput(const PortInfo& port) {
    IE_ASSERT(_func == nullptr);

    const auto outputLayer = getLayer(port.layerName);

    const auto resultNode = std::make_shared<ngraph::op::Result>(Output(outputLayer, port.index));

    _results.push_back(resultNode);

    return *this;
}

TestNetwork& TestNetwork::setUserOutput(const PortInfo& port, const Precision& precision, const Layout& layout) {
    IE_ASSERT(_func == nullptr);

    auto outputNode = _nodes[port.layerName];
    auto outputCount = outputNode->outputs().size();
    auto finalPortName = createNameWithPort(port.layerName, outputCount, port.index);
    _outputPrecisions[finalPortName] = precision;
    _outputLayouts[finalPortName] = layout;

    return *this;
}

TestNetwork& TestNetwork::addConst(const std::string& name, const std::shared_ptr<ngraph::op::Constant>& node) {
    IE_ASSERT(_func == nullptr);
    IE_ASSERT(node != nullptr);

    const auto refFunc = [](const NodePtr& layer, const BlobVector& inputs, const TestNetwork&) -> BlobVector {
        IE_ASSERT(layer != nullptr);
        IE_ASSERT(inputs.empty());

        const auto constLayer = std::dynamic_pointer_cast<ngraph::op::Constant>(layer);
        IE_ASSERT(constLayer != nullptr);

        // TODO: use actual precision? What about unsupported cases?
        const auto blobDesc = TensorDesc(Precision::U8, layer->output(0).get_shape(),
                                         TensorDesc::getLayoutByDims(layer->output(0).get_shape()));
        const auto blob = make_blob_with_precision(blobDesc);
        blob->allocate();

        const auto blobPtr = blob->buffer().as<uint8_t*>();
        IE_ASSERT(blobPtr != nullptr);

        std::copy_n(constLayer->get_data_ptr<uint8_t>(), blob->byteSize(), blobPtr);

        return {blob};
    };

    return addLayer(name, node, refFunc);
}

TestNetwork& TestNetwork::addConst(const std::string& name, Blob::Ptr blob) {
    IE_ASSERT(_func == nullptr);
    IE_ASSERT(blob != nullptr);

    blob = vpux::toDefLayout(as<MemoryBlob>(blob));

    const auto type = precisionToType(blob->getTensorDesc().getPrecision());
    const auto shape = ngraph::Shape(blob->getTensorDesc().getDims());
    const auto node = std::make_shared<ngraph::op::Constant>(type, shape, blob->buffer());

    const auto refFunc = [blob](const NodePtr& layer, const BlobVector& inputs, const TestNetwork&) -> BlobVector {
        IE_ASSERT(layer != nullptr);
        IE_ASSERT(inputs.empty());

        return {vpux::toDefPrecision(as<MemoryBlob>(blob))};
    };

    return addLayer(name, node, refFunc);
}

TestNetwork& TestNetwork::addLayer(const std::string& name, const NodePtr& node, const RefFunc& refFunc) {
    IE_ASSERT(_func == nullptr);
    IE_ASSERT(_nodes.find(name) == _nodes.end());
    IE_ASSERT(node != nullptr);
    IE_ASSERT(refFunc != nullptr);

    node->set_friendly_name(name);
    _nodes.insert({name, node});
    _refFuncs.insert({name, refFunc});

    return *this;
}

CNNNetwork TestNetwork::getCNNNetwork() const {
    IE_ASSERT(_func != nullptr);

    const auto net = CNNNetwork(_func, _exts);

    auto inputsInfo = net.getInputsInfo();
    auto outputsInfo = net.getOutputsInfo();

    for (const auto& p : _inputPrecisions) {
        inputsInfo.at(p.first)->setPrecision(p.second);
    }
    for (const auto& p : _inputLayouts) {
        inputsInfo.at(p.first)->setLayout(p.second);
    }

    for (const auto& p : _outputPrecisions) {
        outputsInfo.at(p.first)->setPrecision(p.second);
    }
    for (const auto& p : _outputLayouts) {
        outputsInfo.at(p.first)->setLayout(p.second);
    }

    return net;
}

namespace {

struct CallNode final {
    TestNetwork::NodePtr op;
    TestNetwork::RefFunc refFunc;
    BlobVector outputs;
    std::vector<std::pair<std::shared_ptr<CallNode>, size_t>> deps;
};

}  // namespace

BlobMap TestNetwork::calcRef(const BlobMap& inputs) const {
    IE_ASSERT(_func != nullptr);

    //
    // Collect calls
    //

    std::queue<std::shared_ptr<CallNode>> callsQueue;
    std::unordered_map<std::shared_ptr<CallNode>, size_t> outputCalls;

    std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<CallNode>> callsMap;

    for (const auto& op : _func->get_ordered_ops()) {
        if (const auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(op)) {
            const auto& input = inputs.at(parameter->get_friendly_name());

            auto callNode = std::make_shared<CallNode>();
            callNode->op = op;
            callNode->outputs.push_back(vpux::toDefLayout(vpux::toDefPrecision(as<MemoryBlob>(input))));

            callsMap.insert({op, callNode});
            callsQueue.push(callNode);
        } else if (const auto result = std::dynamic_pointer_cast<ngraph::op::Result>(op)) {
            const auto prev = result->input(0).get_source_output();
            const auto prevCall = callsMap.at(prev.get_node_shared_ptr());
            outputCalls.insert({prevCall, prev.get_index()});
        } else {
            auto callNode = std::make_shared<CallNode>();
            callNode->op = op;
            callNode->refFunc = _refFuncs.at(op->get_friendly_name());
            for (const auto& opInput : op->inputs()) {
                const auto prev = opInput.get_source_output();
                const auto prevCall = callsMap.at(prev.get_node_shared_ptr());
                callNode->deps.emplace_back(prevCall, prev.get_index());
            }

            callsMap.insert({op, callNode});
            callsQueue.push(callNode);
        }
    }

    callsMap.clear();

    //
    // Run calls
    //

    while (!callsQueue.empty()) {
        const auto callNode = callsQueue.front();
        callsQueue.pop();

        if (callNode->refFunc == nullptr) {
            // Input call
            continue;
        }

        IE_ASSERT(callNode->outputs.empty());
        IE_ASSERT(callNode->op != nullptr);

        BlobVector inputs;

        for (const auto& dep : callNode->deps) {
            const auto& depCall = dep.first;
            IE_ASSERT(depCall != nullptr);

            const auto& depOutput = depCall->outputs.at(dep.second);
            IE_ASSERT(depOutput != nullptr);

            inputs.push_back(depOutput);
        }

        callNode->outputs = callNode->refFunc(callNode->op, inputs, *this);
    }

    //
    // Get outputs
    //

    BlobMap outputs;

    for (const auto& p : outputCalls) {
        const auto& callNode = p.first;
        IE_ASSERT(callNode != nullptr);
        IE_ASSERT(callNode->op != nullptr);
        const auto& output = callNode->outputs.at(p.second);
        IE_ASSERT(output != nullptr);
        auto outputName =
                createNameWithPort(callNode->op->get_friendly_name(), callNode->op->outputs().size(), p.second);

        outputs.insert({outputName, output});
    }

    return outputs;
}

std::vector<DataPtr> TestNetwork::getInputsInfo() const {
    std::vector<DataPtr> out;

    for (const auto& param : _params) {
        const auto& inputName = param->get_friendly_name();
        const auto info = param->output(0);

        const auto precisionIt = _inputPrecisions.find(inputName);
        const auto layoutIt = _inputLayouts.find(inputName);

        const auto precision =
                precisionIt != _inputPrecisions.end() ? precisionIt->second : typeToPrecision(info.get_element_type());
        const auto& dims = info.get_shape();
        const auto layout = layoutIt != _inputLayouts.end() ? layoutIt->second : TensorDesc::getLayoutByDims(dims);
        const auto desc = TensorDesc(precision, dims, layout);

        const auto data = std::make_shared<Data>(inputName, desc);
        out.push_back(data);
    }

    return out;
}

std::vector<DataPtr> TestNetwork::getOutputsInfo() const {
    std::vector<DataPtr> out;

    for (const auto& result : _results) {
        const auto prev = result->input(0).get_source_output();

        const auto& outputName = createNameWithPort(prev.get_node()->get_friendly_name(),
                                                    prev.get_node()->outputs().size(), prev.get_index());

        const auto info = prev.get_node()->output(prev.get_index());

        const auto precisionIt = _outputPrecisions.find(outputName);
        const auto layoutIt = _outputLayouts.find(outputName);

        const auto precision =
                precisionIt != _outputPrecisions.end() ? precisionIt->second : typeToPrecision(info.get_element_type());
        const auto& dims = info.get_shape();
        const auto layout = layoutIt != _outputLayouts.end() ? layoutIt->second : TensorDesc::getLayoutByDims(dims);
        const auto desc = TensorDesc(precision, dims, layout);

        const auto data = std::make_shared<Data>(outputName, desc);
        out.push_back(data);
    }

    return out;
}
