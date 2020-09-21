//
// Copyright 2020 Intel Corporation.
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

#include <Inference.h>
#include <WorkloadContext.h>
#include <hddl2_exceptions.h>
#include <hddl_unite/hddl2_unite_graph.h>

#include <string>

#include "hddl2_metrics.h"

namespace vpu {
namespace HDDL2Plugin {
namespace IE = InferenceEngine;

static const HddlUnite::Device::Ptr getUniteDeviceByID(const std::string& deviceID) {
    if (deviceID.empty()) return nullptr;

    std::vector<HddlUnite::Device> cores;
    getAvailableDevices(cores);
    const auto deviceIt = std::find_if(cores.begin(), cores.end(), [&deviceID](const HddlUnite::Device& device) {
        return std::to_string(device.getSwDeviceId()) == deviceID;
    });
    if (deviceIt == cores.end()) {
        return nullptr;
    }
    return std::make_shared<HddlUnite::Device>(*deviceIt);
}

HddlUniteGraph::HddlUniteGraph(
    const vpux::NetworkDescription::CPtr& network, const std::string& deviceID,
        const std::unordered_map<std::string, std::string>& config, const vpu::LogLevel& logLevel)
    : _logger(std::make_shared<Logger>("Graph", logLevel, consoleOutput())) {
    if (!network) {
        throw std::invalid_argument("Network pointer is null!");
    }
    HddlStatusCode statusCode;

    const std::string graphName = network->getName();
    const std::vector<char> graphData = network->getCompiledNetwork();

    std::vector<HddlUnite::Device> devices_to_use = {};

    const HddlUnite::Device::Ptr core = getUniteDeviceByID(deviceID);
    if (core != nullptr) {
        devices_to_use.push_back(*core);
        _logger->info("Graph: %s to device id: %d | Device: %s ", graphName, core->getSwDeviceId(), core->getName());
    } else {
        _logger->info("All devices will be used.");
    }

    // TODO we need to get number of NN shaves and threads via config, not as parameters
    const int nnThreadNum = 1;
    const int nnShaveNum = 16;
    statusCode = HddlUnite::Inference::loadGraph(_uniteGraphPtr, graphName, graphData.data(), graphData.size(),
        devices_to_use, nnThreadNum, nnShaveNum, config);

    // FIXME This error handling part should be refactored according to new api
    if (statusCode == HddlStatusCode::HDDL_CONNECT_ERROR) {
        THROW_IE_EXCEPTION << IE::details::as_status << IE::StatusCode::NETWORK_NOT_LOADED;
    } else if (statusCode != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }

    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::HddlUniteGraph(const vpux::NetworkDescription::CPtr& network,
    const HDDL2RemoteContext::CPtr& contextPtr, const std::unordered_map<std::string, std::string>& config,
    const vpu::LogLevel& logLevel)
    : _logger(std::make_shared<Logger>("Graph", logLevel, consoleOutput())) {
    HddlStatusCode statusCode;
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Workload context is null";
    }

    const std::string graphName = network->getName();
    const std::vector<char> graphData = network->getCompiledNetwork();

    HddlUnite::WorkloadContext::Ptr workloadContext = contextPtr->getHddlUniteWorkloadContext();

    // TODO we need to get number of NN shaves and threads via config, not as parameters
    const int nnThreadNum = 1;
    const int nnShaveNum = 16;
    statusCode = HddlUnite::Inference::loadGraph(_uniteGraphPtr, graphName, graphData.data(), graphData.size(),
        {*workloadContext}, nnThreadNum, nnShaveNum, config);

    if (statusCode != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }
    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::~HddlUniteGraph() {
    if (_uniteGraphPtr != nullptr) {
        HddlUnite::Inference::unloadGraph(_uniteGraphPtr);
    }
}

void HddlUniteGraph::InferAsync(const HddlUniteInferData::Ptr& data) const {
    if (data == nullptr) {
        THROW_IE_EXCEPTION << "Data for inference is null!";
    }
    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << "Graph is null!";
    }

    HddlStatusCode inferStatus = HddlUnite::Inference::inferAsync(*_uniteGraphPtr, data->getHddlUniteInferData());

    if (inferStatus != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << "InferAsync FAILED! return code:" << inferStatus;
    }
}

}  // namespace HDDL2Plugin
}  // namespace vpu
