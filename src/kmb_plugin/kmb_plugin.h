//
// Copyright 2019 Intel Corporation.
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

#pragma once

// clang-format off
// Can get compile error, if the order of the headers will be changed.

#include <ie_metric_helpers.hpp>
#include "inference_engine.hpp"
#include "description_buffer.hpp"
#include "kmb_executable_network.h"
#include "kmb_metrics.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include "kmb_remote_context.h"
#include <vpux_compiler.hpp>

#include <vpux.hpp>
// clang-format on

namespace vpu {
namespace KmbPlugin {

class Engine final : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    using ie::InferencePluginInternal::ImportNetwork;

    InferenceEngine::ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                                     const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                         const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel, const RemoteContext::Ptr& context,
                                                         const std::map<std::string, std::string>& config) override;

    InferenceEngine::Parameter GetMetric(
            const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    RemoteContext::Ptr CreateContext(const ParamMap& map) override;

    static const char deviceName[];

private:
    RemoteContext::Ptr GetDefaultContext(const std::string& deviceId = "VPU-0");

    vpux::VPUXConfig _parsedConfig;
    std::mutex _contextCreateMutex;

    std::shared_ptr<vpux::EngineBackend> _backend;
    KmbMetrics _metrics;
    // map to cover the case when networks use different device IDs
    std::map<std::string, KmbRemoteContext::Ptr> _defaultContextMap;
    vpux::Compiler::Ptr _compiler;
};

}  // namespace KmbPlugin
}  // namespace vpu
