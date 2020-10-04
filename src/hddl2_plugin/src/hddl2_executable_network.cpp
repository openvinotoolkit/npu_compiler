//
// Copyright 2019-2020 Intel Corporation.
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

// System
#include <fstream>
#include <memory>
#include <string>
#include <vector>
// IE
#include <net_pass.h>

#include <convert_function_to_cnn_network.hpp>
#include <generic_ie.hpp>
#include <ngraph/pass/manager.hpp>
#include <threading/ie_executor_manager.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_quantize_dequantize.hpp>
// Plugin
#include "hddl2_async_infer_request.h"
#include "hddl2_exceptions.h"
#include "hddl2_executable_network.h"
#include "hddl2_infer_request.h"
#include "hddl2_metrics.h"
// Subplugin
#include "subplugin/hddl2_device.h"
#include "vpux.hpp"
#include "vpux_compiler.hpp"

namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static vpux::Executor::Ptr createExecutor(const vpux::NetworkDescription::Ptr& network, const vpu::HDDL2Config& config,
    std::shared_ptr<vpux::IDevice>& device) {
    if (network == nullptr) {
        THROW_IE_EXCEPTION << "Network is null!";
    }
    // Default executor is nullptr, allow only perform export
    vpux::Executor::Ptr executor = nullptr;
    if (device != nullptr) {
        executor = device->createExecutor(network, config);
    }
    return executor;
}

static vpux::Executor::Ptr getExecutorForInference(const vpux::Executor::Ptr& executor) {
    if (executor == nullptr) {
        THROW_IE_EXCEPTION << NO_EXECUTOR_FOR_INFERENCE;
    }
    return executor->clone();
}
//------------------------------------------------------------------------------
//      Shared init ctor
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const vpu::HDDL2Config& config)
    : _config(config),
      _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())),
      _compiler(vpux::Compiler::create(vpux::CompilerType::MCMCompiler)) {}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    IE::ICNNNetwork& network, std::shared_ptr<vpux::IDevice>& device, const HDDL2Config& config)
    : ExecutableNetwork(config) {
    // FIXME: This is a copy-paste from kmb_executable_network.cpp
    // should be fixed after switching to VPUX completely
    const bool kmb_use_ngraph = (NULL != getenv("KMB_USE_NGRAPH_PARSER"));
    if (kmb_use_ngraph || _config.useNGraphParser()) {
        if (const auto func = network.getFunction()) {
            _logger->info("Using NGraph parser");
            IE::InputsDataMap inputsInfo;
            network.getInputsInfo(inputsInfo);

            IE::OutputsDataMap outputsInfo;
            network.getOutputsInfo(outputsInfo);

            _networkPtr = _compiler->compile(func, network.getName(), inputsInfo, outputsInfo, _config);
        } else {
            _logger->warning("Failed to read NGraph network");
            throw std::runtime_error("Failed to read NGraph network");
        }
    } else {
        _logger->info("NGraph parser disabled");
        _logger->info("Using CNNNetwork parser");
        // HACK: convert nGraph to old CNNNetwork to fix LP transformations

        std::shared_ptr<IE::ICNNNetwork> convertedNetwork;
        auto actualNetwork = &network;

        if (network.getFunction()) {
            auto nGraphFunc = network.getFunction();
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
            manager.run_passes(nGraphFunc);
            // Disable shape inference (WA for generic operations)
            ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

            // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
            ngraph::pass::ConvertPriorBox().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
            convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network, true);
            actualNetwork = convertedNetwork.get();
        }

        _networkPtr = _compiler->compile(*actualNetwork, _config);
    }
    _executorPtr = createExecutor(_networkPtr, config, device);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    std::istream& networkModel, std::shared_ptr<vpux::IDevice>& device, const vpu::HDDL2Config& config)
    : ExecutableNetwork(config) {
    _networkPtr = _compiler->parse(networkModel, _config);
    _executorPtr = createExecutor(_networkPtr, config, device);
    _networkInputs = vpux::helpers::dataMapIntoInputsDataMap(_networkPtr->getInputsInfo());
    _networkOutputs = vpux::helpers::dataMapIntoOutputsDataMap(_networkPtr->getOutputsInfo());
}

//------------------------------------------------------------------------------
//      Create infer requests
//------------------------------------------------------------------------------
IE::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    const IE::InputsDataMap networkInputs, const IE::OutputsDataMap networkOutputs) {
    auto inferExecutor = getExecutorForInference(_executorPtr);
    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs, inferExecutor, _config);
}

void ExecutableNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) {
    auto inferExecutor = getExecutorForInference(_executorPtr);
    auto syncRequestImpl = std::make_shared<HDDL2InferRequest>(_networkInputs, _networkOutputs, inferExecutor, _config);

    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    const std::string resultExecutorName = "HDDL2ResultExecutor";
    auto resultExecutor = IE::ExecutorManager::getInstance()->getExecutor(resultExecutorName);

    auto asyncThreadSafeImpl =
        std::make_shared<HDDL2AsyncInferRequest>(syncRequestImpl, _taskExecutor, resultExecutor, _callbackExecutor);
    asyncRequest.reset(
        new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(asyncThreadSafeImpl),
        [](InferenceEngine::IInferRequest* p) {
            p->Release();
        });
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}

//------------------------------------------------------------------------------
//      Export
//------------------------------------------------------------------------------
void ExecutableNetwork::ExportImpl(std::ostream& model) {
    auto graphBlob = _networkPtr->getCompiledNetwork();
    model.write(graphBlob.data(), graphBlob.size());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    std::ofstream modelFile(modelFileName, std::ios::binary);

    if (modelFile.is_open()) {
        ExportImpl(modelFile);
    } else {
        THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export.";
    }
}

}  // namespace HDDL2Plugin
}  // namespace vpu
