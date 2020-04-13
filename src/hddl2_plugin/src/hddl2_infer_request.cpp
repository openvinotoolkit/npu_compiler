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

#include "hddl2_infer_request.h"

#include <InferBlob.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8) {
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                           << "! Supported precisions: FP32, FP16, U8, I8";
    }
}

static InferenceEngine::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(tensorDesc);
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "InputBlob is nullptr.";
    }
    blob->allocate();
    return blob;
}

static void ensureBlobsExistForInputs(
    const InferenceEngine::InputsDataMap& inputData, const InferenceEngine::BlobMap& inputBlobs) {
    for (const auto& networkInput : inputData) {
        const std::string& inputName = networkInput.first;

        const auto& foundInputBlob = inputBlobs.find(inputName);
        if (foundInputBlob == inputBlobs.end()) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
        }
        if (foundInputBlob->second == nullptr) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is null.";
        }
    }
}

static void ensureBlobsExistForOutputs(
    const InferenceEngine::OutputsDataMap& outputData, const InferenceEngine::BlobMap& outputBlobs) {
    for (const auto& networkOutput : outputData) {
        const std::string& outputName = networkOutput.first;

        const auto& foundOutputBlob = outputBlobs.find(outputName);
        if (foundOutputBlob == outputBlobs.end()) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
        }
        if (foundOutputBlob->second == nullptr) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is null.";
        }
    }
}

HDDL2InferRequest::HDDL2InferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
    const HddlUniteGraph::Ptr& loadedGraph, const HDDL2RemoteContext::Ptr& context, const HDDL2Config& config)
    : InferRequestInternal(networkInputs, networkOutputs),
      _loadedGraphPtr(loadedGraph),
      _context(context),
      _config(config),
      _logger(std::make_shared<Logger>("HDDL2InferRequest", config.logLevel(), consoleOutput())) {
    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        _inputs[inputName] = allocateLocalBlob(inputTensorDesc);
    }

    for (auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        _outputs[outputName] = allocateLocalBlob(outputTensorDesc);
    }
}

void HDDL2InferRequest::InferImpl() {
    ensureBlobsExistForInputs(_networkInputs, _inputs);
    ensureBlobsExistForOutputs(_networkOutputs, _outputs);

    // TODO [Design flaw] InferData need to know if preprocessing required on creation.
    bool needPreProcessing = false;

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::Blob::Ptr inputBlobPtr = _inputs.find(inputName)->second;
        if (preProcessingRequired(networkInput.second, inputBlobPtr)) {
            needPreProcessing = true;
        }
    }

    _inferDataPtr = std::make_shared<HddlUniteInferData>(needPreProcessing, _context);

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::InputInfo::Ptr inputDesc = networkInput.second;

        // TODO [Design flaw] At this point we have input blob or preprocessing blob specified inside _preProcData
        if (_preProcData.find(inputName) != _preProcData.end()) {
            const IE::PreProcessDataPtr preprocessData = _preProcData.find(inputName)->second;
            const IE::Blob::Ptr blobForPreprocessing = preprocessData->getRoiBlob();

            _inferDataPtr->prepareUniteInput(blobForPreprocessing, inputDesc);
        } else {
            const IE::Blob::Ptr inputBlobPtr = _inputs.find(inputName)->second;
            _inferDataPtr->prepareUniteInput(inputBlobPtr, inputDesc);
        }
    }

    for (const auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::Blob::Ptr outputBlobPtr = _outputs.find(outputName)->second;

        _inferDataPtr->prepareUniteOutput(outputBlobPtr, networkOutput.second);
    }

    _loadedGraphPtr->InferSync(_inferDataPtr);
    GetResult();
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2InferRequest::GetResult() {
    if (_networkOutputs.size() != 1) {
        THROW_IE_EXCEPTION << "Only one output is supported!";
    }

    const std::string outputName = _networkOutputs.begin()->first;
    auto foundOutputBlob = _outputs.find(outputName);
    if (foundOutputBlob == _outputs.end()) {
        THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
    }
    const IE::Blob::Ptr outputBlobPtr = foundOutputBlob->second;

    const std::string outputUniteData = _inferDataPtr->getOutputData(outputName);

    const auto networkOutputPrecision = _networkOutputs.begin()->second->getPrecision();
    const auto blobOutputPrecision = outputBlobPtr->getTensorDesc().getPrecision();

    if (networkOutputPrecision == IE::Precision::FP32 || blobOutputPrecision == IE::Precision::FP32) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << "FP32 output is not supported.";
    }

    if (networkOutputPrecision != blobOutputPrecision) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                           << "Output blob precision is not the same as in network. Conversion is not supported.";
    }

    if (outputUniteData.size() != _outputs[outputName]->byteSize()) {
        THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite: " << outputUniteData.size()
                           << " and network expected output: " << _outputs[outputName]->byteSize();
    }

    {
        IE::MemoryBlob::Ptr mblob = IE::as<IE::MemoryBlob>(outputBlobPtr);
        if (!mblob) {
            THROW_IE_EXCEPTION << "Failed output blob type!";
        }
        auto lockedMemory = mblob->rmap();
        void* data = lockedMemory.as<void*>();
        memcpy(data, outputUniteData.data(), outputUniteData.size());
    }
}
