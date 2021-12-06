//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// System
#include <memory>
#include <string>
// IE
#include "ie_compound_blob.h"
#include "ie_preprocess_data.hpp"
// Plugin
#include "hddl_unite/infer_data_adapter.h"
#include "vpux_remote_blob.h"
// Low-level
#include "Inference.h"

namespace vpux {
namespace hddl2 {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkDataNotNull(const IE::DataPtr& desc) {
    if (!desc) {
        IE_THROW() << "Data is null";
    }
}

// TODO [Workaround] Until we will be able to reset input blobs and call createBlob for same name again
//  It is useful if user sets NV12 blob, runs inference, and after that calls set blob with BGR blob, not NV
//  This will require recreating of BlobDesc due to different color format / size;
void InferDataAdapter::createInferData() {
    _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, _workloadContext, maxRoiNum,
                                                        _networkDescription->getDeviceOutputsInfo().size());
    if (_inferDataPtr.get() == nullptr) {
        IE_THROW() << "InferDataAdapter: Failed to create Unite inferData";
    }

    for (const auto& deviceInput : _networkDescription->getDeviceInputsInfo()) {
        const auto& inputName = deviceInput.first;
        const auto& blobDesc = deviceInput.second;
        checkDataNotNull(blobDesc);

        const bool isInput = true;
        std::shared_ptr<BlobDescriptorAdapter> blobDescriptorPtr(
                new BlobDescriptorAdapter(getBlobType(_haveRemoteContext), blobDesc, _graphColorFormat, isInput));

        _inputs[inputName] = blobDescriptorPtr;
    }

    for (const auto& networkOutput : _networkDescription->getDeviceOutputsInfo()) {
        const auto& outputName = networkOutput.first;
        const auto& blobDesc = networkOutput.second;
        checkDataNotNull(blobDesc);

        const bool isInput = false;
        std::shared_ptr<BlobDescriptorAdapter> blobDescriptorPtr(
                new BlobDescriptorAdapter(getBlobType(_haveRemoteContext), blobDesc, _graphColorFormat, isInput));

        const auto HDDLUniteBlobDesc = blobDescriptorPtr->createUniteBlobDesc(isInput);
        _inferDataPtr->createBlob(outputName, HDDLUniteBlobDesc, isInput);

        _outputs[outputName] = blobDescriptorPtr;
    }
}
//------------------------------------------------------------------------------
InferDataAdapter::InferDataAdapter(const vpux::NetworkDescription::CPtr& networkDescription,
                                   const HddlUnite::WorkloadContext::Ptr& workloadContext,
                                   const InferenceEngine::ColorFormat graphColorFormat)
        : _networkDescription(networkDescription),
          _workloadContext(workloadContext),
          _graphColorFormat(graphColorFormat),
          _haveRemoteContext(workloadContext != nullptr),
          _needUnitePreProcessing(true) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};
    if (networkDescription == nullptr) {
        IE_THROW() << "InferDataAdapter: NetworkDescription is null";
    }
    _needUnitePreProcessing = false;

    createInferData();
}

void InferDataAdapter::setPreprocessFlag(const bool preprocessingRequired) {
    _needUnitePreProcessing = preprocessingRequired;
}

static bool isInputBlobDescAlreadyCreated(const HddlUnite::Inference::InferData::Ptr& inferDataPtr,
                                          const std::string& inputBlobName) {
    const auto& inputBlobs = inferDataPtr->getInBlobs();
    auto result =
            std::find_if(inputBlobs.begin(), inputBlobs.end(),
                         [&inputBlobName](const std::pair<std::string, HddlUnite::Inference::InBlob::Ptr>& element) {
                             return element.first == inputBlobName;
                         });
    return result != inputBlobs.end();
}

void InferDataAdapter::prepareUniteInput(const InferenceEngine::Blob::CPtr& blob, const std::string& inputName,
                                         const InferenceEngine::ColorFormat inputColorFormat) {
    if (blob == nullptr) {
        IE_THROW() << "InferDataAdapter: Blob for input is null";
    }
    if (_inputs.find(inputName) == _inputs.end()) {
        IE_THROW() << "InferDataAdapter: Failed to find BlobDesc for: " << inputName;
    }

    std::cout << "PrepareUniteInput" << std::endl;
    auto blobDescriptorPtr = _inputs.at(inputName);

    // Check that created blob description is suitable for input blob
    const auto blobDescSuitable = blobDescriptorPtr->isBlobDescSuitableForBlob(blob);
    if (!blobDescSuitable) {
        std::cout << "blobDescSuitable = false" << std::endl;
        const auto& deviceInputInfo = _networkDescription->getDeviceInputsInfo().at(inputName);
        std::shared_ptr<BlobDescriptorAdapter> newBlobDescriptorPtr(
                new BlobDescriptorAdapter(blob, _graphColorFormat, deviceInputInfo));

        _inputs[inputName] = newBlobDescriptorPtr;
        blobDescriptorPtr = newBlobDescriptorPtr;

        // TODO [Worklaround] !!! Check if blob already exists. If so, well, create inferRequest again
        if (isInputBlobDescAlreadyCreated(_inferDataPtr, inputName)) {
            std::cout << "input blob desc already created - call createInferData() again" << std::endl;
            createInferData();
        }
    }

    createInferData();

    const bool isInput = true;
    const auto& HDDLUniteBlobDesc = blobDescriptorPtr->createUniteBlobDesc(isInput);

    // Postponed blob creation
    if (/*!isInputBlobDescAlreadyCreated(_inferDataPtr, inputName)*/ true) {
        std::cout << "Postponed blob creation - just before inferDataPtr->createBlob" << std::endl;
        const auto success = _inferDataPtr->createBlob(inputName, HDDLUniteBlobDesc, isInput);
        if (!success) {
            IE_THROW() << "InferDataAdapter: Error creating HDDLUnite Blob";
        }
        /** setPPFlag should be called after inferData->createBLob call but before updateBlob **/
        _inferDataPtr->setPPFlag(_needUnitePreProcessing);

        if ((_needUnitePreProcessing || blobDescriptorPtr->isROIPreprocessingRequired()) && _haveRemoteContext) {
            std::cout << "Remote condition - creating NNDesc & set NNInputDesc" << std::endl;
            auto nnBlobDesc = blobDescriptorPtr->createNNDesc();
            _inferDataPtr->setNNInputDesc(nnBlobDesc);
        } else {
            std::cout << "Local condition - doing nothing" << std::endl;
        }
    }

    const auto updatedHDDLUniteBlobDesc = blobDescriptorPtr->updateUniteBlobDesc(blob, inputColorFormat);
    if (!_inferDataPtr->getInputBlob(inputName)->updateBlob(updatedHDDLUniteBlobDesc)) {
        IE_THROW() << "InferDataAdapter: Error updating Unite Blob";
    }
    // Strides alignment is supported only for VideoWorkload + PP case
    if (updatedHDDLUniteBlobDesc.m_widthStride % updatedHDDLUniteBlobDesc.m_resWidth) {
        if (!_needUnitePreProcessing || !updatedHDDLUniteBlobDesc.m_isRemoteMem) {
            IE_THROW()
                    << "InferDataAdapter: strides alignment is supported only for Video Workload + preprocessing case.";
        }
    }
}

std::string InferDataAdapter::getOutputData(const std::string& outputName) {
    // TODO send roiIndex (second parameter)
    auto outputData = _inferDataPtr->getOutputData(outputName);
    if (outputData.empty()) {
        IE_THROW() << "Failed to get blob from hddlUnite!";
    }
    _profileData = _inferDataPtr->getProfileData();
    return outputData;
}

void InferDataAdapter::waitInferDone() const {
    auto status = _inferDataPtr->waitInferDone(_asyncInferenceWaitTimeoutMs);
    if (status != HDDL_OK) {
        IE_THROW() << "Failed to wait for inference result with error: " << status;
    }
}

std::map<std::string, IE::InferenceEngineProfileInfo> InferDataAdapter::getHDDLUnitePerfCounters() const {
    std::map<std::string, IE::InferenceEngineProfileInfo> perfCounts;
    IE::InferenceEngineProfileInfo info;
    info.status = IE::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;
    info.realTime_uSec = 0;

    info.realTime_uSec = static_cast<long long>(_profileData.infer.time);
    perfCounts["Total scoring time"] = info;

    info.realTime_uSec = static_cast<long long>(_profileData.nn.time);
    perfCounts["Total scoring time on inference"] = info;

    info.realTime_uSec = static_cast<long long>(_profileData.pp.time);
    perfCounts["Total scoring time on preprocess"] = info;
    return perfCounts;
}
}  // namespace hddl2
}  // namespace vpux
