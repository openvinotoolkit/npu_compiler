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

#define NOMINMAX
#include "kmb_infer_request.h"

#include <debug.h>
#include <ie_blob.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <description_buffer.hpp>
#include <ie_plugin.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/perf_report.hpp>

#include "kmb_executable_network.h"
#include "kmb_preproc.hpp"

// TODO https://jira.devtools.intel.com/browse/CVS-21391
// FIXME: does not work for batch != 1
static bool is2DTensor(const InferenceEngine::SizeVector& dims) {
    size_t ones = std::count(dims.begin(), dims.end(), 1);
    return (dims.size() - ones) == 1;
}

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

KmbInferRequest::KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<StageMetaInfo>& blobMetaData,
    const KmbConfig& kmbConfig, const KmbExecutor::Ptr& executor)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executor(executor),
      _deviceLayout(Layout::NHWC),
      _stagesMetaData(blobMetaData),
      _config(kmbConfig),
      _blobWithResult(nullptr),
      _logger(std::make_shared<Logger>("KmbInferRequest", kmbConfig.logLevel(), consoleOutput())) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }

    if (_networkInputs.size() != 1) {
        THROW_IE_EXCEPTION << "Infer request supports only 1 input";
    }
    for (auto& networkInput : _networkInputs) {
        Precision precision = networkInput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr inputBlob = make_blob_with_precision(networkInput.second->getTensorDesc(), getKmbAllocator());
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
    }

    if (_networkOutputs.size() != 1) {
        THROW_IE_EXCEPTION << "Infer request supports only 1 output";
    }
    for (auto& networkOutput : _networkOutputs) {
        Precision precision = networkOutput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr outputBlob = nullptr;
        if (_config.forceFP16ToFP32() && precision == InferenceEngine::Precision::FP16) {
            const InferenceEngine::TensorDesc& outDesc = networkOutput.second->getTensorDesc();
            InferenceEngine::TensorDesc fp32Desc(
                InferenceEngine::Precision::FP32, outDesc.getDims(), outDesc.getLayout());
            _logger->warning("VPU_KMB_FORCE_FP16_TO_FP32 is enabled. Need to convert precision.");
            outputBlob = make_blob_with_precision(fp32Desc, getKmbAllocator());
        } else {
            outputBlob = make_blob_with_precision(networkOutput.second->getTensorDesc(), getKmbAllocator());
        }
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::dumpOutputBlobHelper(const Blob::Ptr& outputBlobPtr, const std::string& dst) {
    static unsigned dumpOutputCounter = 0;
    std::ostringstream inputFullPath;
    inputFullPath << dst;
    inputFullPath << "/output-dump";
    inputFullPath << dumpOutputCounter++;
    inputFullPath << ".bin";
    _logger->info("dumpOutputBlobHelper: dump to file ", inputFullPath.str());
    std::ofstream dumper(inputFullPath.str(), std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(outputBlobPtr->cbuffer().as<char*>(), outputBlobPtr->byteSize());
    } else {
        _logger->warning("dumpOutputBlobHelper: failed to open ", inputFullPath.str());
    }
    dumper.close();
}

void KmbInferRequest::InferAsync() {
    execPreprocessing(_inputs);

    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        dumpInputs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"));
    }

    // TODO: would be better to find a better place for such checks
    for (const auto& input : _inputs) {
        auto const inputBlobPtr = input.second;
        auto inputBlobPrecision = inputBlobPtr->getTensorDesc().getPrecision();
        if (inputBlobPrecision != Precision::FP16 && inputBlobPrecision != Precision::FP32 &&
            inputBlobPrecision != Precision::I8 && inputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";
    }
    for (const auto& output : _outputs) {
        auto const outputBlobPtr = output.second;
        auto outputBlobPrecision = outputBlobPtr->getTensorDesc().getPrecision();
        if (outputBlobPrecision != Precision::FP16 && outputBlobPrecision != Precision::FP32 &&
            outputBlobPrecision != Precision::I8 && outputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    const auto& deviceInputs = _executor->getNetworkInputs();
    if (deviceInputs.begin() == deviceInputs.end()) THROW_IE_EXCEPTION << "DeviceInputs are empty.";
    const auto deviceInputDesc = deviceInputs.begin()->second->getTensorDesc();
    const auto input = _inputs.begin()->second;

    auto updatedInput = prepareInputForInference(input, deviceInputDesc);

    _executor->queueInference(updatedInput->buffer().as<void*>(), updatedInput->byteSize());
}

void KmbInferRequest::execPreprocessing(InferenceEngine::BlobMap& inputs) {
    if (SippPreproc::useSIPP() && SippPreproc::isApplicable(inputs, _preProcData, _networkInputs)) {
        relocationAndExecSIPPDataPreprocessing(
            inputs, _networkInputs, _config.outColorFmtSIPP(), _config.numberOfSIPPShaves(), _config.SIPPLpi());
    } else {
        execDataPreprocessing(inputs);
    }
}

static bool isBlobPlacedInShareableMemory(const Blob::Ptr& blob) {
    return getKmbAllocator()->isValidPtr(blob->buffer().as<void*>());
}

// TODO: SIPP preprocessing usage can be merged to common preprocessing pipeline
void KmbInferRequest::relocationAndExecSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
    InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
    unsigned int lpi) {
    std::map<std::string, PreProcessDataPtr> preprocDataRealloc;
    for (const auto& input : inputs) {
        const std::string& inputName = input.first;
        auto preProcDataIter = _preProcData.find(inputName);
        if (preProcDataIter == _preProcData.end()) {
            continue;
        }

        preprocDataRealloc[preProcDataIter->first] = CreatePreprocDataHelper();
        Blob::Ptr blobData = preProcDataIter->second->getRoiBlob();
        if (blobData->is<NV12Blob>()) {
            // check if planes of nv12 blob were allocated with KMB allocator
            NV12Blob::Ptr origNV12Blob = as<NV12Blob>(blobData);
            Blob::Ptr& origYBlob = origNV12Blob->y();
            Blob::Ptr& origUVBlob = origNV12Blob->uv();

            Blob::Ptr kmbYBlob = origYBlob;
            if (!isBlobPlacedInShareableMemory(origYBlob)) {
                kmbYBlob = reallocateBlob(origYBlob);
            }
            Blob::Ptr kmbUVBlob = origUVBlob;
            if (!isBlobPlacedInShareableMemory(origUVBlob)) {
                kmbUVBlob = reallocateBlob(origUVBlob);
            }

            InferenceEngine::Blob::Ptr nv12Blob =
                InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(kmbYBlob, kmbUVBlob);
            preprocDataRealloc[preProcDataIter->first]->setRoiBlob(nv12Blob);
        } else {
            THROW_IE_EXCEPTION << "Attempt to pass non-NV12 image to SIPP preprocessing.";
        }
    }
    this->execSIPPDataPreprocessing(inputs, preprocDataRealloc, networkInputs, out_format, numShaves, lpi);
}

void KmbInferRequest::execSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
    std::map<std::string, PreProcessDataPtr>& preprocData, InferenceEngine::InputsDataMap& networkInputs,
    InferenceEngine::ColorFormat out_format, unsigned int numShaves, unsigned int lpi) {
    SippPreproc::execSIPPDataPreprocessing(inputs, preprocData, networkInputs, out_format, numShaves, lpi);
}

static bool needRepacking(const Blob::Ptr& actualInput, const TensorDesc& deviceTensorDesc) {
    // TODO: is2DTensor is a workaround for NHWC -> NC case
    // remove when mcm will support different input layout
    return (deviceTensorDesc.getLayout() != actualInput->getTensorDesc().getLayout() &&
            !is2DTensor(actualInput->getTensorDesc().getDims()));
}

static Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(
    const Blob::Ptr& blob, const Layout srcLayout, const Layout dstLayout) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    Blob::Ptr srcBlob = make_blob_with_precision(srcTensorDesc, blob->buffer());

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), dstLayout};
    Blob::Ptr dstBlob = make_blob_with_precision(dstTensorDesc, getKmbAllocator());
    dstBlob->allocate();

    vpu::copyBlob(srcBlob, dstBlob);
    return dstBlob;
}

static Blob::Ptr reallocateBlobToLayout(const Blob::Ptr& blob, const Layout layout) {
    auto allocator = getKmbAllocator();

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout};
    Blob::Ptr kmbBlob = make_blob_with_precision(dstTensorDesc, allocator);
    kmbBlob->allocate();

    vpu::copyBlob(blob, kmbBlob);

    return kmbBlob;
}

Blob::Ptr KmbInferRequest::reallocateBlob(const Blob::Ptr& blob) {
    return reallocateBlobToLayout(blob, blob->getTensorDesc().getLayout());
}

Blob::Ptr KmbInferRequest::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& expectedDesc) {
    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.forceNCHWToNHWC()) {
        _logger->warning("VPU_KMB_FORCE_NCHW_TO_NHWC is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(actualInput, Layout::NCHW, Layout::NHWC);
    } else {
        Blob::Ptr inputForInference;
        if (!isBlobPlacedInShareableMemory(actualInput)) {
            _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
            inputForInference = reallocateBlob(actualInput);
        } else {
            inputForInference = actualInput;
        }

        if (needRepacking(actualInput, expectedDesc)) {
            _logger->warning("Input blob is inconsistent with network input. Need to do re-layout.");
            inputForInference = reallocateBlobToLayout(actualInput, expectedDesc.getLayout());
        }

        return inputForInference;
    }
}

void KmbInferRequest::dumpInputs(const InferenceEngine::BlobMap& inputs, const std::string dstPath) const {
    if (dstPath.empty()) {
        _logger->warning(
            "Can not dump inputs since destination path is empty. Please check IE_VPU_KMB_DUMP_INPUT_PATH.");
        return;
    }
    for (const auto& input : inputs) {
        dumpInputBlobHelper(input.second, dstPath);
    }
}

void KmbInferRequest::dumpInputBlobHelper(const Blob::Ptr& inputBlobPtr, const std::string& dst) const {
    static unsigned dumpInputCounter = 0;
    std::ostringstream inputFullPath;
    inputFullPath << dst;
    inputFullPath << "/input-dump";
    inputFullPath << dumpInputCounter++;
    inputFullPath << ".bin";
    _logger->info("dumpInputBlobHelper: dump to file ", inputFullPath.str());
    std::ofstream dumper(inputFullPath.str(), std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(inputBlobPtr->cbuffer().as<char*>(), inputBlobPtr->byteSize());
    } else {
        _logger->warning("dumpInputBlobHelper: failed to open ", inputFullPath.str());
    }
    dumper.close();
}

static Blob::Ptr convertPrecision(
    const InferenceEngine::Blob::Ptr& sourceData, const InferenceEngine::TensorDesc& targetDesc) {
    InferenceEngine::TensorDesc sourceTensorDesc = sourceData->getTensorDesc();
    InferenceEngine::Precision targetPrecision = targetDesc.getPrecision();
    InferenceEngine::Precision sourcePrecision = sourceTensorDesc.getPrecision();
    if (sourcePrecision == targetPrecision) {
        return sourceData;
    }

    Blob::Ptr target =
        make_blob_with_precision(TensorDesc(targetPrecision, sourceTensorDesc.getDims(), sourceTensorDesc.getLayout()));
    target->allocate();
    if (sourcePrecision == InferenceEngine::Precision::FP16 && targetPrecision == InferenceEngine::Precision::FP32) {
        InferenceEngine::PrecisionUtils::f16tof32Arrays(
            target->buffer(), sourceData->cbuffer().as<ie_fp16*>(), sourceData->size(), 1.0f, 0.0f);
    } else if (sourcePrecision == InferenceEngine::Precision::FP32 &&
               targetPrecision == InferenceEngine::Precision::FP16) {
        InferenceEngine::PrecisionUtils::f32tof16Arrays(
            target->buffer(), sourceData->cbuffer().as<float*>(), sourceData->size());
    } else {
        THROW_IE_EXCEPTION << "Error: output precision conversion from " << sourcePrecision << " to " << targetPrecision
                           << " is not supported.";
    }
    return target;
}

void KmbInferRequest::GetResult() {
    auto dataName = _networkOutputs.begin()->first;

    auto foundInputBlob = _outputs.find(dataName);
    if (foundInputBlob == _outputs.end()) THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    // check that output layout is the same as device layout
    const InferenceEngine::OutputsDataMap& deviceOutputs = _executor->getNetworkOutputs();
    IE_ASSERT(deviceOutputs.size() == 1) << "Networks with " << deviceOutputs.size() << " outputs are not supported. "
                                         << "Only networks with 1 output are supported.";

    size_t output_size_total = std::accumulate(
        _outputs.begin(), _outputs.end(), 0, [](size_t sum, InferenceEngine::BlobMap::value_type& outputs) {
            return sum + outputs.second->byteSize();
        });

    Blob::Ptr& outputBlobRef = _outputs.begin()->second;
    InferenceEngine::TensorDesc deviceTensorDesc = deviceOutputs.begin()->second->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = outputBlobRef->getTensorDesc();

    InferenceEngine::Precision devicePrecision = deviceTensorDesc.getPrecision();
    InferenceEngine::Precision outputPrecision = outputTensorDesc.getPrecision();
    InferenceEngine::Layout deviceLayout = deviceTensorDesc.getLayout();
    InferenceEngine::Layout outputLayout = outputTensorDesc.getLayout();

    // is2DTensor is a workaround for NHWC -> NC case
    // TODO: remove when mcm will support different output layout
    if ((devicePrecision == outputPrecision) &&
        (deviceLayout == outputLayout || is2DTensor(outputTensorDesc.getDims()))) {
        // read result directly into output, do not copy blob
        void* outputPtr = outputBlobRef->buffer();
        _executor->getResult(outputPtr, output_size_total);
    } else {
        // read result into _blobWithResult
        if (_blobWithResult == nullptr) {
            _blobWithResult = make_blob_with_precision(deviceTensorDesc);
            _blobWithResult->allocate();
        }
        void* outputPtr = _blobWithResult->buffer();
        _executor->getResult(outputPtr, output_size_total);
        // do precision conversion when necessary
        Blob::Ptr blobWithCorrectPrecision = convertPrecision(_blobWithResult, outputTensorDesc);
        // copy blob with correct precision to the output blob
        // copyBlob does layout conversion on its own
        if (!is2DTensor(outputTensorDesc.getDims()) || devicePrecision != outputPrecision) {
            copyBlob(blobWithCorrectPrecision, _outputs.begin()->second);
        }
    }

    if (!_custom_outputs.empty()) {
        for (auto& output : _outputs) {
            auto name = output.first;

            auto custom_outputBlob = _custom_outputs[name];
            auto outputBlob = output.second;

            copyBlob(outputBlob, custom_outputBlob);
        }
    }

    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        dumpOutputBlobHelper(outputBlobRef, dumpOutputPathEnv);
    }
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}

void KmbInferRequest::Infer() {
    KmbInferRequest::checkBlobs();
    InferImpl();
}

void KmbInferRequest::checkBlobs() {
    if (_custom_outputs.empty()) {
        for (auto const& output : _outputs) {
            checkBlob(output.second, output.first, false);
        }
    } else {
        for (auto const& output : _custom_outputs) {
            checkBlob(output.second, output.first, false);
        }
    }
}
