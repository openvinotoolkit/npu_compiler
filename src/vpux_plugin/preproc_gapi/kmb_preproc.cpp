// Copyright (C) 2019 Intel Corporation
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

#include "kmb_preproc.hpp"

#include "vpux/utils/core/helper_macros.hpp"

#include <map>
#include <memory>
#include <string>

#if defined(__arm__) || defined(__aarch64__)
#include "kmb_preproc_pool.hpp"
#endif

namespace InferenceEngine {
namespace KmbPreproc {

#if defined(__arm__) || defined(__aarch64__)
static bool supported(ResizeAlgorithm interp, ColorFormat inFmt) {
    return (interp == RESIZE_BILINEAR) && (inFmt == ColorFormat::NV12);
}
#endif

bool isApplicable(const InferenceEngine::BlobMap& inputs, const std::map<std::string, PreProcessDataPtr>& preprocData,
                  InputsDataMap& networkInputs) {
#if defined(__arm__) || defined(__aarch64__)
    if (inputs.size() != 1 || preprocData.empty())
        return false;

    for (auto& input : inputs) {
        const auto& blobName = input.first;
        auto it = preprocData.find(blobName);
        if (it != preprocData.end()) {
            const auto& preprocInfo = networkInputs[blobName]->getPreProcess();
            if (!supported(preprocInfo.getResizeAlgorithm(), preprocInfo.getColorFormat())) {
                return false;
            }
        }
    }
    return true;
#else
    VPUX_UNUSED(inputs);
    VPUX_UNUSED(preprocData);
    VPUX_UNUSED(networkInputs);
    return false;
#endif
}

void execDataPreprocessing(InferenceEngine::BlobMap& inputs, std::map<std::string, PreProcessDataPtr>& preprocData,
                           InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format,
                           unsigned int numShaves, unsigned int lpi, unsigned int numPipes,
                           const std::string& preprocPoolId, const int deviceId, Path ppPath) {
#if defined(__arm__) || defined(__aarch64__)
    IE_ASSERT(numShaves > 0 && numShaves <= 16)
            << "KmbPreproc::execDataPreprocessing "
            << "attempt to set invalid number of shaves: " << numShaves << ", valid numbers are from 1 to 16";
    preprocPool().execDataPreprocessing({inputs, preprocData, networkInputs, out_format}, numShaves, lpi, numPipes,
                                        ppPath, preprocPoolId, deviceId);
#else
    VPUX_UNUSED(inputs);
    VPUX_UNUSED(preprocData);
    VPUX_UNUSED(networkInputs);
    VPUX_UNUSED(out_format);
    VPUX_UNUSED(numShaves);
    VPUX_UNUSED(lpi);
    VPUX_UNUSED(numPipes);
    VPUX_UNUSED(preprocPoolId);
    VPUX_UNUSED(deviceId);
    VPUX_UNUSED(ppPath);
    IE_THROW() << "VPUAL is disabled. Used only for arm";
#endif
}
}  // namespace KmbPreproc
}  // namespace InferenceEngine
