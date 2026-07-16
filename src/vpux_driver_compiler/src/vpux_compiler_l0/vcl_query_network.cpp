//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_query_network.hpp"

#include <memory>

namespace VPUXDriverCompiler {

VPUXQueryNetworkL0::VPUXQueryNetworkL0(std::shared_ptr<VCLLogger> vclLogger): _logger(std::move(vclLogger)) {
}

vcl_result_t VPUXQueryNetworkL0::setQueryResult(const std::map<std::string, std::string>& layerMap) {
    /// Set the value of queryResultString
    /// Format query result
    /// <name_0><name_1><name_2>...<name_n>
    /// size = (layerName.length + 2) * layerCount
    size = 0;
    size_t i = 0;
    for (auto& name : layerMap) {
        size = size + name.first.length() + 2;
    }
    queryResultVec.resize(size);
    uint8_t charSplitLeft = '<';
    uint8_t charSplitRight = '>';
    for (auto& name : layerMap) {
        queryResultVec[i++] = charSplitLeft;
        memcpy(&queryResultVec[i], (uint8_t*)(name.first.c_str()), name.first.length());
        i += name.first.length();
        queryResultVec[i++] = charSplitRight;
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXQueryNetworkL0::getQueryResultSize(uint64_t* stringSize) const {
    /// Get the size of queryResultString
    if (stringSize == nullptr) {
        _logger->outputError("Can not return size for NULL argument!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    *stringSize = size;
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXQueryNetworkL0::getQueryString(uint8_t* inputStr, uint64_t inputSize) const {
    /// Copy the value from queryResultString to inputStr
    if (inputSize != size) {
        _logger->outputError("Input size does not match size of queryResultString!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (inputStr == nullptr) {
        _logger->outputError("Invalid input pointer of queryResult!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    memcpy(inputStr, queryResultVec.data(), size);
    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler
