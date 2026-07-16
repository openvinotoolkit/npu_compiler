//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_query_network.hpp
 * @brief Define VPUXQueryNetworkL0 which stores query result of a model.
 */

#pragma once

#include "vcl_logger.hpp"

#include <memory>

namespace VPUXDriverCompiler {

class VPUXExecutableL0;

/**
 * @brief Store query result and serialize data for adapter.
 *
 */
class VPUXQueryNetworkL0 final {
public:
    explicit VPUXQueryNetworkL0(std::shared_ptr<VCLLogger> vclLogger);

    /**
     * @brief Serialize supported layer to special format which can be passed to adapter.
     *
     * @note The final format: <name_0><name_1><name_2>...<name_n>.
     *
     * @param layerMap Include the support layers of a model by VPUX MLIR compiler.
     * @return vcl_result_t
     */
    vcl_result_t setQueryResult(const std::map<std::string, std::string>& layerMap);

    /**
     * @brief Store the size of serialized query result.
     *
     * @param stringSize Store the size of serialized query result.
     * @return vcl_result_t
     */
    vcl_result_t getQueryResultSize(uint64_t* stringSize) const;

    /**
     * @brief Get the Query String object.
     *
     * @param inputStr The buffer to store serialized query result.
     * @param inputSize  The size of inputStr, need to be same with result of getQueryResultSize().
     * @return vcl_result_t
     */
    vcl_result_t getQueryString(uint8_t* inputStr, uint64_t inputSize) const;

private:
    std::vector<uint8_t> queryResultVec;  ///< The serialized query result
    uint64_t size = 0;
    std::shared_ptr<VCLLogger> _logger;
};

}  // namespace VPUXDriverCompiler
