//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_logger.hpp
 * @brief Define VCLLogger which stores error message
 */

#pragma once

#include "compatibility/npu_driver_compiler.h"
#include "vpux/utils/logger/logger.hpp"

#include <mutex>

namespace VPUXDriverCompiler {

/**
 * @brief Store error log and output other logs to terminal based on the debug level
 *
 */
class VCLLogger final : public vpux::Logger {
public:
    /**
     * @brief Create logger based on debug level
     *
     * @param name The name of this logger
     * @param lvl Log smaller than this level will not show
     * @param saveErrorLog Save error message for user
     */
    VCLLogger(llvm::StringLiteral name, vpux::LogLevel level, bool saveErrorLog)
            : Logger(name, level), _saveErrorLog(saveErrorLog) {
    }

    /**
     * @brief Get the stored error log
     *
     * @param size The size of stored error msg
     * @param log  The user buffer to store error msg
     * @return vcl_result_t
     */
    vcl_result_t getString(size_t* size, char* log) {
        if (size == nullptr) {
            Logger::error("Invalid argument to get log!");
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        std::lock_guard<std::mutex> mLock(_lock);
        const char* localLog = _log.c_str();
        auto localLogSize = _log.size();
        if (localLog == nullptr || localLogSize == 0) {
            /// No error log
            *size = 0;
        } else {
            if (log == nullptr) {
                /// Return actual size if pointer is nullptr, extra 1 to store '\0'
                *size = localLogSize + 1;
            } else if (log != nullptr && *size == localLogSize + 1) {
                /// Copy local log content if the pointer is valid
                memcpy(log, localLog, localLogSize + 1);
            } else {
                Logger::error("Invalid value of size to get log!");
                return VCL_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }
        return VCL_RESULT_SUCCESS;
    }

    /**
     * @brief Save error log if the swich is open, otherwrise, output to terminal
     *
     * @param log The content of error message
     */
    void outputError(const std::string& log) {
        if (_saveErrorLog) {
            /// Append error message to local container
            auto size = log.size();
            if (size == 0) {
                return;
            }
            _lock.lock();
            // Show new log in next line
            _log.append(formatv("[{0}] {1}\n", name(), log));
            _lock.unlock();
        } else {
            /// Use terminal to process error message, output to terminal based on log level
            Logger::error("{0}", log.c_str());
        }
    }

private:
    bool _saveErrorLog;  ///< Save error message to local string instead of output to terminal
    std::mutex _lock;
    std::string _log;  ///< Store the error message if _saveErrorLog is true
};

}  // namespace VPUXDriverCompiler
