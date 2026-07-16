//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <sstream>

using namespace vpux;

//
// Exceptions
//

template <typename ExceptionT>
void vpux::details::throwFormat(const char* file, int line, const std::string& message) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

#ifdef VPUX_DEVELOPER_BUILD
    // The error will be reported once caught. Reporting it at throw site is purely debug feature
    // and must be limited to respect user log level. Use OV_NPU_LOG_LEVEL=LOG_DEBUG to enable it.
    if (Logger::global().isActive(LogLevel::Debug)) {
        Logger::global().error("Got exception in {0}:{1} : {2}", file, line, message);
    }
#endif

    std::stringstream strm;
    strm
#ifndef NDEBUG
            << '\n'
            << file << ':' << line << ' '
#endif
            << message;
    throw ExceptionT(strm.str());
}

template void vpux::details::throwFormat<vpux::Exception>(const char* file, int line, const std::string& message);
