//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef MV_ORDER_ERROR_HPP_
#define MV_ORDER_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class OrderError : public LoggedError
    {

    public:
            
        explicit OrderError(const LogSender& sender, const std::string& whatArg);
        explicit OrderError(const std::string& senderID, const std::string& whatArg);
        
    };

}

#endif //MV_ORDER_ERROR_HPP_
