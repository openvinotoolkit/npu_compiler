//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

namespace vpux {
namespace hddl2 {

#define HDDLUNITE_ERROR_str std::string("[HDDLUNITE_ERROR] ")
#define FILES_ERROR_str std::string("[FILES_ERROR] ")
#define CONFIG_ERROR_str std::string("[INVALID CONFIG] ")
#define PARAMS_ERROR_str std::string("[INVALID PARAMS] ")
#define CONTEXT_ERROR_str std::string("[INVALID REMOTE CONTEXT] ")

///  Scheduler
#define FAILED_START_SERVICE                                     \
    std::string("Couldn't start the device scheduler service.\n" \
                "Please start the service or check the environment variable \"KMB_INSTALL_DIR\".")
#define SERVICE_AVAILABLE std::string("HDDL Scheduler service is available. Ready to go.")
#define SERVICE_NOT_AVAILABLE \
    std::string("HDDL Scheduler service is not available. Please start scheduler before running application.")

///  Network
#define FAILED_LOAD_NETWORK                                   \
    std::string("Couldn't load the graph into the device.\n"  \
                "Please check the service logs for errors.\n" \
                "A reboot may be required to restore the device to a functional state.")

/// Executor
#define EXECUTOR_NOT_CREATED                                                              \
    std::string("No executor has been created for the device, only export is possible.\n" \
                "For execution, please start the service or check the environment variable \"KMB_INSTALL_DIR\".")

///  Infer request
#define NO_EXECUTOR_FOR_INFERENCE               \
    std::string("Can't create infer request!\n" \
                "Please make sure that the device is available. Only exports can be made.")

}  //  namespace hddl2
}  //  namespace vpux
