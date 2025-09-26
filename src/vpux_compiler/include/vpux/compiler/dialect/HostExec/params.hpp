//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::HostExec {

enum HostMainFuncArgs {
    HOST_MAIN_FUNC_ARGS_CONTEXT,
    HOST_MAIN_FUNC_ARGS_DEVICE,
    HOST_MAIN_FUNC_ARGS_DDI_TABLE,
    HOST_MAIN_FUNC_ARGS_COMMAND_LIST,
    HOST_MAIN_FUNC_ARGS_COMMAND_LIST_COUNT,
    HOST_MAIN_FUNC_ARGS_COMMAND_QUEUE,
    HOST_MAIN_FUNC_ARGS_FENCE,
    HOST_MAIN_FUNC_ARGS_EVENT,
    HOST_MAIN_FUNC_ARGS_COUNT
};

#define GET_ARG_INDEX_CONTEXT(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_CONTEXT)
#define GET_ARG_INDEX_DEVICE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_DEVICE)
#define GET_ARG_INDEX_DDI_TABLE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_DDI_TABLE)
#define GET_ARG_INDEX_COMMAND_LIST(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_LIST)
#define GET_ARG_INDEX_COMMAND_LIST_COUNT(numFuncArgs) \
    ((numFuncArgs) -                                  \
     (vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_LIST_COUNT))
#define GET_ARG_INDEX_COMMAND_QUEUE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_QUEUE)
#define GET_ARG_INDEX_COMMAND_FENCE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_FENCE)
#define GET_ARG_INDEX_COMMAND_EVENT(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_EVENT)

#define HOST_EXEC_NETWORK_METADATA_NAME "HostExec.networkMetadata"
#define HOST_EXEC_NUM_SUBGRAPH_ATTR_NAME "HostExec.numSubgraphs"
}  // namespace vpux::HostExec
