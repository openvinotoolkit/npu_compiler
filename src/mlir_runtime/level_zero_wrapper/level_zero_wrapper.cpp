//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#define __STDC_FORMAT_MACROS 1

#include "level_zero_wrapper.h"

#include <inttypes.h>

#include <stdio.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/network_metadata.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux_headers/serial_metadata.hpp"
#include "ze_graph_ext.h"

/* #define TEST */

// Workaround for win specific save funtions
#ifndef _WIN32
#include <errno.h>
#include <string.h>

// Basic strcpy_s implementation for Linux
inline error_t strcpy_s(char* dest, size_t destsz, const char* src) {
    if (dest == NULL || src == NULL || destsz == 0) {
        return EINVAL;
    }

    size_t srcsz = strlen(src);
    if (srcsz >= destsz) {
        dest[0] = '\0';  // Null terminate even in case of failure
        return ERANGE;
    }

    strcpy(dest, src);
    return 0;
}

template <size_t size>
inline error_t strcpy_s(char (&dest)[size], const char* src) {
    return strcpy_s(dest, size, src);
}

// Basic memcpy_s implementation for Linux
inline error_t memcpy_s(void* dest, size_t destsz, const void* src, size_t count) {
    if (dest == NULL || src == NULL) {
        return EINVAL;
    }

    if (destsz < count) {
        return ERANGE;
    }

    memcpy(dest, src, count);
    return 0;
}
#endif

// End of workaround for win specific save funtions

#define RETURN_SUCCESS() return static_cast<uint32_t>(ZE_RESULT_SUCCESS);
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#define ENABLE_COMMANDLIST_SUBMISSION_MODE
#endif

#ifdef TEST
NPU_API(void*) npu_level_zero_alloc(int64_t size, void*, void*) {
    printf("npu_level_zero_alloc was called %lld\n", size);
#if !defined(WIN32)
    void* result = aligned_alloc(64, size);
#else
    void* result = _aligned_malloc(size, 64);
#endif
    return result;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void* commandList) {
    printf("npu_level_zero_append_memory_copy was called %ld\n", size);
    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    printf("npu_level_zero_append_barrier was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue, void* execCtx) {
    printf("npu_level_zero_create_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue, void* execCtx) {
    printf("npu_level_zero_create_graphs was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** input, int32_t numInputs, void** output, int32_t numOutputs, void* kernel,
                             int64_t kernelSize, void* context, void* device, void* ddiTable, void* commandList) {
    printf("npu_level_zero_execute_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void* commandList, void* commandQueue, void* fence, void* event, void* execCtx) {
    printf("npu_level_zero_submit_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    printf("npu_level_zero_get_last_error was called\n");
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void* commandList) {
    printf("npu_level_zero_reset_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandList, int32_t numCommandLists) {
    printf("npu_level_zero_reset_commandlists was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* levelZeroMetadata, void* inputDescs,
                                    void* outputDescs) {
    printf("npu_level_zero_metadata was called\n");
    RETURN_SUCCESS();
}

#else
struct graph_info {
    ze_graph_handle_t graphHandle;
    uint32_t numArgs;
    uint32_t numInputArgs;
    graph_info(): graphHandle(nullptr), numArgs(0), numInputArgs(0) {
    }
    graph_info(ze_graph_handle_t handle, uint32_t numArgs, uint32_t numInputArgs)
            : graphHandle(handle), numArgs(numArgs), numInputArgs(numInputArgs) {
    }
};

#if defined(WIN32)
constexpr uint32_t default_cmdlist_id = 1;
#else
constexpr uint32_t default_cmdlist_id = 0;
#endif

constexpr size_t max_message_length = 256;
std::unique_ptr<vpux::Logger> logger = nullptr;
static char lastErrorMessage[max_message_length];
#define ERROR_HANDLE(result, format, ...)                                                              \
    if (result != ZE_RESULT_SUCCESS) {                                                                 \
        std::snprintf(lastErrorMessage, max_message_length, format ": 0x%04X", ##__VA_ARGS__, result); \
        if (logger != nullptr) {                                                                       \
            logger->error("{0}", lastErrorMessage);                                                    \
        }                                                                                              \
        return static_cast<int32_t>(result);                                                           \
    }

ze_graph_dditable_ext_t* ddiTableHandle = nullptr;
std::map<void*, graph_info>* graphMap = nullptr;

enum UseInternalCmdListMode {
    USE_INTERNAL_CMDLIST_MODE_DEFAULT = 0,  // No use of internal command list
    USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP =
            1,  // Use internal command list for each inference group, and submit after each group
    USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE =
            2,  // Use internal command list for each inference, and submit after each inference
};

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
UseInternalCmdListMode useInternalCmdListMode = USE_INTERNAL_CMDLIST_MODE_DEFAULT;
uint64_t maxCmdListCount = 200;
std::vector<ze_command_list_handle_t>* commandlistPool = nullptr;
uint64_t curCmdListIndex = 0;
ze_command_list_handle_t curCmdListHandle = nullptr;

uint64_t maxInferenceCountPerGroup = 10;
uint64_t curInferenceCountInGroup = 0;
#endif

struct scratch_buffer {
    ze_context_handle_t contextHandle;
    void* data;
    int64_t size;

    bool inRange(uint64_t address) const {
        return ((address >= reinterpret_cast<uint64_t>(data)) && (address < (reinterpret_cast<uint64_t>(data) + size)));
    }
};

struct ze_memory_deleter {
    void operator()(scratch_buffer* buffer) const {
        if (buffer != nullptr) {
            if (buffer->data != nullptr) {
                zeMemFree(buffer->contextHandle, buffer->data);
            }
        }
    }
};

struct graph_argument_binding {
    uint64_t cmdId;
    ze_command_list_handle_t commandListHandle;
    uint64_t networkArgIndex;
    uint64_t argIndex;

    uint64_t bufferOffset;
};

std::ostream& operator<<(std::ostream& o, const graph_argument_binding& binding) {
    o << "cmdId: " << binding.cmdId;
    o << ", commandListHandle: 0x" << std::hex << reinterpret_cast<uint64_t>(binding.commandListHandle) << std::dec;
    o << ", networkArgIndex: " << binding.networkArgIndex;
    o << ", argIndex: " << binding.argIndex;
    o << ", bufferOffset: " << binding.bufferOffset;
    return o;
}

struct execution_context {
    // commandListIndex, networkArgIndex, list of bindings for the same network argument in the same command list
    std::vector<std::vector<std::vector<graph_argument_binding>>> argumentBindings;
    std::vector<uint64_t> mutableCommandListIds;
    std::vector<ze_event_handle_t> events;
    ze_event_pool_handle_t eventPool;
    size_t numSubGraphs;
    bool isInitialized;
    size_t curEventIndex;
    size_t signalEventCount;
    std::unique_ptr<scratch_buffer, ze_memory_deleter> scratchBuffer = nullptr;
    bool isOptimizedDynamicStridesSupported = false;

    execution_context(size_t numSubGraphs, size_t numNetworkArgs)
            : argumentBindings(numSubGraphs), mutableCommandListIds(numSubGraphs), numSubGraphs(numSubGraphs) {
        for (auto& bindings : argumentBindings) {
            bindings.resize(numNetworkArgs);
        }
        eventPool = nullptr;
        isInitialized = false;
        curEventIndex = 0;
        signalEventCount = 0;
    }

    execution_context(const execution_context&) = delete;
    execution_context(execution_context&&) = delete;
    execution_context& operator=(const execution_context&) = delete;
    execution_context& operator=(execution_context&&) = delete;
    ~execution_context() {
        if (eventPool != nullptr) {
            for (auto& event : events) {
                zeEventDestroy(event);
            }

            zeEventPoolDestroy(eventPool);
        }

        if (scratchBuffer != nullptr) {
            scratchBuffer.reset();
        }
    }

    void add_binding(size_t graphIndex, const graph_argument_binding& binding) {
        argumentBindings[graphIndex][binding.networkArgIndex].emplace_back(binding);

        if (logger) {
            std::ostringstream oss;
            oss << "Added a binding[" << graphIndex << ", " << binding.networkArgIndex << "] " << binding;
            logger->info("{0}", oss.str());
        }
    }

    std::tuple<uint32_t, std::string> queryDriverExtensionVersion(
            const char* extName, uint32_t extCurrentVersion, std::vector<ze_driver_extension_properties_t>& extProps,
            uint32_t count) {
        const char* functionExtName = nullptr;
        uint32_t targetVersion = 0;

        for (uint32_t i = 0; i < count; ++i) {
            auto& property = extProps[i];

            if (strncmp(property.name, extName, strlen(extName)) != 0) {
                continue;
            }

            if (property.version >= extCurrentVersion) {
                functionExtName = property.name;
                targetVersion = extCurrentVersion;
                break;
            }

            // Use the latest version supported by the driver - We need to go through all the properties for older
            // drivers that use specific names for different graph ext versions, e.g.: ZE_extension_graph_1_1,
            // ZE_extension_graph_1_2
            if (property.version > targetVersion) {
                functionExtName = property.name;
                targetVersion = property.version;
            }
        }

        return std::make_tuple(targetVersion, functionExtName ? functionExtName : "");
    }

    void reset(void** commandList, uint64_t numCommandLists) {
        for (auto& bindingsPerCommandList : argumentBindings) {
            for (auto& bindings : bindingsPerCommandList) {
                bindings.clear();
            }
        }
        if (commandList == nullptr || numCommandLists == 0) {
            for (uint64_t i = 0; i < numCommandLists; ++i) {
                mutableCommandListIds[i] = 0;
            }
        } else {
            ze_command_list_handle_t* commandListHandles = reinterpret_cast<ze_command_list_handle_t*>(commandList);
            if (commandListHandles[0] != nullptr) {
                // the first command list may have some commands recorded in npu plugin
                auto commandListHandle = commandListHandles[0];
                uint64_t cmdId = 0;
                if (commandListHandle != nullptr) {
                    ze_mutable_command_id_exp_desc_t mutable_cmd_id_desc = {};
                    mutable_cmd_id_desc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC;
                    mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS;
                    auto result = zeCommandListGetNextCommandIdExp(commandListHandle, &mutable_cmd_id_desc, &cmdId);
                    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
                        // If the command list is closed, initialize cmdId with default_cmdlist_id
                        cmdId = default_cmdlist_id;
                    } else {
                        if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
                            // If ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS is not supported by the driver, try again
                            // with ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED
                            mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED;
                            zeCommandListGetNextCommandIdExp(commandListHandle, &mutable_cmd_id_desc, &cmdId);
                        }
                    }
                }
                mutableCommandListIds[0] = cmdId;
            }

            for (uint64_t i = 1; i < numCommandLists; ++i) {
                // For command lists after the first, initialize the mutable command id with
                // default_cmdlist_id because the NPU plugin does not pre-record inference
                // execution commands in those command lists. The default value is platform-specific.
                mutableCommandListIds[i] = default_cmdlist_id;
            }
        }

        resetEvents();
    }

    int32_t initialize(ze_graph_dditable_ext_t* ddiTable, ze_device_handle_t deviceHandle,
                       ze_context_handle_t context) {
        isOptimizedDynamicStridesSupported = false;
        if (ddiTable != nullptr && ddiTable->pfnCompilerIsOptionSupported != nullptr) {
            isOptimizedDynamicStridesSupported =
                    ddiTable->pfnCompilerIsOptionSupported(deviceHandle, ZE_NPU_DRIVER_OPTIONS,
                                                           "OPTIMIZED_DYNAMIC_STRIDES", nullptr) == ZE_RESULT_SUCCESS;
        }

        return createEventPool(deviceHandle, context);
    }

    int32_t createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context) {
        if (numSubGraphs > 1) {
            auto eventCount = (numSubGraphs - 1);
            ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                                    ZE_EVENT_POOL_FLAG_HOST_VISIBLE, static_cast<uint32_t>(eventCount)};
            auto result = zeEventPoolCreate(context, &event_pool_desc, /*numDevices*/ 1, &deviceHandle, &eventPool);
            ERROR_HANDLE(result, "Failed to create event pool for execution context");

            events.resize(eventCount);
            for (size_t i = 0; i < eventCount; i++) {
                ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, static_cast<uint32_t>(i), 0, 0};
                result = zeEventCreate(eventPool, &event_desc, &events[i]);
                ERROR_HANDLE(result, "Failed to create event");
            }
        }

        isInitialized = true;
        return ZE_RESULT_SUCCESS;
    }

    void resetEvents() {
        curEventIndex = 0;
        signalEventCount = 0;
    }

    ze_event_handle_t getSignalEvent() {
        if (events.empty()) {
            return nullptr;
        }

        if (curEventIndex < events.size()) {
            signalEventCount++;
            return events[curEventIndex];
        }

        return nullptr;
    }

    ze_event_handle_t getWaitEvent() {
        if (events.empty() || (signalEventCount == 0)) {
            return nullptr;
        }

        signalEventCount = 0;
        if (curEventIndex < events.size()) {
            return events[curEventIndex++];
        }
        return nullptr;
    }
};

std::unique_ptr<scratch_buffer, ze_memory_deleter> scratchBuffer = nullptr;

// shared library initialization function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_init() {
    graphMap = new std::map<void*, graph_info>();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    std::string logLevelFlag;
    vpux::parseEnv("OV_NPU_LOG_LEVEL", logLevelFlag);
    if (logLevelFlag.size() > 0 && std::string(logLevelFlag).find("LOG_") == 0) {
        logger = std::make_unique<vpux::Logger>(llvm::StringLiteral("LEVEL_ZERO_WRAPPER"),
                                                vpux::Logger::global().level());
    }
#endif

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    std::string internalCmdListFlag;
    vpux::parseEnv("ENABLE_INTERNAL_CMDLIST", internalCmdListFlag);
    if (internalCmdListFlag.size() > 0) {
        useInternalCmdListMode = internalCmdListFlag == "1"
                                         ? USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP
                                         : (internalCmdListFlag == "2" ? USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE
                                                                       : USE_INTERNAL_CMDLIST_MODE_DEFAULT);
        if (logger) {
            logger->info("Internal CommandList Mode: {0}", static_cast<uint64_t>(useInternalCmdListMode));
        }
    }

    std::string internalCmdListCountFlag;
    vpux::parseEnv("INTERNAL_CMDLIST_MAX_COUNT", internalCmdListCountFlag);
    if (internalCmdListCountFlag.size() > 0) {
        int64_t internalCmdListCount = std::stoll(internalCmdListCountFlag);
        if (internalCmdListCount > 0) {
            maxCmdListCount = internalCmdListCount;
            if (logger) {
                logger->info("Max number of command lists: {0}", maxCmdListCount);
            }
        }
    }

    std::string internalMaxInferenceCountPerGroupFlag;
    vpux::parseEnv("INTERNAL_CMDLIST_MAX_INFERENCE_COUNT_PER_GROUP", internalMaxInferenceCountPerGroupFlag);
    if (internalMaxInferenceCountPerGroupFlag.size() > 0) {
        int64_t internalCmdListMaxInferenceCountPerGroup = std::stoll(internalMaxInferenceCountPerGroupFlag);
        if (internalCmdListMaxInferenceCountPerGroup > 0) {
            maxInferenceCountPerGroup = internalCmdListMaxInferenceCountPerGroup;
            if (logger) {
                logger->info("Max number of inferences per group: {0}", maxInferenceCountPerGroup);
            }
        }
    }

    if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
        commandlistPool = new std::vector<ze_command_list_handle_t>();
        commandlistPool->reserve(maxCmdListCount);
    }
#endif
}

// shared library destroy function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_destroy() {
    if (graphMap) {
        if (ddiTableHandle) {
            for (auto& g : *graphMap) {
                ddiTableHandle->pfnDestroy(g.second.graphHandle);
                g.second.graphHandle = nullptr;
            }
        }
        delete graphMap;
        graphMap = nullptr;
    }

    if (scratchBuffer) {
        scratchBuffer.reset();
    }

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    if (commandlistPool) {
        for (auto handle : *commandlistPool) {
            if (handle != nullptr) {
                zeCommandListDestroy(handle);
            }
        }

        delete commandlistPool;
    }
#endif

    if (logger) {
        logger.reset();
    }
}

NPU_API(void*) npu_level_zero_alloc(int64_t bytes, void* context, void* executionContext) {
    auto execCtx = reinterpret_cast<execution_context*>(executionContext);
    auto& scratchBuffer = (execCtx != nullptr) ? execCtx->scratchBuffer : ::scratchBuffer;

    if (scratchBuffer != nullptr && scratchBuffer->size >= bytes) {
        return scratchBuffer->data;
    }

    if (scratchBuffer == nullptr || scratchBuffer->size < bytes) {
        if (logger) {
            logger->info("Allocating scratch buffer of size {0} bytes, previous ptr {1}", bytes,
                         scratchBuffer ? scratchBuffer->data : nullptr);
        }
        ze_host_mem_alloc_flag_t flag = {};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                         static_cast<ze_host_mem_alloc_flags_t>(flag)};
        auto contextHandle = static_cast<ze_context_handle_t>(context);
        void* data = nullptr;
        // user is responsible for alignment, so we pass 0 for alignment
        int32_t res = zeMemAllocHost(contextHandle, &desc, bytes, /*no alignment*/ 0, &data);

        if (res == ZE_RESULT_SUCCESS) {
            scratchBuffer.reset(new scratch_buffer{contextHandle, data, bytes});
            if (logger) {
                logger->info("Allocated scratch buffer of size {0} bytes, ptr {1}", bytes, data);
            }
            return data;
        } else if (logger) {
            logger->info("Cannot allocate scratch buffer of size {0} bytes, result {1}", bytes, res);
        }
    }

    return nullptr;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void** commandList) {
    auto commandListHandle =
            (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle");
    }
    auto result = zeCommandListAppendMemoryCopy(commandListHandle, dst, src, size, nullptr, 0, nullptr);
    ERROR_HANDLE(result, "Failed to append memory copy from: %p to: %p, size: %" PRId64, src, dst, size);

    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        ERROR_HANDLE(result, "Failed to append barrier");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue, void* execCtx) {
    if (logger) {
        logger->info("Creating graph for kernel at address {0} of size {1}", kernel, kernelSize);
    }

    auto ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    auto contextHandle = static_cast<ze_context_handle_t>(context);
    auto deviceHandle = static_cast<ze_device_handle_t>(device);
    auto execContext = reinterpret_cast<execution_context*>(execCtx);

    if (::ddiTableHandle == nullptr) {
        ::ddiTableHandle = ddiTableHandle;
    }

    // Note: ZE_BIT(4) will be replaced when ZE_GRAPH_FLAG_OPTIMIZE_FOR_DYNAMIC_SHAPES is available
    //       in openvino/level-zero-ext
    const uint32_t flag = (execContext != nullptr && execContext->isOptimizedDynamicStridesSupported) ? ZE_BIT(4) : 0;
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_2,
                              nullptr,
                              ZE_GRAPH_FORMAT_NATIVE,
                              static_cast<size_t>(kernelSize),
                              reinterpret_cast<uint8_t*>(kernel),
                              nullptr /* build flag */,
                              flag};

    auto pfnCreate = ddiTableHandle->pfnCreate2;
    ze_graph_handle_t graphHandle = nullptr;
    auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
    ERROR_HANDLE(result, "Failed to create graph, kern: %p, size: %" PRId64, kernel, kernelSize);

    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
    auto numInputArguments = 0;

    if (logger) {
        logger->debug("Get properties of graph arguments: {0}, kernel: {1}, size: {2}", props.numGraphArgs, kernel,
                      kernelSize);
    }
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;

        ze_graph_argument_property_strides_t strides{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        arg3.pNext = reinterpret_cast<void*>(&strides);
        result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
        ERROR_HANDLE(result, "Failed to get properties of arg: %" PRIu32 " kern: %p size: %" PRId64, index, kernel,
                     kernelSize);

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            numInputArguments++;
        }
    }

    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    if (logger) {
        logger->debug("Initialize graph of kernel: {0} size: {1}", kernel, kernelSize);
    }
    if (commandListHandle != nullptr) {
        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize, kern: %p size: %" PRId64, kernel, kernelSize);
    }

    (*graphMap)[kernel] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);

    if (logger) {
        logger->info("Created graph for kernel: {0}, size: {1}, commandList: {2}", kernel, kernelSize,
                     commandListHandle);
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue, void* execCtx) {
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    if (logger) {
        logger->debug("Create multiple graphs: {0}", numKernels);
    }
    for (int32_t kernelIndex = 0; kernelIndex < numKernels; ++kernelIndex) {
        ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES,
                                nullptr,
                                ZE_GRAPH_FORMAT_NATIVE,
                                static_cast<size_t>(kernelSizes[kernelIndex]),
                                reinterpret_cast<uint8_t*>(kernels[kernelIndex]),
                                nullptr};

        auto contextHandle = static_cast<ze_context_handle_t>(context);
        auto deviceHandle = static_cast<ze_device_handle_t>(device);

        ze_pfnGraphCreate_ext_t pfnCreate = ddiTableHandle->pfnCreate;
        ze_graph_handle_t graphHandle = nullptr;
        auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
        ERROR_HANDLE(result, "Failed to create graph, idx: %" PRId32 ", kern: %p, size: %" PRId64, kernelIndex,
                     kernels[kernelIndex], kernelSizes[kernelIndex]);

        ze_graph_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
        auto numInputArguments = 0;
        if (logger) {
            logger->debug("Get properties of arguments: {0} of idx: {1}", props.numGraphArgs, kernelIndex);
        }
        for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
            ze_graph_argument_properties_3_t arg3{};
            arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
            ERROR_HANDLE(result,
                         "Failed to get properties of arg: %" PRIu32 ", idx: %" PRId32 ", kern: %p, size: %" PRId64,
                         index, kernelIndex, kernels[kernelIndex], kernelSizes[kernelIndex]);

            if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
                numInputArguments++;
            } else {
                break;
            }
        }

        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize, idx: %" PRId32 " kern: %p, size: %" PRId64,
                     kernelIndex, kernels[kernelIndex], kernelSizes[kernelIndex]);

        (*graphMap)[kernels[kernelIndex]] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);
    }

    RETURN_SUCCESS();
}

int32_t set_arguments(uint64_t index, const vpux::HostExec::MemRefDesc& desc, ze_graph_handle_t graphHandle,
                      ze_graph_dditable_ext_t* ddiTableHandle) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_15) {
        // Below is an example implementation.
        // When a new graph ext is available in master, this will be finalized.
        //
        ze_graph_argument_value_tensor_t tensor_value{
                ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_TENSOR, nullptr,
                reinterpret_cast<void*>(reinterpret_cast<uint64_t>(desc.data) + desc.elementByteSize * desc.offset)};

        // Strides information
        ze_graph_argument_value_strides_t tensor_strides = {};
        tensor_strides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
        tensor_strides.pNext = nullptr;
        for (auto dim = 0; dim < desc.dimCount; dim++) {
            // store strides in reverse order
            tensor_strides.userStrides[dim] = static_cast<uint32_t>(desc.strides[(desc.dimCount - 1) - dim]);
        }
        tensor_value.pNext = reinterpret_cast<void*>(&tensor_strides);
        result = ddiTableHandle->pfnSetArgumentValue2(graphHandle, index, &tensor_value);
    } else {
        result = ddiTableHandle->pfnSetArgumentValue(graphHandle, index, desc.data);
    }

    if (logger) {
        logger->info("{0}", desc);
    }
    return result;
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** inputDescs, int32_t numInputs, void** outputDescs, int32_t numOutputs,
                             void* kernelName, void* kernel, int64_t kernelSize, void* context, void* device,
                             void* ddiTable, void** commandList, int64_t commandListIndex, void* commandQueue,
                             void* execCtx) {
    if (logger) {
        const char* kernelNameStr = static_cast<const char*>(kernelName);
        logger->info(
                "Executing graph for kernel {0} at address {1} of size {2} in cmdListIndex {3} and execContext {4}",
                kernelNameStr, kernel, kernelSize, commandListIndex, execCtx);
    }

    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    auto* execContext = reinterpret_cast<execution_context*>(execCtx);
    if (execCtx != nullptr && execContext->isInitialized == false) {
        if (logger) {
            logger->info("Creating event pool for execution context");
        }

        auto deviceHandle = static_cast<ze_device_handle_t>(device);
        auto contextHandle = static_cast<ze_context_handle_t>(context);
        auto result = execContext->initialize(ddiTableHandle, deviceHandle, contextHandle);
        ERROR_HANDLE(result, "Failed to create event pool for execution context");
    }
    auto inputs = reinterpret_cast<vpux::HostExec::MemRefDesc*>(inputDescs);
    auto outputs = reinterpret_cast<vpux::HostExec::MemRefDesc*>(outputDescs);

    if (inputs == nullptr || outputs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid nullpointer, input: %p, output: %p", inputs,
                     outputs);
    }

    if (numInputs <= 0 || numOutputs <= 0) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_SIZE, "Invalid size, inputs: %" PRId32 ", outputs: %" PRId32, numInputs,
                     numOutputs);
    }

    graph_info graphInfo = (*graphMap)[kernel];

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
        if (curCmdListHandle != nullptr) {
            if (useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP) {
                if (curInferenceCountInGroup >= maxInferenceCountPerGroup) {
                    npu_level_zero_submit_commandlist(reinterpret_cast<void**>(&curCmdListHandle), commandQueue,
                                                      nullptr, nullptr, execCtx);
                }
            } else {
                npu_level_zero_submit_commandlist(reinterpret_cast<void**>(&curCmdListHandle), commandQueue, nullptr,
                                                  nullptr, execCtx);
            }
        }

        if (commandlistPool == nullptr) {
            commandlistPool = new std::vector<ze_command_list_handle_t>();
        }

        if (commandlistPool->size() == 0) {
            commandlistPool->resize(maxCmdListCount);
            auto deviceHandle = static_cast<ze_device_handle_t>(device);
            auto contextHandle = static_cast<ze_context_handle_t>(context);
            auto commandQueueGroupOrdinal = intel_npu::zeroUtils::findCommandQueueGroupOrdinal(
                    deviceHandle, ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
            for (uint64_t i = 0; i < maxCmdListCount; ++i) {
                ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC,
                                                                   nullptr, 0};
                ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc,
                                               commandQueueGroupOrdinal, 0};
                ze_command_list_handle_t cmdListHandle = nullptr;
                zeCommandListCreate(contextHandle, deviceHandle, &desc, &cmdListHandle);
                commandlistPool->at(i) = cmdListHandle;
            }
        }
    }
#endif

    if (graphInfo.graphHandle == nullptr) {
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
        if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
            if (curCmdListHandle != nullptr) {
                if (useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP) {
                    if (curInferenceCountInGroup > 0) {
                        npu_level_zero_submit_commandlist(reinterpret_cast<void**>(&curCmdListHandle), commandQueue,
                                                          nullptr, nullptr, execCtx);
                    }
                }
            }
        }
#endif
        // this is required until graph_init function is generated
        auto result =
                npu_level_zero_create_graph(kernel, kernelSize, context, device, ddiTable, nullptr, nullptr, execCtx);

        ERROR_HANDLE(result, "Failed to compile a graph, kern: %p, size: %" PRId64, kernel, kernelSize);

        graphInfo = (*graphMap)[kernel];
        if (graphInfo.graphHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid graph handle, kern: %p, size: %" PRId64, kernel,
                         kernelSize);
        }
    }
    if (graphInfo.numArgs != (numInputs + numOutputs)) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                     "Invalid arguments, kern: %p, size: %" PRId64 ", numArgs: %" PRIu32 ", inputs: %" PRId32
                     ", outputs: %" PRId32,
                     kernel, kernelSize, graphInfo.numArgs, numInputs, numOutputs);
    }

    const auto graphHandle = graphInfo.graphHandle;
    if (logger) {
        logger->debug("Begin setting arguments: {0} of kern: {1}", graphInfo.numArgs, kernel);
    }
    for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
        // Process inputs
        if (index < graphInfo.numInputArgs) {
            ERROR_HANDLE(set_arguments(index, inputs[index], graphHandle, ddiTableHandle),
                         "Failed to set input argument [%" PRIu32 "/%" PRIu32 "] for kern: %p", index,
                         graphInfo.numArgs, kernel);

        } else {
            ERROR_HANDLE(set_arguments(index, outputs[index - graphInfo.numInputArgs], graphHandle, ddiTableHandle),
                         "Failed to set output argument [%" PRIu32 "/%" PRIu32 "] for kern: %p", index,
                         graphInfo.numArgs, kernel);
        }
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = ddiTableHandle->pfnAppendGraphExecute;
    ze_command_list_handle_t commandListHandle = nullptr;

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
        if (useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP) {
            if (curCmdListHandle == nullptr) {
                curCmdListHandle = commandlistPool->at(curCmdListIndex);
            }
            curInferenceCountInGroup += 1;
        } else {
            curCmdListHandle = commandlistPool->at(curCmdListIndex);
        }
        commandListHandle = curCmdListHandle;
    } else {
#endif
        commandListHandle =
                (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    }
#endif

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle");
    }

    ze_event_handle_t waitEvent = nullptr;
    uint32_t numWaitEvents = 0;
    if (execContext != nullptr) {
        const auto mutableCmdListCount = execContext->mutableCommandListIds.size();
        if (static_cast<size_t>(commandListIndex) < mutableCmdListCount) {
            auto id = execContext->mutableCommandListIds[commandListIndex];
            if (id == default_cmdlist_id) {
                waitEvent = execContext->getWaitEvent();
                if (waitEvent != nullptr) {
                    numWaitEvents = 1;
                }
            }
        } else {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index: %" PRId64 ", got: %" PRIu64,
                         commandListIndex, mutableCmdListCount);
        }
    }

    ze_result_t result = ZE_RESULT_SUCCESS;
    if (numWaitEvents > 0 && waitEvent != nullptr) {
        if (logger) {
            logger->debug("Appending a barrier with a WAIT event: {0}, numWaitEvents: {1}", waitEvent, numWaitEvents);
        }
        result = zeCommandListAppendBarrier(commandListHandle, nullptr, numWaitEvents, &waitEvent);
        if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
            result = zeCommandListReset(commandListHandle);
            ERROR_HANDLE(result, "Failed to reset a command list");
            result = zeCommandListAppendBarrier(commandListHandle, nullptr, numWaitEvents, &waitEvent);
            ERROR_HANDLE(result, "Failed to append barrier before graph execute, kern: %p, numWaitEvents: %d", kernel,
                         numWaitEvents);
            result = zeCommandListAppendEventReset(commandListHandle, waitEvent);
        } else {
            ERROR_HANDLE(result, "Failed to append a barrier before graph execute, kern: %p, numWaitEvents: %d", kernel,
                         numWaitEvents);
            result = zeCommandListAppendEventReset(commandListHandle, waitEvent);
        }

        ERROR_HANDLE(result, "Failed to append an event reset waitEvent: %p, numWaitEvents: %d", waitEvent,
                     numWaitEvents);
    }

    result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);

    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset command list: %p, kern: %p", commandListHandle, kernel);
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    }

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if ((result != ZE_RESULT_SUCCESS) && logger) {
        for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
            vpux::HostExec::MemRefDesc desc;
            if (index < graphInfo.numInputArgs) {
                desc = inputs[index];
            } else {
                desc = outputs[index - graphInfo.numInputArgs];
            }
            logger->error("Set argument index({0}) with {1}", index, desc);
        }
    }
#endif

    ERROR_HANDLE(result, "Failed to append graph execute in command list: %p, kern: %p, numWaits: %d",
                 commandListHandle, kernel, numWaitEvents);

#if !defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    const bool isMutableCommandListEnabled = execCtx != nullptr;
#else
    const bool isMutableCommandListEnabled =
            execCtx != nullptr && useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_DEFAULT;
#endif

    if (isMutableCommandListEnabled) {
        if (commandListIndex >= execContext->mutableCommandListIds.size()) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index: %" PRId64 ", got: %" PRIu64,
                         commandListIndex, execContext->mutableCommandListIds.size());
        }

        // Need to increase cmdId for each execute graph call
        // as it is used to index inferences stored in a command list
        uint64_t cmdId = execContext->mutableCommandListIds[commandListIndex]++;
        if (logger) {
            logger->debug("Begin execution ctx arguments: {0} rebinding, kern: {1}", graphInfo.numArgs, kernel);
        }
        for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
            // Process inputs
            if (index < graphInfo.numInputArgs) {
                auto& input = inputs[index];
                // For repeating block use case, inputs from the second iteration will be scratch buffer.
                // so skip those too.
                if (input.networkArgIndex >= graphInfo.numArgs ||
                    (scratchBuffer != nullptr && scratchBuffer->inRange(reinterpret_cast<uint64_t>(input.data)))) {
                    // no need to track this argument as it is not mapped to network argument of main module
                    continue;
                }
                execContext->add_binding(commandListIndex, {cmdId, commandListHandle, input.networkArgIndex, index,
                                                            input.elementByteSize * input.offset});

            } else {
                auto& output = outputs[index - graphInfo.numInputArgs];
                // For repeating block use case, outputs can be from the first iteration to N-1 th iteration.
                // so skip those too.
                if (output.networkArgIndex >= graphInfo.numArgs ||
                    (scratchBuffer != nullptr && scratchBuffer->inRange(reinterpret_cast<uint64_t>(output.data)))) {
                    // no need to track this argument as it is not mapped to network argument of main module
                    continue;
                }
                execContext->add_binding(commandListIndex, {cmdId, commandListHandle, output.networkArgIndex, index,
                                                            output.elementByteSize * output.offset});
            }
        }
    }

    if (logger) {
        logger->info("Executed graph for kernel: {0}", kernel);
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void* commandLists, void* commandQueue, void* fence, void* event, void* execCtx) {
    if (logger) {
        logger->info("Submitting command list: {0}, fence: {1}, event: {2}", commandLists, fence, event);
    }

    ze_command_list_handle_t commandListHandle = nullptr;
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
        commandListHandle = curCmdListHandle;
        curCmdListHandle = nullptr;
        curCmdListIndex = (curCmdListIndex + 1) % maxCmdListCount;
    } else {
#endif
        commandListHandle =
                (commandLists == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandLists);
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    }
#endif

    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    auto fenceHandle = static_cast<ze_fence_handle_t>(fence);
    auto eventHandle = static_cast<ze_event_handle_t>(event);
    auto result = ZE_RESULT_SUCCESS;
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
                     "Invalid commandListHandle for submission, command queue: %p, fence: %p, event: %p", commandQueue,
                     fence, event);
    }

#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
    if (useInternalCmdListMode != USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
        if (fenceHandle != nullptr) {
            curCmdListIndex = 0;
            if (useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_PER_INFERENCE_GROUP) {
                curInferenceCountInGroup = 0;
            }
        }
    }
#endif

    // note commnad queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        if (eventHandle != nullptr) {
            result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            ERROR_HANDLE(result, "Failed to zeCommandListAppendBarrier");

            result = zeCommandListAppendSignalEvent(commandListHandle, eventHandle);
            ERROR_HANDLE(result, "Failed to zeCommandListAppendSignalEvent");

            result = zeCommandListClose(commandListHandle);
            ERROR_HANDLE(result, "Failed to zeCommandListClose");

            result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, nullptr);
            ERROR_HANDLE(
                    result,
                    "Failed to zeCommandQueueExecuteCommandList with event: %p, command queue: %p, command list: %p",
                    event, commandQueue, commandLists);

        } else {
            if (fence == nullptr) {
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
                if (useInternalCmdListMode == USE_INTERNAL_CMDLIST_MODE_DEFAULT) {
#endif
                    // add a barrier at the end of command list to ensure all commands are finished
                    auto execContext = reinterpret_cast<execution_context*>(execCtx);
                    auto signalEvent = (execContext != nullptr) ? execContext->getSignalEvent() : nullptr;
                    if (logger) {
                        logger->debug("Append barrier with SIGNAL event: {0} for command list: {1}", signalEvent,
                                      commandListHandle);
                    }
                    result = zeCommandListAppendBarrier(commandListHandle, signalEvent, 0, nullptr);
                    ERROR_HANDLE(result, "Failed to zeCommandListAppendBarrier");
#if defined(ENABLE_COMMANDLIST_SUBMISSION_MODE)
                } else {
                    // add a barrier at the end of command list to ensure all commands are finished
                    result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
                    ERROR_HANDLE(result, "Failed to zeCommandListAppendBarrier");
                }
#endif
            }
            result = zeCommandListClose(commandListHandle);
            ERROR_HANDLE(result, "Failed to zeCommandListClose");
            result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fenceHandle);
            ERROR_HANDLE(
                    result,
                    "Failed to zeCommandQueueExecuteCommandList with fence: %p, command queue: %p, command list: %p",
                    fence, commandQueue, commandLists);
        }
    }
    if (logger) {
        logger->info("Submitted command list: {0}", commandLists);
    }
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    *pError = lastErrorMessage;
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void** commandLists) {
    if (commandLists == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid commandLists");
    }

    auto commandListHandle = *reinterpret_cast<ze_command_list_handle_t*>(commandLists);

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid commandListHandle");
    }

    auto result = zeCommandListReset(commandListHandle);
    ERROR_HANDLE(result, "Failed to reset a commandlist: %p", commandLists);
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandLists, int32_t numCommandLists) {
    if (logger) {
        logger->debug("Reset command lists: {0}", numCommandLists);
    }
    for (int32_t index = 0; index < numCommandLists; ++index) {
        auto commandListHandle = static_cast<ze_command_list_handle_t>(commandLists[index]);

        if (commandListHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid commandListHandle: [%" PRIu32 "/%" PRIu32 "]",
                         index, numCommandLists);
        }

        auto result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset a commandlist: %p, ind: [%" PRIu32 "/%" PRIu32 "]", commandLists[index],
                     index, numCommandLists);
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* networkMetadata, void* inputDescs,
                                    void* outputDescs) {
    if (metadata == nullptr || networkMetadata == nullptr || inputDescs == nullptr || outputDescs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                     "Invalid argument, meta: %p, netMeta: %p, inputDescs: %p, outputDescs: %p", metadata,
                     networkMetadata, inputDescs, outputDescs);
    }

    auto blob = reinterpret_cast<uint8_t*>(metadata);
    auto deserializedMetadata = elf::MetadataSerialization::deserialize(blob, metadataSize);

    auto input_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(inputDescs);
    auto output_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(outputDescs);

    vpux::NetworkMetadata* network = reinterpret_cast<vpux::NetworkMetadata*>(networkMetadata);

    *network = vpux::VPUMI37XX::getNetworkMetadata(reinterpret_cast<uint8_t*>(metadata), metadataSize);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: The blow coders are from L0 to initialize as L0 driver does not support IRGraph(LLVM w/ ELFs for subgraphs)
    // This will be remove when dynamic model compilation is supported by L0 (CID)
    static std::map<elf::DType, ze_graph_argument_precision_t> precisions = {
            {elf::DType::DType_NOT_SET, ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN},
            {elf::DType::DType_FP64, ZE_GRAPH_ARGUMENT_PRECISION_FP64},
            {elf::DType::DType_FP32, ZE_GRAPH_ARGUMENT_PRECISION_FP32},
            {elf::DType::DType_FP16, ZE_GRAPH_ARGUMENT_PRECISION_FP16},
            {elf::DType::DType_U64, ZE_GRAPH_ARGUMENT_PRECISION_UINT64},
            {elf::DType::DType_U32, ZE_GRAPH_ARGUMENT_PRECISION_UINT32},
            {elf::DType::DType_U16, ZE_GRAPH_ARGUMENT_PRECISION_UINT16},
            {elf::DType::DType_U8, ZE_GRAPH_ARGUMENT_PRECISION_UINT8},
            {elf::DType::DType_U4, ZE_GRAPH_ARGUMENT_PRECISION_UINT4},
            {elf::DType::DType_I64, ZE_GRAPH_ARGUMENT_PRECISION_INT64},
            {elf::DType::DType_I32, ZE_GRAPH_ARGUMENT_PRECISION_INT32},
            {elf::DType::DType_I16, ZE_GRAPH_ARGUMENT_PRECISION_INT16},
            {elf::DType::DType_I8, ZE_GRAPH_ARGUMENT_PRECISION_INT8},
            {elf::DType::DType_I4, ZE_GRAPH_ARGUMENT_PRECISION_INT4},
            {elf::DType::DType_BIN, ZE_GRAPH_ARGUMENT_PRECISION_BIN},
            {elf::DType::DType_I4X, ZE_GRAPH_ARGUMENT_PRECISION_NF4},
            {elf::DType::DType_BFP16, ZE_GRAPH_ARGUMENT_PRECISION_BF16}};

    static std::map<size_t, ze_graph_argument_layout_t> layouts = {
            {0x1, ZE_GRAPH_ARGUMENT_LAYOUT_C},         {0x12, ZE_GRAPH_ARGUMENT_LAYOUT_NC},
            {0x21, ZE_GRAPH_ARGUMENT_LAYOUT_CN},       {0x123, ZE_GRAPH_ARGUMENT_LAYOUT_CHW},
            {0x1234, ZE_GRAPH_ARGUMENT_LAYOUT_NCHW},   {0x1342, ZE_GRAPH_ARGUMENT_LAYOUT_NHWC},
            {0x12345, ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW}, {0x13452, ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC}};

    auto set_properties = [](ze_graph_argument_properties_3_t& properties, elf::TensorRef& tensor_desc,
                             elf::TensorRef& network_desc, elf::OVNode& node, vpux::IODescriptor& io_desc) {
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.name, tensor_desc.name);
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.debug_friendly_name, node.friendly_name);

        if (node.tensor_names_count == 0) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], tensor_desc.name);
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], node.friendly_name);
        }

        for (uint32_t i = 0; i < node.tensor_names_count; i++) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.associated_tensor_names[i], node.tensor_names[i]);

            io_desc.outputTensorNames.insert(node.tensor_names[i]);
        }
        properties.associated_tensor_names_count = node.tensor_names_count;

        memcpy_s(properties.dims, sizeof(properties.dims), tensor_desc.dimensions,
                 tensor_desc.dimensions_size * sizeof(uint32_t));

        properties.dims_count = tensor_desc.dimensions_size;
        properties.networkPrecision = precisions[network_desc.data_type];
        properties.devicePrecision = precisions[tensor_desc.data_type];
        properties.networkLayout = layouts[network_desc.order];
        properties.deviceLayout = layouts[tensor_desc.order];
    };

    auto set_property_strides = [](ze_graph_argument_property_strides_t& strides, elf::OVNode& node) {
        strides = {ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        const auto dynamicDim = std::numeric_limits<uint64_t>::max();
        for (uint32_t index = 0; index < node.shape_size; index++) {
            // store strides in reverse order
            if (node.shape[index] == dynamicDim) {
                strides.supportsDynamicStrides = true;
                break;
            }
        }
    };

    input_descriptors->resize(network->inputs.size());
    for (size_t index = 0; index < input_descriptors->size(); ++index) {
        auto& descriptor = input_descriptors->at(index);
        descriptor.idx = index;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;

        auto tensor = index < deserializedMetadata->mInTensorDescriptors.size()
                              ? deserializedMetadata->mInTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetInputs.size() ? deserializedMetadata->mNetInputs[index]
                                                                   : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVParameters.size() ? deserializedMetadata->mOVParameters[index]
                                                                       : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->inputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    output_descriptors->resize(network->outputs.size());
    const auto inputCount = network->inputs.size();
    for (size_t index = 0; index < output_descriptors->size(); ++index) {
        auto& descriptor = output_descriptors->at(index);
        descriptor.idx = index + inputCount;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_OUTPUT;

        auto tensor = index < deserializedMetadata->mOutTensorDescriptors.size()
                              ? deserializedMetadata->mOutTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetOutputs.size() ? deserializedMetadata->mNetOutputs[index]
                                                                    : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVResults.size() ? deserializedMetadata->mOVResults[index]
                                                                    : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->outputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_execution_context(void* handle, int64_t numSubGraphs, int64_t numNetworkArgs, void** ret) {
    if (logger) {
        logger->info("npu_level_zero_create_execution_context: {0} {1}", numSubGraphs, numNetworkArgs);
    }
    execution_context* context = new execution_context(numSubGraphs, numNetworkArgs);
    *ret = static_cast<void*>(context);

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_execution_context(void* handle, void** commandList, int64_t numCommandLists) {
    if (logger) {
        logger->info("npu_level_zero_reset_execution_context");
    }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    if (context != nullptr) {
        context->reset(commandList, numCommandLists);
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_destroy_execution_context(void* handle) {
    if (logger) {
        logger->info("npu_level_zero_destroy_execution_context");
    }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    if (context != nullptr) {
        delete context;
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_update_mutable_command_list(void* handle, void* networkArgArr, uint64_t networkArgArraySize,
                                           void* argIndexArr, uint64_t argIndexSize) {
    if (logger) {
        logger->info("npu_level_zero_update_mutable_command_list");
    }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    if (context != nullptr) {
        context->resetEvents();
    }
    uint64_t* networkArgArray = reinterpret_cast<uint64_t*>(networkArgArr);
    uint64_t* argIndexArray = reinterpret_cast<uint64_t*>(argIndexArr);
    if (context != nullptr && argIndexArr != nullptr && networkArgArray != nullptr) {
        for (auto& bindingsPerCmdList : context->argumentBindings) {
            for (uint64_t index = 0; index < argIndexSize; ++index) {
                uint64_t argIndex = argIndexArray[index];

                if (argIndex >= networkArgArraySize) {
                    ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument index");
                }

                // Process mutable arguments
                for (auto& binding : bindingsPerCmdList[argIndex]) {
                    const void* bufferPtr = reinterpret_cast<void*>((networkArgArray)[argIndex] + binding.bufferOffset);
                    ze_mutable_graph_argument_exp_desc_t desc = {ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC,
                                                                 nullptr, binding.cmdId,
                                                                 static_cast<uint32_t>(binding.argIndex), bufferPtr};
                    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {
                            ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC, &desc, 0};
                    auto result = zeCommandListUpdateMutableCommandsExp(binding.commandListHandle,
                                                                        &mutable_commands_exp_desc_t);
                    if (logger) {
                        std::ostringstream oss;
                        oss << "Updating mutable argument:" << binding << " with buffer pointer " << std::hex
                            << bufferPtr << std::dec << ", result: " << result;
                        logger->info("{0}", oss.str());
                    }

                    if (result != ZE_RESULT_SUCCESS) {
                        std::ostringstream oss;
                        oss << "Failed to set mutable argument:" << binding << " with buffer pointer " << std::hex
                            << bufferPtr << std::dec;

                        ERROR_HANDLE(result, "%s", oss.str().c_str());
                    }
                }
            }
        }
    } else {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid nullpointer");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_execute_mutable_command_list(void* handle, void* networkArgArr, uint64_t networkArgArraySize,
                                            void* argIndexArr, uint64_t argIndexSize, void* commandLists,
                                            uint64_t numCommandLists, void* commandQueue, void* fence) {
    if (logger) {
        logger->info("npu_level_zero_execute_mutable_command_list");
    }

    if (handle == nullptr || argIndexArr == nullptr || networkArgArr == nullptr || commandLists == nullptr ||
        commandQueue == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid nullpointer");
    }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    ze_command_list_handle_t* cmdLists = reinterpret_cast<ze_command_list_handle_t*>(commandLists);
    uint64_t* networkArgArray = reinterpret_cast<uint64_t*>(networkArgArr);
    uint64_t* argIndexArray = reinterpret_cast<uint64_t*>(argIndexArr);

    if (numCommandLists < context->argumentBindings.size()) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Number of command lists does not match with execution context");
    }

    auto commandQueueHandle = reinterpret_cast<ze_command_queue_handle_t>(commandQueue);
    auto fenceHandle = reinterpret_cast<ze_fence_handle_t>(fence);
    for (size_t i = 0; i < context->argumentBindings.size(); ++i) {
        auto& bindingsPerCmdList = context->argumentBindings[i];

        bool updatedCmdLists = false;
        for (uint64_t index = 0; index < argIndexSize; ++index) {
            uint64_t argIndex = argIndexArray[index];

            if (argIndex >= networkArgArraySize || argIndex >= context->argumentBindings[i].size()) {
                ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument index");
            }
            // Process mutable arguments
            for (auto& binding : bindingsPerCmdList[argIndex]) {
                const void* bufferPtr = reinterpret_cast<void*>((networkArgArray)[argIndex] + binding.bufferOffset);
                ze_mutable_graph_argument_exp_desc_t desc = {ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC, nullptr,
                                                             binding.cmdId, static_cast<uint32_t>(binding.argIndex),
                                                             bufferPtr};
                ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {
                        ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC, &desc, 0};
                auto result =
                        zeCommandListUpdateMutableCommandsExp(binding.commandListHandle, &mutable_commands_exp_desc_t);
                if (logger) {
                    std::ostringstream oss;
                    oss << "Updating mutable argument:" << binding << " with buffer pointer " << std::hex << bufferPtr
                        << std::dec << ", result: " << result;
                    logger->info("{0}", oss.str());
                }

                if (result != ZE_RESULT_SUCCESS) {
                    std::ostringstream oss;
                    oss << "Failed to set mutable argument:" << binding << " with buffer pointer " << std::hex
                        << bufferPtr << std::dec;

                    ERROR_HANDLE(result, "%s", oss.str().c_str());
                }
                updatedCmdLists = true;
            }
        }

        if (updatedCmdLists) {
            auto result = zeCommandListClose(cmdLists[i]);
            ERROR_HANDLE(result, "Failed to zeCommandListClose with command list: %p", cmdLists[i]);
        }

        ze_fence_handle_t usedFence = i == context->argumentBindings.size() - 1 ? fenceHandle : nullptr;

        auto result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &cmdLists[i], usedFence);
        ERROR_HANDLE(result, "Failed to zeCommandQueueExecuteCommandList with command queue: %p, command list: %p",
                     commandQueue, cmdLists[i]);
    }

    RETURN_SUCCESS();
}
#endif
