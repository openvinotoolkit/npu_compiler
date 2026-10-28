//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_execution.hpp"
#include <ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>
#include "intel_npu/config/config.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace intel_npu::vm;

namespace {

result_t success() {
    return ZE_RESULT_SUCCESS;
}

template <typename... Args>
void failureLogImpl(std::string_view format, Args... args) {
    NPU_VM_LOG_ERROR(format, args...);
}

template <typename... Args>
result_t failure(result_t errorCode, const std::string& format, Args... args) {
    failureLogImpl(format + ": {}", args..., errorCode);
    return errorCode;
}

int32_t setArguments(uint64_t index, const BufferMapperItem& desc, ze_graph_handle_t graphHandle,
                     ze_graph_dditable_ext_t* ddiTableHandle, bool supportsDynamicStrides) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    auto buffer = desc.first;
    auto bufferMetadata = desc.second;

    // where to get buffer offset
    uint64_t offsetInElements = 0;
    const auto byteOffset = static_cast<std::ptrdiff_t>(bufferMetadata->elemByteSize * offsetInElements);
    auto address = std::next(buffer->getData(), byteOffset);

    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_15) {
        // Below is an example implementation.
        // When a new graph ext is available in master, this will be finalized.

        ze_graph_argument_value_tensor_t tensorValue{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_TENSOR, nullptr, address};

        // Only attach strides when the driver reported this graph argument expects dynamic strides;
        // otherwise the argument is bound without stride metadata.
        // tensorStrides must be declared here, outside the `if` below, so it stays alive until
        // pfnSetArgumentValue2 is called
        ze_graph_argument_value_strides_t tensorStrides = {};
        if (supportsDynamicStrides) {
            tensorStrides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
            tensorStrides.pNext = nullptr;
            auto& strides = bufferMetadata->strides;
            if (strides.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
                return failure(ZE_RESULT_ERROR_INVALID_SIZE,
                               "Buffer rank {} for argument {} exceeds ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE ({})",
                               strides.size(), index, ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
            }
            const size_t dimCount = strides.size();
            for (size_t dim = 0; dim < dimCount; dim++) {
                // store strides in reverse order
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index) - index values are guaranteed above
                tensorStrides.userStrides[dim] =
                        static_cast<uint32_t>(bufferMetadata->strides.at((dimCount - 1) - dim));
            }
            tensorValue.pNext = static_cast<void*>(&tensorStrides);
        }
        result = ddiTableHandle->pfnSetArgumentValue2(graphHandle, index, &tensorValue);
    } else {
        result = ddiTableHandle->pfnSetArgumentValue(graphHandle, index, address);
    }

    NPU_VM_LOG_TRACE("setArguments index={}, address={}, buffer={}, metadata={}, elemByteSize={}, "
                     "supportsDynamicStrides={}",
                     index, address, static_cast<void*>(buffer), static_cast<void*>(bufferMetadata),
                     bufferMetadata->elemByteSize, supportsDynamicStrides);
    return result;
}

result_t createCmdList(ze_context_handle_t context, ze_device_handle_t device, uint32_t groupOrdinal,
                       ze_command_list_handle_t& handle) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc, groupOrdinal, 0};
    return zeCommandListCreate(context, device, &desc, &handle);
}

}  // namespace

namespace intel_npu::vm {

result_t ExecutionContext::CmdListPool::initialize(uint32_t cmdListCount, ze_context_handle_t context,
                                                   ze_device_handle_t device, uint32_t groupOrdinal) {
    for (uint32_t i = 0; i < _cmdLists.size(); ++i) {
        if (_cmdLists.at(i) != nullptr) {
            auto result = zeCommandListDestroy(_cmdLists.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to destroy command lists");
            }
        }
    }

    _cmdLists.resize(cmdListCount, nullptr);
    for (uint32_t i = 0; i < cmdListCount; ++i) {
        auto result = ::createCmdList(context, device, groupOrdinal, _cmdLists.at(i));
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create a fence");
        }
    }

    return success();
}

void ExecutionContext::CmdListPool::destroy() {
    for (auto cmdList : _cmdLists) {
        if (cmdList != nullptr) {
            auto result = zeCommandListDestroy(cmdList);
            if (result != ZE_RESULT_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to destroy command list during move assignment");
            }
        }
    }
    _cmdLists.clear();
}

result_t ExecutionContext::createCmdList(ze_command_list_handle_t& cmdListhandle) {
    uint32_t currentCmdListIndex = _nextCmdListIndex;

    // If the concurrent job count is greater than 1,
    // we need to create multiple command lists for concurrent execution.
    for (uint32_t i = 0; i < _concurrentJobCount; ++i) {
        // The round index is used internally to determine which command list
        // to use for the current execution
        auto handle = getCmdList(currentCmdListIndex);
        if (handle != nullptr) {
            // stores it to the recorded command lists for reuse in future executions
            // when no tensor shape changes are detected and the recorded command lists
            // can be reused for execution.
            _recordedCommandLists.push_back(handle);
        } else {
            return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }
    // updates the current command list index to the next command list index
    _currentCmdListIndex = _nextCmdListIndex;

    if (_concurrentJobCount > 1) {
        // return nullptr here
        // the caller should use getCmdList() to get the command list handle for execution
        cmdListhandle = nullptr;
    } else {
        cmdListhandle = _recordedCommandLists.at(_currentCmdListIndex);
    }

    // increases the next command list index
    _nextCmdListIndex += _concurrentJobCount;
    return success();
}

ze_event_handle_t ExecutionContext::getSignalEvent() {
    if (_events.empty()) {
        return nullptr;
    }

    if (_curEventIndex < _events.size()) {
        ze_event_handle_t eventHandle = _events.at(_curEventIndex++);

        if (_signalEventsArrayIndex >= _signalEvents.size()) {
            _signalEvents.push_back(std::vector<ze_event_handle_t>());
        }

        // stores the signal event handle to the signal events array for synchronization
        _signalEvents[_signalEventsArrayIndex].push_back(eventHandle);
        return eventHandle;
    }

    return nullptr;
}

std::vector<ze_event_handle_t>& ExecutionContext::getWaitEvents() {
    if (_signalEventsArrayIndex == 1) {
        if (_signalEventsArrayIndex >= _signalEvents.size()) {
            _signalEvents.push_back(std::vector<ze_event_handle_t>());
        }
    }

    if (_signalEventsArrayIndex > 0 && _signalEventsArrayIndex < _signalEvents.size()) {
        // returns the signal events from the previous signal events array index
        // as wait events for synchronization
        return _signalEvents[_signalEventsArrayIndex - 1];
    }

    return _signalEvents.back();
}

void ExecutionContext::destroySyncObjects() {
    // Release currently owned resources (if any)
    for (auto& event : _events) {
        auto result = zeEventDestroy(event);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy event during move assignment");
        }
    }
    _events.clear();

    if (_eventPool != nullptr) {
        auto result = zeEventPoolDestroy(_eventPool);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy event pool during move assignment");
        }
        _eventPool = nullptr;
    }

    if (_internalHostSyncObjectType == HostSyncOjbectType::FENCE && _hostSyncFence != nullptr) {
        auto result = zeFenceDestroy(_hostSyncFence);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy host sync fence during destruction");
        }
        _hostSyncFence = nullptr;
    }
}

ExecutionContext::ExecutionContext(ExecutionContext&& other) noexcept
        : _nextCmdListIndex(other._nextCmdListIndex),
          _currentCmdListIndex(other._currentCmdListIndex),
          _cmdListRoundIndex(other._cmdListRoundIndex),
          _cmdListCount(other._cmdListCount),
          _concurrentJobCount(other._concurrentJobCount),
          _curEventIndex(other._curEventIndex),
          _events(std::move(other._events)),
          _eventPool(other._eventPool),
          _inferenceCmdIds(std::move(other._inferenceCmdIds)),
          _executionContextType(other._executionContextType),
          _executionMode(other._executionMode),
          _internalHostSyncObjectType(other._internalHostSyncObjectType),
          _hostSyncFence(other._hostSyncFence),
          _hostSyncEvent(other._hostSyncEvent),
          _executeParamsV1(other._executeParamsV1),
          _executeParamsV2(other._executeParamsV2),
          _recordedCommandLists(std::move(other._recordedCommandLists)),
          _signalEventsArrayIndex(other._signalEventsArrayIndex),
          _context(other._context),
          _device(other._device),
          _commandQueue(other._commandQueue),
          _graphDdiTable(other._graphDdiTable),
          _commandQueueDdiTableExt(other._commandQueueDdiTableExt),
          _executionObjectsInitialized(other._executionObjectsInitialized),
          _optimizedDynamicStridesSupported(other._optimizedDynamicStridesSupported),
          _recordedCmdListReusable(other._recordedCmdListReusable) {
    _cmdListPool._cmdLists = std::move(other._cmdListPool._cmdLists);
    other._nextCmdListIndex = 0;
    other._currentCmdListIndex = 0;
    other._cmdListCount = 0;
    other._eventPool = nullptr;
    other._curEventIndex = 0;
    other._executionObjectsInitialized = false;
    other._optimizedDynamicStridesSupported = false;
    other._hostSyncFence = nullptr;
    other._hostSyncEvent = nullptr;
    other._context = nullptr;
    other._device = nullptr;
    other._commandQueue = nullptr;
    other._graphDdiTable = nullptr;
    other._commandQueueDdiTableExt = nullptr;
}

ExecutionContext& ExecutionContext::operator=(ExecutionContext&& other) noexcept {
    if (this != &other) {
        _nextCmdListIndex = other._nextCmdListIndex;
        _currentCmdListIndex = other._currentCmdListIndex;
        _cmdListRoundIndex = other._cmdListRoundIndex;
        _cmdListCount = other._cmdListCount;
        _concurrentJobCount = other._concurrentJobCount;
        destroySyncObjects();
        _events = std::move(other._events);
        _eventPool = other._eventPool;

        _curEventIndex = other._curEventIndex;
        _executionObjectsInitialized = other._executionObjectsInitialized;
        _inferenceCmdIds = std::move(other._inferenceCmdIds);
        _optimizedDynamicStridesSupported = other._optimizedDynamicStridesSupported;
        _recordedCmdListReusable = other._recordedCmdListReusable;
        _cmdListPool.destroy();
        _cmdListPool._cmdLists = std::move(other._cmdListPool._cmdLists);
        _internalHostSyncObjectType = other._internalHostSyncObjectType;
        _hostSyncFence = other._hostSyncFence;
        _hostSyncEvent = other._hostSyncEvent;
        _executeParamsV1 = other._executeParamsV1;
        _executeParamsV2 = other._executeParamsV2;
        _executionContextType = other._executionContextType;
        _executionMode = other._executionMode;
        _recordedCommandLists = std::move(other._recordedCommandLists);
        _signalEventsArrayIndex = other._signalEventsArrayIndex;
        _context = other._context;
        _device = other._device;
        _commandQueue = other._commandQueue;
        _graphDdiTable = other._graphDdiTable;
        _commandQueueDdiTableExt = other._commandQueueDdiTableExt;
        _executeParamsV1 = other._executeParamsV1;
        _executeParamsV2 = other._executeParamsV2;
        _executionContextType = other._executionContextType;
        _executionMode = other._executionMode;

        _context = other._context;
        _device = other._device;
        _commandQueue = other._commandQueue;
        _graphDdiTable = other._graphDdiTable;
        _commandQueueDdiTableExt = other._commandQueueDdiTableExt;

        other._nextCmdListIndex = 0;
        other._cmdListCount = 0;
        other._eventPool = nullptr;
        other._curEventIndex = 0;
        other._executionObjectsInitialized = false;
        other._optimizedDynamicStridesSupported = false;

        other._hostSyncFence = nullptr;
        other._hostSyncEvent = nullptr;

        other._context = nullptr;
        other._device = nullptr;
        other._commandQueue = nullptr;
        other._graphDdiTable = nullptr;
        other._commandQueueDdiTableExt = nullptr;
    }
    return *this;
}

ExecutionContext::~ExecutionContext() {
    destroySyncObjects();
}

void ExecutionContext::initialize(ExecutionContextType type, uint32_t cmdListCount, uint32_t /*numNetworkArgs*/,
                                  size_t concurrentJobCount) {
    // release resources
    destroySyncObjects();
    _cmdListPool.destroy();

    _executionContextType = type;
    _cmdListCount = cmdListCount;
    _concurrentJobCount = concurrentJobCount;
    _executionMode = (_concurrentJobCount > 1) ? ExecutionMode::TILE_AGNOSTIC : ExecutionMode::TILE_BOUND;
}

result_t ExecutionContext::initializeExecutionObjects(ze_context_handle_t context, ze_device_handle_t device,
                                                      ze_command_queue_handle_t commandQueue,
                                                      ze_graph_dditable_ext_t* ddiTable,
                                                      ze_command_queue_npu_dditable_ext_t* cmdQueueDdiTable,
                                                      bool /*useInternalCmdQueue*/) {
    if (ddiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for ddi table");
    }
    if (device == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for device handle");
    }

    _optimizedDynamicStridesSupported =
            (ddiTable->pfnCompilerIsOptionSupported != nullptr) &&
            (ddiTable->pfnCompilerIsOptionSupported(device, ZE_NPU_DRIVER_OPTIONS, "OPTIMIZED_DYNAMIC_STRIDES",
                                                    nullptr) == ZE_RESULT_SUCCESS);

    // Initialize the execution context with the provided context, device, command queue, and DDI tables.
    _graphDdiTable = ddiTable;
    _commandQueueDdiTableExt = cmdQueueDdiTable;
    _device = device;
    _context = context;
    _commandQueue = commandQueue;

    // recorded command list can be reused for execution
    // if the tensor shapes do not change between executions
    // If command list pool or immediate command list are used
    // it will be false
    _recordedCmdListReusable = true;
#if defined(VPUX_DEVELOPER_BUILD)
    const char* envVar = std::getenv("NPU_VM_RUNTIME_FORCE_CMD_LIST_RECORDING");
    if (envVar != nullptr) {
        _recordedCmdListReusable = OptionParser<bool>::parse(envVar);
    }
#endif

    // Set the internal host synchronization object type based on the execution context type.
    // It should be set before creating the event pool, as the event pool creation may depend on the host sync object
    // type.
    if (_executionContextType == ExecutionContextType::V2) {
        _internalHostSyncObjectType =
                (_recordedCmdListReusable) ? HostSyncOjbectType::FENCE : HostSyncOjbectType::EVENT;
    }

    auto result = createEventPool(device, context);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to create event pool for execution context");
    }

    // V2 requires to create command lists and sync objects internally
    if (_executionContextType == ExecutionContextType::V2) {
        auto totalCmdLists = _cmdListCount * _concurrentJobCount;
        _cmdListPool.initialize(totalCmdLists, context, device, 0);
        _inferenceCmdIds.assign(totalCmdLists, DEFAULT_INFERENCE_ID);
        result = createHostSyncObject(commandQueue);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create host sync object for execution context");
        }
    } else {
        _inferenceCmdIds.assign(_cmdListCount, DEFAULT_INFERENCE_ID);
    }
    _executionObjectsInitialized = true;

    return success();
}

result_t ExecutionContext::createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context) {
    auto eventCount = (_cmdListCount > 1) ? (_cmdListCount - 1) * _concurrentJobCount : 0;
    if (_internalHostSyncObjectType == HostSyncOjbectType::EVENT) {
        // event will b used for host sync, so need to create an additional event
        eventCount += 1;
    }

    if (eventCount > 0) {
        ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                              ZE_EVENT_POOL_FLAG_HOST_VISIBLE, static_cast<uint32_t>(eventCount)};
        auto result = zeEventPoolCreate(context, &eventPoolDesc, /*numDevices*/ 1, &deviceHandle, &_eventPool);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create event pool for execution context");
        }

        _events.resize(eventCount);
        for (uint32_t i = 0; i < eventCount; i++) {
            ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, 0, 0};
            result = zeEventCreate(_eventPool, &eventDesc, &_events.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to create event");
            }
        }
    }

    return ZE_RESULT_SUCCESS;
}

result_t ExecutionContext::prepareSubmission(ze_command_list_handle_t commandListHandle, bool needHostSync) {
    result_t result = ZE_RESULT_SUCCESS;
    if (commandListHandle == nullptr) {
        // Need to reset the command list round index to 0
        // This is because we are starting a new submission and need to use the first command list
        _cmdListRoundIndex = 0;
        auto cmdListIndex = getCurrentCmdListIndex();

        const bool hostEventSyncRequired = needHostSync && _hostSyncEvent != nullptr;
        // No host sync means add sync objects to make sure that execution of the current command lists completes.
        if ((!needHostSync || _hostSyncFence == nullptr) && _concurrentJobCount > 1) {
            for (uint32_t i = 0; i < _concurrentJobCount; ++i) {
                auto cmdListHandle = getCmdList(cmdListIndex);
                if (cmdListHandle == nullptr) {
                    return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle for submission");
                }

                // Need to add signal event to the command list to ensure that the host sync event is signaled
                // after all command lists complete execution
                auto signalEvent = getSignalEvent();
                result = zeCommandListAppendBarrier(cmdListHandle, signalEvent, 0, nullptr);
                if (result != ZE_RESULT_SUCCESS) {
                    return failure(result, "Failed to append barrier to command list");
                }

                if (!hostEventSyncRequired) {
                    // no additional record required
                    auto result = zeCommandListClose(cmdListHandle);
                    if (result != ZE_RESULT_SUCCESS) {
                        return failure(result, "Failed to zeCommandListClose");
                    }
                }
            }

            // requires to add a barrier to the command lists to ensure that the host sync event is signaled
            // after all command lists complete execution
            if (hostEventSyncRequired) {
                // increase the signal events array index to ensure the current submission uses
                // a set of signal events as wait events
                increaseSignalEventsArrayIndex();

                auto& waitEvents = getWaitEvents();
                auto cmdListHandle = getCmdList(cmdListIndex);
                if (cmdListHandle == nullptr) {
                    return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle for submission");
                }

                result =
                        zeCommandListAppendBarrier(cmdListHandle, _hostSyncEvent, waitEvents.size(), waitEvents.data());
                if (result != ZE_RESULT_SUCCESS) {
                    return failure(result, "Failed to append barrier to command list");
                }

                auto result = zeCommandListClose(cmdListHandle);
                if (result != ZE_RESULT_SUCCESS) {
                    return failure(result, "Failed to zeCommandListClose");
                }
            }
        }
    } else {
        if (!needHostSync) {
            // Need to add signal event to the command list to ensure that the host sync event is signaled
            // after all commands complete execution
            auto signalEvent = getSignalEvent();
            result = zeCommandListAppendBarrier(commandListHandle, signalEvent, 0, nullptr);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to append barrier to command list");
            }
        } else if (_hostSyncEvent != nullptr) {
            result = zeCommandListAppendBarrier(commandListHandle, _hostSyncEvent, 0, nullptr);
        }

        auto result = zeCommandListClose(commandListHandle);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to zeCommandListClose");
        }
    }

    // increase the signal events array index to ensure the current submission uses
    // a set of signal events as wait events
    increaseSignalEventsArrayIndex();

    return result;
}

result_t ExecutionContext::hostSync() {
    // No need to synchronize host with fence or event if the execution parameters V1 are set
    if (_executeParamsV2) {
        if (_hostSyncFence != nullptr) {
            auto result = zeFenceHostSynchronize(_hostSyncFence, UINT64_MAX);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to synchronize host with fence");
            }
        } else if (_hostSyncEvent != nullptr) {
            auto result = zeEventHostSynchronize(_hostSyncEvent, UINT64_MAX);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to synchronize host with event");
            }
        }
    }
    return ZE_RESULT_SUCCESS;
}

result_t createKernel(ExecutionContext& executionContext, uint8_t* kernelBlob, size_t kernelBlobSize,
                      const std::string& kernelName, ze_graph_handle_t& graphHandle, KernelInfo& graphInfo) {
    NPU_VM_LOG_TRACE("Creating graph for kernel {} at address {} of size {}", kernelName, kernelBlob, kernelBlobSize);

    const uint32_t flag = ZE_GRAPH_FLAG_OPTIMIZE_FOR_DYNAMIC_SHAPES;
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_2,
                              nullptr,
                              ZE_GRAPH_FORMAT_NATIVE,
                              static_cast<size_t>(kernelBlobSize),
                              kernelBlob,
                              nullptr /* build flag */,
                              flag};

    auto ddiTableHandle = executionContext.getGraphDdiTable();
    auto context = executionContext.getContext();
    auto device = executionContext.getDevice();

    if (ddiTableHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for graph ddi table");
    }
    if (context == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for context");
    }
    if (device == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for device");
    }

    auto pfnCreate = ddiTableHandle->pfnCreate2;
    auto result = pfnCreate(context, device, &desc, &graphHandle);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to create graph, kern: {}, size: {}", kernelBlob, kernelBlobSize);
    }

    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to get graph properties");
    }
    auto numInputArguments = 0;
    std::vector<bool> supportsDynamicStrides(props.numGraphArgs, false);

    NPU_VM_LOG_TRACE("Get properties of graph arguments: {}, kernel: {}, size: {}", props.numGraphArgs, kernelBlob,
                     kernelBlobSize);

    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;

        ze_graph_argument_property_strides_t strides{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        arg3.pNext = static_cast<void*>(&strides);
        result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to get properties of arg: {}, kern: {}, size: {}", index, kernelBlob,
                           kernelBlobSize);
        }

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            numInputArguments++;
        }

        supportsDynamicStrides[index] = strides.supportsDynamicStrides;
        NPU_VM_LOG_DEBUG("Graph arg {} of kernel {} supportsDynamicStrides={}", index, kernelName,
                         strides.supportsDynamicStrides);
    }

    graphInfo = KernelInfo(graphHandle, props.numGraphArgs, numInputArguments, kernelName,
                           std::move(supportsDynamicStrides));

    NPU_VM_LOG_TRACE("Created graph for kernel: {}, size: {}, graph_handle: {}", kernelBlob, kernelBlobSize,
                     graphHandle);

    return success();
}

result_t setBindings(ExecutionContext& executionContext, ze_graph_handle_t graphHandle,
                     const std::vector<BufferMapperItem>& inputs, const std::vector<BufferMapperItem>& outputs,
                     KernelInfo& graphInfo) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto context = executionContext.getContext();
    auto device = executionContext.getDevice();
    auto ddiTable = executionContext.getGraphDdiTable();
    auto cmdQueue = executionContext.getCommandQueue();
    if (!executionContext.executionObjectsInitialized()) {
        // Initialization will be done here for VM runtime V1
        NPU_VM_LOG_TRACE("Creating event pool for execution context");
        // V1 does not support internal command queue, so we set useInternalCmdQueue to false
        const bool useInternalCmdQueue = false;
        auto result = executionContext.initializeExecutionObjects(context, device, cmdQueue, ddiTable, nullptr,
                                                                  useInternalCmdQueue);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to initialize an execution context");
        }
    }

    if (inputs.empty() || outputs.empty()) {
        return failure(ZE_RESULT_ERROR_INVALID_SIZE, "Invalid size, inputs: {}, outputs: {}", inputs.size(),
                       outputs.size());
    }

    if ((inputs.size() + outputs.size()) != graphInfo.getNumArgs()) {
        return failure(
                ZE_RESULT_ERROR_INVALID_SIZE,
                "The total number of inputs and outputs does not match the number of graph arguments, numInputs: "
                "{} , numOutputs: {} , numGraphArgs: {}",
                inputs.size(), outputs.size(), graphInfo.getNumArgs());
    }

    NPU_VM_LOG_TRACE("Begin setting arguments: {} of kern: {} and kern name: {}", graphInfo.getNumArgs(), graphHandle,
                     graphInfo.getKernelName());

    for (uint32_t index = 0; index < graphInfo.getNumArgs(); ++index) {
        // Process inputs
        if (index < graphInfo.getNumInputArgs()) {
            auto result = setArguments(index, inputs.at(index), graphHandle, ddiTable,
                                       graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set input argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        } else {
            auto result = setArguments(index, outputs.at(index - graphInfo.getNumInputArgs()), graphHandle, ddiTable,
                                       graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set output argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        }
    }

    return success();
}

result_t executeGraph(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                      ze_graph_handle_t graphHandle, void* kernelName) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto ddiTableHandle = executionContext.getGraphDdiTable();
    auto cmdListIndex = executionContext.getCurrentCmdListIndex();

    if (commandListHandle == nullptr) {
        commandListHandle = executionContext.getCmdList(cmdListIndex);
    }

    if (kernelName == nullptr) {
        NPU_VM_LOG_TRACE("Executing graph for unnamed kernel at handle {}", graphHandle);
    } else {
        NPU_VM_LOG_TRACE("Executing graph for kernel {} at handle {} in cmdList {} and execContext {}",
                         static_cast<const char*>(kernelName), graphHandle, commandListHandle, &executionContext);
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = ddiTableHandle->pfnAppendGraphExecute;
    std::vector<ze_event_handle_t>& waitEvents = executionContext.getWaitEvents();
    auto& inferenceCmdIds = executionContext.getInferenceCmdIds();
    const auto inferenceCmdCount = inferenceCmdIds.size();
    bool waitSyncRequired = false;
    if (static_cast<size_t>(cmdListIndex) < inferenceCmdCount) {
        auto id = inferenceCmdIds.at(cmdListIndex);
        if (id == DEFAULT_INFERENCE_ID) {
            // For the first inference execution in the command list,
            // wait events need to be added to ensure that the execution of the current command list start
            // after the previous command list execution
            waitSyncRequired = waitEvents.size() > 0;
        }
    } else {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index: {}, got: {}", cmdListIndex,
                       inferenceCmdCount);
    }

    ze_result_t result = ZE_RESULT_SUCCESS;
    if (waitSyncRequired == true) {
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, waitEvents.size(),
                                       waitEvents.data());
        if (executionContext.getConcurrentJobCount() == 1 && waitEvents.size() == 1) {
            result = zeCommandListAppendEventReset(commandListHandle, waitEvents[0]);
        }
    } else {
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    }

    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}",
                       commandListHandle, graphHandle, waitEvents.size());
    }

    executionContext.increaseInferenceCmdId(cmdListIndex);

    NPU_VM_LOG_TRACE("Executed graph for kernel: {}", graphHandle);

    return success();
}

result_t submitCmdList(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                       bool needHostSync) {
    auto commandQueueHandle = executionContext.getCommandQueue();
    ze_fence_handle_t fence = nullptr;
    ze_event_handle_t event = nullptr;

    if (needHostSync) {
        fence = executionContext.getHostSyncFence();
        event = executionContext.getHostSyncEvent();

        if (event != nullptr && fence != nullptr) {
            // Both event and fence are provided, which is not expected
            // as they serve similar synchronization purposes.
            return failure(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                           "Both event and fence are not nullptr(event: {}, fence: {})", event, fence);
        }
    }

    NPU_VM_LOG_TRACE("Submitting command list: {}, fence: {}, event: {}", commandListHandle, fence, event);

    // prepare for submission
    executionContext.prepareSubmission(commandListHandle, needHostSync);

    auto commandQueue = executionContext.getCommandQueue();
    auto concurrentJobCount = executionContext.getConcurrentJobCount();
    // note command queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        auto result = ZE_RESULT_SUCCESS;
        if (commandListHandle) {
            // tile bound execution will pass not null commandListHandle
            result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fence);
        } else {
            // tile agnostic execution will pass null commandListHandle
            // need to get the command list array from the execution context
            result = zeCommandQueueExecuteCommandLists(commandQueueHandle, concurrentJobCount,
                                                       executionContext.getCmdListArrayPtr(), fence);
        }
        if (result != ZE_RESULT_SUCCESS) {
            return failure(
                    result,
                    "Failed to zeCommandQueueExecuteCommandList with event: {}, fence: {}, command queue: {}, command "
                    "list: {}",
                    event, fence, commandQueue, commandListHandle);
        }
    }

    NPU_VM_LOG_TRACE("Submitted command list: {}", commandListHandle);

    return success();
}

}  // namespace intel_npu::vm
