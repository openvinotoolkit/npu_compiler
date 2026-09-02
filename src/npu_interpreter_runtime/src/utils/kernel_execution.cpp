//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_execution.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_interpreter_runtime/npu_vm_runtime.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#include <cstddef>
#include <cstdint>
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

ze_command_list_handle_t getCmdList(const npu_vm_runtime_execute_params_t& params, uint64_t cmdListIndex) {
    if (cmdListIndex < params.numCommandLists && params.commandLists != nullptr) {
        return *std::next(params.commandLists, static_cast<std::ptrdiff_t>(cmdListIndex));
    }
    return nullptr;
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

}  // namespace

namespace intel_npu::vm {

ze_command_list_handle_t ExecutionContext::createCommandList(const npu_vm_runtime_execute_params_t& params) {
    if (_nextCmdListIndex < params.numCommandLists) {
        auto cmdListHandle = getCmdList(params, _nextCmdListIndex);
        if (cmdListHandle != nullptr) {
            _nextCmdListIndex++;
            return cmdListHandle;
        }
    }
    return nullptr;
}

ze_event_handle_t ExecutionContext::getSignalEvent() {
    if (_events.empty()) {
        return nullptr;
    }

    if (_curEventIndex < _events.size()) {
        _signalEventCount++;
        return _events.at(_curEventIndex);
    }

    return nullptr;
}

// Returns the next event handle for synchronization, and increments the event index if a signal event has been
// used.
ze_event_handle_t ExecutionContext::getWaitEvent() {
    if (_events.empty() || (_signalEventCount == 0)) {
        return nullptr;
    }

    _signalEventCount = 0;
    if (_curEventIndex < _events.size()) {
        return _events.at(_curEventIndex++);
    }
    return nullptr;
}

ExecutionContext::ExecutionContext(size_t numCmdLists, size_t /*numNetworkArgs*/)
        : _numCmdLists(numCmdLists), _inferenceCmdIds(numCmdLists) {
    // In a more complete implementation, the constructor would use the numSubGraphs and numNetworkArgs parameters
    // to set up the execution context appropriately. For this initial implementation, we will simply ignore them.
}

ExecutionContext::ExecutionContext(ExecutionContext&& other) noexcept
        : _nextCmdListIndex(other._nextCmdListIndex),
          _numCmdLists(other._numCmdLists),
          _events(std::move(other._events)),
          _eventPool(other._eventPool),
          _curEventIndex(other._curEventIndex),
          _signalEventCount(other._signalEventCount),
          _isInitialized(other._isInitialized),
          _inferenceCmdIds(std::move(other._inferenceCmdIds)),
          _optimizedDynamicStridesSupported(other._optimizedDynamicStridesSupported) {
    other._nextCmdListIndex = 0;
    other._numCmdLists = 0;
    other._eventPool = nullptr;
    other._curEventIndex = 0;
    other._signalEventCount = 0;
    other._isInitialized = false;
    other._optimizedDynamicStridesSupported = false;
}

ExecutionContext& ExecutionContext::operator=(ExecutionContext&& other) noexcept {
    if (this != &other) {
        // Release currently owned resources (if any)
        if (_eventPool != nullptr) {
            for (auto& event : _events) {
                zeEventDestroy(event);
            }
            zeEventPoolDestroy(_eventPool);
        }

        _nextCmdListIndex = other._nextCmdListIndex;
        _numCmdLists = other._numCmdLists;
        _events = std::move(other._events);
        _eventPool = other._eventPool;
        _curEventIndex = other._curEventIndex;
        _signalEventCount = other._signalEventCount;
        _isInitialized = other._isInitialized;
        _inferenceCmdIds = std::move(other._inferenceCmdIds);
        _optimizedDynamicStridesSupported = other._optimizedDynamicStridesSupported;

        other._nextCmdListIndex = 0;
        other._numCmdLists = 0;
        other._eventPool = nullptr;
        other._curEventIndex = 0;
        other._signalEventCount = 0;
        other._isInitialized = false;
        other._optimizedDynamicStridesSupported = false;
    }
    return *this;
}

ExecutionContext::~ExecutionContext() {
    if (_eventPool != nullptr) {
        for (auto& event : _events) {
            zeEventDestroy(event);
        }

        zeEventPoolDestroy(_eventPool);
    }
}

result_t ExecutionContext::initialize(ze_graph_dditable_ext_t* ddiTable, ze_device_handle_t deviceHandle,
                                      ze_context_handle_t contextHandle) {
    if (ddiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for ddi table");
    }
    if (deviceHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for device handle");
    }

    _optimizedDynamicStridesSupported =
            (ddiTable->pfnCompilerIsOptionSupported != nullptr) &&
            (ddiTable->pfnCompilerIsOptionSupported(deviceHandle, ZE_NPU_DRIVER_OPTIONS, "OPTIMIZED_DYNAMIC_STRIDES",
                                                    nullptr) == ZE_RESULT_SUCCESS);

    return createEventPool(deviceHandle, contextHandle);
}

result_t ExecutionContext::createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context) {
    if (_numCmdLists > 1) {
        auto eventCount = (_numCmdLists - 1);
        ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                              ZE_EVENT_POOL_FLAG_HOST_VISIBLE, static_cast<uint32_t>(eventCount)};
        auto result = zeEventPoolCreate(context, &eventPoolDesc, /*numDevices*/ 1, &deviceHandle, &_eventPool);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create event pool for execution context");
        }

        _events.resize(eventCount);
        for (size_t i = 0; i < eventCount; i++) {
            ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, static_cast<uint32_t>(i), 0, 0};
            result = zeEventCreate(_eventPool, &eventDesc, &_events.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to create event");
            }
        }
    }

    _isInitialized = true;
    return ZE_RESULT_SUCCESS;
}

result_t createCmdList(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                       ze_command_list_handle_t& cmdListHandle) {
    // In a more complete implementation, this function would use the handle parameter to create a command list
    // appropriate for the current execution context. For this initial implementation, we will simply ignore the handle
    // and do nothing.
    NPU_VM_LOG_TRACE("create a command list");

    if (params == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for execution param handle");
    }

    cmdListHandle = executionContext.createCommandList(*params);

    if (cmdListHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "No more command list available for execution context");
    }

    return ZE_RESULT_SUCCESS;
}

result_t createKernel(npu_vm_runtime_execute_params_t* params, uint8_t* kernelBlob, size_t kernelBlobSize,
                      const std::string& kernelName, ze_graph_handle_t& graphHandle, KernelInfo& graphInfo) {
    NPU_VM_LOG_TRACE("Creating graph for kernel {} at address {} of size {}", kernelName, kernelBlob, kernelBlobSize);

    if (params == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for execution param handle");
    }

    const uint32_t flag = ZE_GRAPH_FLAG_OPTIMIZE_FOR_DYNAMIC_SHAPES;
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_2,
                              nullptr,
                              ZE_GRAPH_FORMAT_NATIVE,
                              static_cast<size_t>(kernelBlobSize),
                              kernelBlob,
                              nullptr /* build flag */,
                              flag};

    auto ddiTableHandle = params->graphDdiTableExt;
    auto deviceContext = params->ctx;
    auto device = params->device;

    auto pfnCreate = ddiTableHandle->pfnCreate2;
    auto result = pfnCreate(deviceContext, device, &desc, &graphHandle);
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

result_t setBindings(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                     ze_graph_handle_t graphHandle, const std::vector<BufferMapperItem>& inputs,
                     const std::vector<BufferMapperItem>& outputs, KernelInfo& graphInfo) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto ddiTableHandle = params->graphDdiTableExt;

    if (!executionContext.isInitialized()) {
        NPU_VM_LOG_TRACE("Creating event pool for execution context");

        auto deviceHandle = params->device;
        auto contextHandle = params->ctx;
        auto result = executionContext.initialize(ddiTableHandle, deviceHandle, contextHandle);
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
            auto result = setArguments(index, inputs.at(index), graphHandle, ddiTableHandle,
                                       graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set input argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        } else {
            auto result = setArguments(index, outputs.at(index - graphInfo.getNumInputArgs()), graphHandle,
                                       ddiTableHandle, graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set output argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        }
    }

    return success();
}

result_t executeGraph(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                      ze_command_list_handle_t commandListHandle, ze_graph_handle_t graphHandle, void* kernelName) {
    if (commandListHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle");
    }
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    if (params == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid execution parameters");
    }
    auto ddiTableHandle = params->graphDdiTableExt;
    auto cmdListIndex = executionContext.getCurrentCmdListIndex();

    if (commandListHandle != getCmdList(*params, cmdListIndex)) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
                       "Invalid commandListHandle for current execution context, got: {}, expected: {}",
                       commandListHandle, getCmdList(*params, cmdListIndex));
    }

    if (kernelName == nullptr) {
        NPU_VM_LOG_TRACE("Executing graph for unnamed kernel at handle {}", graphHandle);
    } else {
        NPU_VM_LOG_TRACE("Executing graph for kernel {} at handle {} in cmdList {} and execContext {}",
                         static_cast<const char*>(kernelName), graphHandle, commandListHandle, &executionContext);
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = ddiTableHandle->pfnAppendGraphExecute;
    ze_event_handle_t waitEvent = nullptr;
    uint32_t numWaitEvents = 0;
    auto& inferenceCmdIds = executionContext.getInferenceCmdIds();
    const auto inferenceCmdCount = inferenceCmdIds.size();
    if (static_cast<size_t>(cmdListIndex) < inferenceCmdCount) {
        auto id = inferenceCmdIds.at(cmdListIndex);
        if (id == DEFAULT_INFERENCE_ID) {
            waitEvent = executionContext.getWaitEvent();
            if (waitEvent != nullptr) {
                numWaitEvents = 1;
            }
        }
    } else {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index: {}, got: {}", cmdListIndex,
                       inferenceCmdCount);
    }

    ze_result_t result = ZE_RESULT_SUCCESS;
    if (numWaitEvents > 0 && waitEvent != nullptr) {
        NPU_VM_LOG_TRACE("Appending a barrier with a WAIT event: {}, numWaitEvents: {}", waitEvent, numWaitEvents);
        result = zeCommandListAppendBarrier(commandListHandle, nullptr, numWaitEvents, &waitEvent);
        if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
            result = zeCommandListReset(commandListHandle);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to reset a command list");
            }
            result = zeCommandListAppendBarrier(commandListHandle, nullptr, numWaitEvents, &waitEvent);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to append barrier before graph execute, kern: {}, numWaitEvents: {}",
                               graphHandle, numWaitEvents);
            }
            result = zeCommandListAppendEventReset(commandListHandle, waitEvent);
        } else {
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to append a barrier before graph execute, kern: {}, numWaitEvents: {}",
                               graphHandle, numWaitEvents);
            }
            result = zeCommandListAppendEventReset(commandListHandle, waitEvent);
        }

        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to append an event reset waitEvent: {}, numWaitEvents: {}", waitEvent,
                           numWaitEvents);
        }
    }

    result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);

    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        result = zeCommandListReset(commandListHandle);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to reset command list: {}, kern: {}", commandListHandle, graphHandle);
        }
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    }

    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}",
                       commandListHandle, graphHandle, numWaitEvents);
    }

    executionContext.increaseInferenceCmdId(cmdListIndex);

    NPU_VM_LOG_TRACE("Executed graph for kernel: {}", graphHandle);

    return success();
}

result_t submitCmdList(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                       ze_command_list_handle_t commandListHandle, bool needHostSync) {
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(params->commandQueue);
    ze_fence_handle_t fence = nullptr;
    ze_event_handle_t event = nullptr;
    if (needHostSync) {
        event = params->event;
        fence = params->inferenceFence;

        if (event != nullptr && fence != nullptr) {
            // Both event and fence are provided, which is not expected
            // as they serve similar synchronization purposes.
            return failure(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                           "Both event and fence are not nullptr(event: {}, fence: {})", event, fence);
        }
    }

    NPU_VM_LOG_TRACE("Submitting command list: {}, fence: {}, event: {}", commandListHandle, fence, event);

    auto commandQueue = params->commandQueue;

    auto result = ZE_RESULT_SUCCESS;
    if (commandListHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
                       "Invalid commandListHandle for submission, command queue: {}, fence: {}, event: {}",
                       commandQueue, fence, event);
    }

    // note command queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        // For the last command list, event or fence will be not nullptr when host synchronization is needed
        if (event != nullptr) {
            // host sync with an event given by npu-plugin
            result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to zeCommandListAppendBarrier");
            }

            result = zeCommandListAppendSignalEvent(commandListHandle, event);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to zeCommandListAppendSignalEvent");
            }
        } else {
            if (fence == nullptr) {
                // if both event and fence are nullptr,
                // add a barrier at the end of command list to ensure all commands are finished
                // before following command lists start inference execution
                auto signalEvent = executionContext.getSignalEvent();
                NPU_VM_LOG_TRACE("Append barrier with SIGNAL event: {} for command list: {}", signalEvent,
                                 commandListHandle);
                result = zeCommandListAppendBarrier(commandListHandle, signalEvent, 0, nullptr);
                if (result != ZE_RESULT_SUCCESS) {
                    return failure(result, "Failed to zeCommandListAppendBarrier");
                }
            }
        }
        result = zeCommandListClose(commandListHandle);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to zeCommandListClose");
        }

        result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fence);
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
