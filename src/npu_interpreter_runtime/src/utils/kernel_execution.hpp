//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "npu_interpreter_runtime/npu_vm_runtime.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace intel_npu::vm {

// Opaque handle representing the execution context for a function call, can be extended in the future to hold more
// information if needed
using result_t = int32_t;
using BufferMapperItem = std::pair<Buffer*, BufferMetadata*>;

#if defined(WIN32)
constexpr uint32_t DEFAULT_INFERENCE_ID = 1;
#else
constexpr uint32_t DEFAULT_INFERENCE_ID = 0;
#endif

// Structure to hold graph information
class KernelInfo {
public:
    // Graph handle created by L0 UMD, used for setting arguments and executing the graph
    ze_graph_handle_t _graphHandle;

    // Total number of arguments (inputs + outputs) for the graph, used for validating the number of arguments
    uint32_t _numArgs;

    // Number of input arguments for the graph, used for validating the number of input arguments and output arguments
    uint32_t _numInputArgs;

    // Kernel symbol name
    std::string _kernelName;

    KernelInfo(): _graphHandle(nullptr), _numArgs(0), _numInputArgs(0) {
    }

    KernelInfo(ze_graph_handle_t handle, uint32_t numArgs, uint32_t numInputArgs, const std::string& kernelName)
            : _graphHandle(handle), _numArgs(numArgs), _numInputArgs(numInputArgs), _kernelName(kernelName) {
    }

    uint32_t getNumArgs() const {
        return _numArgs;
    }

    uint32_t getNumInputArgs() const {
        return _numInputArgs;
    }

    ze_graph_handle_t getGraphHandle() const {
        return _graphHandle;
    }

    std::string& getKernelName() {
        return _kernelName;
    }

    const std::string& getKernelName() const {
        return _kernelName;
    }
};

class ExecutionContext {
    // Stores index to get the next command list from an array of command lists
    uint64_t _nextCmdListIndex = 0;

    // Stores the total number of command lists
    size_t _numCmdLists = 0;

    // Stores events for synchronization. The number of events is determined by the number of command lists
    std::vector<ze_event_handle_t> _events;

    ze_event_pool_handle_t _eventPool;

    // Stores the current index of the event to be used for synchronization.
    size_t _curEventIndex;

    // Stores the count of signal events that have been used, used for determining
    // whether to return a signal event or a wait event for synchronization.
    size_t _signalEventCount;

    // Stores whether the execution context has been initialized,
    // used for determining whether to initialize the event pool and events
    bool _isInitialized;

    // Stores the current cmd ids
    std::vector<uint64_t> _inferenceCmdIds;

    // Stores whether optimized dynamic strides is supported, which will be queried from the driver
    // and used for determining whether to use optimized dynamic strides when setting graph arguments
    bool _optimizedDynamicStridesSupported = false;

public:
    // note: numNetworkArgs will be used in support for mutable command list in the future
    // but currently it is not used.
    ExecutionContext(size_t numCmdLists, size_t /*numNetworkArgs*/);
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) noexcept;
    ExecutionContext& operator=(ExecutionContext&&) noexcept;
    ~ExecutionContext();

    // Returns whether the execution context has been initialized,
    // used for determining whether to initialize the event pool and events
    bool isInitialized() const {
        return _isInitialized;
    }

    // Resets the execution context to its initial state, including resetting the command list index and events.
    void reset() {
        _nextCmdListIndex = 0;
        resetEvents();
    }

    // Returns the next command list handle from the array of command lists in the execution parameters,
    // and increments the command list index.
    ze_command_list_handle_t createCommandList(const npu_vm_runtime_execute_params_t& params);

    // Resets the event index and signal event count to their initial state,
    // used for resetting the synchronization state of the execution context.
    void resetEvents() {
        _curEventIndex = 0;
        _signalEventCount = 0;
    }

    result_t initialize(ze_graph_dditable_ext_t* ddiTable, ze_device_handle_t deviceHandle,
                        ze_context_handle_t context);

    // Returns the next event handle for synchronization.
    ze_event_handle_t getSignalEvent();

    // Returns the next event handle for synchronization, and increments the event index if a signal event has been
    // used.
    ze_event_handle_t getWaitEvent();

    // Returns the current command list index,
    // which is the index of the command list that was most recently returned by createCommandList.
    uint64_t getCurrentCmdListIndex() const {
        return _nextCmdListIndex > 0 ? (_nextCmdListIndex - 1) : 0;
    }

    // Returns the inference command ids, which is used for tracking the command id for each command list in the
    // execution context.
    std::vector<uint64_t>& getInferenceCmdIds() {
        return _inferenceCmdIds;
    }

    // Increases the inference command id for the given command list index,
    // which is used for tracking the command id for each command list in the execution context.
    void increaseInferenceCmdId(uint64_t cmdListIndex) {
        if (cmdListIndex < _inferenceCmdIds.size()) {
            _inferenceCmdIds[cmdListIndex]++;
        }
    }

    // Returns whether optimized dynamic strides is supported,
    // which is used for determining whether to use optimized dynamic strides when setting graph arguments.
    bool IsOptimizedDynamicStridesSupported() const {
        return _optimizedDynamicStridesSupported;
    }

private:
    result_t createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context);
};

// Function declarations for creating command list, creating kernel, executing graph and submitting command list.

// Returns a command list handle created based on the execution parameters, which will be used for recording graph
// execution commands.
result_t createCmdList(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                       ze_command_list_handle_t& cmdListHandle);

// Closes a command list after recording graph execution commands, which will be used for submitting the command list
// for execution.
// @note Currently, close will be done in submitCmdList, so this function is a NO OP,
//       but it is defined here for completeness and future extension if needed.
inline result_t closeCmdList(npu_vm_runtime_execute_params_t*, ze_command_list_handle_t) {
    // NO OP as close will be done in command list submission
    return ZE_RESULT_SUCCESS;
}

result_t createKernel(npu_vm_runtime_execute_params_t* params, void* blob, size_t blobSize,
                      const std::string& kernelName, ze_graph_handle_t& graphHandle, KernelInfo& graphInfo);

// Sets the graph arguments for the given kernel handle based on the input and output buffer descriptions,
// which will be used for executing the graph.
result_t setBindings(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                     ze_graph_handle_t kernelHandle, const std::vector<BufferMapperItem>& inputs,
                     const std::vector<BufferMapperItem>& outputs, KernelInfo& graphInfo);

// Executes the graph for the given kernel handle and command list handle
result_t executeGraph(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                      ze_command_list_handle_t cmdListHandle, ze_graph_handle_t kernelHandle, void* kernelName);

// Submit a command list
result_t submitCmdList(ExecutionContext& executionContext, npu_vm_runtime_execute_params_t* params,
                       ze_command_list_handle_t cmdListHandle, bool needHostSync);
}  // namespace intel_npu::vm
