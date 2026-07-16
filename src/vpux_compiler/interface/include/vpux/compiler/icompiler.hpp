//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/config/config.hpp"
#include "npu_driver_compiler.h"

#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/profiling_info.hpp"

#include "vpux/compiler/network_metadata.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpux {

constexpr uint32_t SUPPORTED_OPSET = 11;

// Same as defined in ze_graph_profiling_ext.h
namespace ze {

#define ZE_MAX_GRAPH_PROFILING_LAYER_NAME 256
#define ZE_MAX_GRAPH_PROFILING_LAYER_TYPE 50

typedef enum _ze_layer_status_t {
    ZE_LAYER_STATUS_NOT_RUN = 1,
    ZE_LAYER_STATUS_OPTIMIZED_OUT,
    ZE_LAYER_STATUS_EXECUTED

} ze_layer_status_t;

typedef struct _ze_profiling_layer_info {
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_layer_status_t status;
    uint64_t start_time_ns;   ///< Absolute start time
    uint64_t duration_ns;     ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id;        ///< Not used
    uint64_t fused_layer_id;  ///< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;

} ze_profiling_layer_info;

typedef enum _ze_task_execute_type_t {
    ZE_TASK_EXECUTE_NONE = 0,
    ZE_TASK_EXECUTE_DPU,
    ZE_TASK_EXECUTE_SW,
    ZE_TASK_EXECUTE_DMA

} ze_task_execute_type_t;

typedef struct _ze_profiling_task_info {
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_task_execute_type_t exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles;
    uint32_t stall_cycles;
    uint32_t task_id;
    uint32_t parent_layer_id;  ///< Not used

} ze_profiling_task_info;

static_assert(sizeof(ze_profiling_task_info) == 344);
static_assert(sizeof(ze_profiling_layer_info) == 368);

}  // namespace ze

/**
 * @brief Input and output precisions from user
 */
struct Precisions {
    using PrecisionMap = std::unordered_map<std::string, ov::element::Type_t>;

    PrecisionMap inputPrecisions;
    PrecisionMap outputPrecisions;

    bool isValid() const {
        return !inputPrecisions.empty() && !outputPrecisions.empty();
    }
};

/**
 * @brief Input and output layouts from user
 */
struct Layouts {
    using LayoutMap = std::unordered_map<std::string, std::string>;

    LayoutMap inputLayouts;
    LayoutMap outputLayouts;

    bool isValid() const {
        return !inputLayouts.empty() && !outputLayouts.empty();
    }
};

/**
 * @brief Aggregates a model and its I/O precision/layout configuration.
 */
struct ModelData {
    std::shared_ptr<ov::Model> model;

    Precisions precisions;
    Layouts layouts;
};

class InvalidIrError : public std::runtime_error {
public:
    explicit InvalidIrError(const std::string& message): runtime_error(message) {
    }
};

class BlobAllocator {
public:
    virtual ~BlobAllocator() = default;
    virtual uint8_t* allocate(vpux::Byte) = 0;
    virtual void deallocate(uint8_t*) = 0;
};

// Non-owning view into a memory occupied by a blob. Used by AllocatedCompiledNetwork
// to store compiled model allocated via BlobAllocator implementation.
struct BlobView final {
    // E#-140887: ptr left mutable to be compatible with initial version of VCL
    // interface; make BlobView immutable and reuse it in CompiledNetwork
    uint8_t* ptr = nullptr;
    uint64_t size = 0;

    BlobView(uint8_t*, uint64_t);
    // E#-140887: enable implicit conversion from std::vector<uint8_t>
    // currently it'll fail due to blob.data() being const uint8_t* that
    // can't be converted to uint8_t*
    // /* implicit */ BlobView(const std::vector<uint8_t>& blob);
};

// The object returned by the compiler to provide such information about a network
// as description of inputs and outputs, name and compiled network in a format
// executable by device
// The difference between NetworkDescriptionView and NetworkDescription is
// compiled network is represented via BlobView. Blob in this case is allocated by
// compiler via provided BlobAllocator implementation.
struct NetworkDescriptionView {
    NetworkDescriptionView(BlobView blob, NetworkMetadata&&);
    NetworkDescriptionView(BlobView blob, BlobView compatibilityString, NetworkMetadata&&);

    NetworkDescriptionView(const NetworkDescriptionView&) = delete;
    NetworkDescriptionView& operator=(const NetworkDescriptionView&) = delete;

    NetworkDescriptionView(NetworkDescriptionView&&) = default;
    NetworkDescriptionView& operator=(NetworkDescriptionView&&) = default;

    ~NetworkDescriptionView() = default;

    BlobView compiledNetwork;
    BlobView compatibilityString;  // To be removed E#219950
    NetworkMetadata metadata;
};

/**
 * @interface ICompiler
 * @brief An interface to be implemented by a concrete compiler to provide
 * methods for preparing a network for execution on a NPU device
 */
class ICompiler : public std::enable_shared_from_this<ICompiler> {
protected:
    ICompiler() = default;

public:
    virtual ~ICompiler() = default;

    ICompiler(const ICompiler&) = delete;
    ICompiler(ICompiler&&) = delete;
    ICompiler& operator=(const ICompiler&) = delete;
    ICompiler& operator=(ICompiler&&) = delete;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return Compiled network description
     */
    virtual NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                                       const intel_npu::Config& config) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return A "NetworkDescription" object for each init schedule, followed by another one corresponding to the main
     * part.
     */
    virtual std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(
            const std::shared_ptr<ov::Model>& /*model*/, const intel_npu::Config& /*config*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

    /**
     * @brief Sequential compilation of Init(s) and Main
     *
     * "Stateless compiler" approach
     * We want to get multiple Inits in the case of a large number of weights.
     * This allows us to build pipeline:
     * Allocate W1 -> Init1
     *             Allocate W2 -> Init2
     *                          Allocate W3 -> Init2
     *
     * This is why there is an additional parameter callNumber:
     * Compiler should somehow understand which Init(or Main) to return
     * Plugin does not know total numbers of Init schedules
     */
    virtual NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& /*model*/,
                                                  const intel_npu::Config& /*config*/, size_t /*callNumber*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                                      const intel_npu::Config& config) const = 0;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to NPUConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @return Network metadata extracted from the compiled network
     */
    virtual NetworkMetadata parse(const std::vector<uint8_t>& network, const intel_npu::Config& config) const = 0;

    virtual std::vector<ov::ProfilingInfo> processProfilingOutput(const std::vector<uint8_t>& profData,
                                                                  const std::vector<uint8_t>& network,
                                                                  const intel_npu::Config& config) const = 0;

    virtual std::vector<ze::ze_profiling_layer_info> getLayerInfo(const uint8_t* blobData, uint64_t blobSize,
                                                                  const uint8_t* profData, uint64_t profSize) const = 0;

    virtual std::vector<ze::ze_profiling_task_info> getTaskInfo(const uint8_t* blobData, uint64_t blobSize,
                                                                const uint8_t* profData, uint64_t profSize) const = 0;

    // CiD-specific methods

    virtual NetworkDescriptionView compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                           BlobAllocator& allocator,
                                           bool allocateCompatibilityString = false) const = 0;

    virtual NetworkDescriptionView compile(const std::shared_ptr<const ov::Model>& model,
                                           const intel_npu::Config& config, BlobAllocator& allocator,
                                           bool allocateCompatibilityString = false) const = 0;

    // VCL specific methods

    virtual ov::SupportedOpsMap queryFromDesc(const vcl_query_desc_t& desc, vcl_compiler_desc_t& compilerDesc,
                                              vcl_compiler_properties_t& compilerProp, vcl_device_desc_t& deviceDesc,
                                              intel_npu::Config& config, bool isDeviceDescEmpty) const = 0;

    virtual NetworkDescription compileFromDesc(const vcl_executable_desc_t& desc,
                                               const vcl_compiler_properties_t& compilerProp,
                                               vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                               intel_npu::Config& config, bool isDeviceDescEmpty) const = 0;

    virtual NetworkDescriptionView compileFromDesc(const vcl_executable_desc_t& desc,
                                                   const vcl_compiler_properties_t& compilerProp,
                                                   vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                                   intel_npu::Config& config, bool isDeviceDescEmpty,
                                                   BlobAllocator& allocator,
                                                   bool generateCompatibilityString = false) const = 0;

    virtual std::vector<std::shared_ptr<NetworkDescriptionView>> compileFromDescWsOneShot(
            const vcl_executable_desc_t& desc, const vcl_compiler_properties_t& compilerProp,
            vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
            bool isDeviceDescEmpty, BlobAllocator& allocator) const = 0;
};

}  // namespace vpux
