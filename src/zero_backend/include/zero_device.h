//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//
#pragma once

#include <ie_allocator.hpp>
#include <memory>
#include <vpux.hpp>
#include <vpux_compiler.hpp>

#include "ze_api.h"
#include "ze_fence_ext.h"
#include "ze_graph_ext.h"

namespace vpux {
class ZeroDevice : public IDevice {
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_fence_dditable_ext_t* _fence_ddi_table_ext = nullptr;

public:
    ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
               ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext)
            : _driver_handle(driver),
              _device_handle(device),
              _context(context),
              _graph_ddi_table_ext(graph_ddi_table_ext),
              _fence_ddi_table_ext(fence_ddi_table_ext) { }

    std::shared_ptr<Allocator> getAllocator() const override;

    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    std::string getName() const override;

    ZeroDevice& operator=(const ZeroDevice&) = default;
    ZeroDevice(const ZeroDevice&) = default;
    ~ZeroDevice() = default;
};
}  // namespace vpux
