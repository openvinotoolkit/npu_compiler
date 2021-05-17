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
// IE
#include <ie_remote_context.hpp>
// Plugin
#include <vpux_config.hpp>
// Low-level
#include <RemoteMemory.h>
#include <WorkloadContext.h>

namespace vpux {
namespace hddl2 {

HddlUnite::RemoteMemory* getRemoteMemoryFromParams(const InferenceEngine::ParamMap& params);
int32_t getRemoteMemoryFDFromParams(const InferenceEngine::ParamMap& params);
std::shared_ptr<InferenceEngine::TensorDesc> getOriginalTensorDescFromParams(const InferenceEngine::ParamMap& params);
WorkloadID getWorkloadIDFromParams(const InferenceEngine::ParamMap& params);
void setUniteLogLevel(const vpu::LogLevel logLevel, const vpu::Logger::Ptr logger = nullptr);
std::map<uint32_t, std::string> getSwDeviceIdNameMap();
std::string getSwDeviceIdFromName(const std::string& devName);

}  // namespace hddl2
}  // namespace vpux
