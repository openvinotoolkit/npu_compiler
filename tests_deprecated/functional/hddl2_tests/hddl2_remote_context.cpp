//
// Copyright 2019 Intel Corporation.
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

#include "core_api.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_workload_context.h"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_remote_context.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_Remote_Context_Tests : public CoreAPI_Tests {};

TEST_F(HDDL2_Remote_Context_Tests, CanCreateContextFromParams) {
    WorkloadID workloadId = WorkloadContext_Helper::createAndRegisterWorkloadContext();

    // Store id param_map
    IE::ParamMap paramMap = {{IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};

    IE::RemoteContext::Ptr remoteContextPtr;
    // Create context from ParamMap
    EXPECT_NO_THROW(remoteContextPtr = ie.CreateContext(pluginName, paramMap));
    EXPECT_NE(remoteContextPtr.get(), nullptr);

    // Destroy after finishing working with context
    WorkloadContext_Helper::destroyHddlUniteContext(workloadId);
}

//------------------------------------------------------------------------------
class HDDL2_Remote_Context_Manipulation_Tests : public HDDL2_Remote_Context_Tests {
public:
    void SetUp() override;
    IE::ParamMap params;

private:
    WorkloadContext_Helper workloadContextHelper;
};

void HDDL2_Remote_Context_Manipulation_Tests::SetUp() {
    WorkloadID workloadId = workloadContextHelper.getWorkloadId();
    params = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
}

TEST_F(HDDL2_Remote_Context_Manipulation_Tests, CanGetDeviceName) {
    IE::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    std::string deviceName;
    ASSERT_NO_THROW(deviceName = remoteContextPtr->getDeviceName());
    ASSERT_GT(deviceName.size(), 0);
}

TEST_F(HDDL2_Remote_Context_Manipulation_Tests, GetDeviceNameReturnCorrectFormat) {
    IE::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    const std::string deviceName = remoteContextPtr->getDeviceName();

    ASSERT_NE(deviceName.find(pluginName), std::string::npos);
}

TEST_F(HDDL2_Remote_Context_Manipulation_Tests, CanGetParams) {
    IE::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    IE::ParamMap paramMap;
    ASSERT_NO_THROW(paramMap = remoteContextPtr->getParams());
    ASSERT_FALSE(paramMap.empty());
}
