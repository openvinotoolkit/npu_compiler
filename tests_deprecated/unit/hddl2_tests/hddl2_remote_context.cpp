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

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <hddl2_helpers/helper_remote_memory.h>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_helpers/helper_workload_context.h"
#include "hddl2_params.hpp"
#include "hddl2_plugin.h"
#include "helper_remote_context.h"
#include "skip_conditions.h"
#include "hddl2_metrics.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_RemoteContext_UnitTests : public ::testing::Test {
public:
    void SetUp() override;

    InferenceEngine::ParamMap params;
    WorkloadContext_Helper::Ptr workloadContextHelperPtr;
    const vpu::HDDL2Config config = vpu::HDDL2Config();
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteContext_UnitTests::SetUp() {
    if (HDDL2Metrics::isServiceAvailable()) {
        workloadContextHelperPtr = std::make_shared<WorkloadContext_Helper>();
        WorkloadID id = workloadContextHelperPtr->getWorkloadId();
        params = RemoteContext_Helper::wrapWorkloadIdToMap(id);
    }
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - constructor
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromCorrectId_NoThrow) {
    SKIP_IF_NO_DEVICE();
    ASSERT_NO_THROW(HDDL2RemoteContext context(params, config));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_TwiceSameId_NoThrow) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr firstContext = nullptr;
    HDDL2RemoteContext::Ptr secondContext = nullptr;

    ASSERT_NO_THROW(firstContext = std::make_shared<HDDL2RemoteContext>(params, config));
    ASSERT_NO_THROW(secondContext = std::make_shared<HDDL2RemoteContext>(params, config));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_TwiceSameId_NoNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr firstContext = nullptr;
    HDDL2RemoteContext::Ptr secondContext = nullptr;

    firstContext = std::make_shared<HDDL2RemoteContext>(params, config);
    secondContext = std::make_shared<HDDL2RemoteContext>(params, config);

    ASSERT_NE(nullptr, firstContext);
    ASSERT_NE(nullptr, secondContext);
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromEmptyPararams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap emptyParams = {};

    ASSERT_ANY_THROW(HDDL2RemoteContext context(emptyParams, config));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromIncorrectPararams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap badParams = {{"Bad key", "Bad value"}};
    ASSERT_ANY_THROW(HDDL2RemoteContext context(badParams, config));
}

TEST_F(HDDL2_RemoteContext_UnitTests, constructor_fromNotExistsContext_ThrowException) {
    SKIP_IF_NO_DEVICE();
    const int badWorkloadId = UINT32_MAX;
    EXPECT_FALSE(workloadContextHelperPtr->isValidWorkloadContext(badWorkloadId));

    InferenceEngine::ParamMap notExistsParams = RemoteContext_Helper::wrapWorkloadIdToMap(badWorkloadId);

    ASSERT_ANY_THROW(HDDL2RemoteContext context(notExistsParams, config));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - destructor
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, destructor_workloadContextNotUnregistered) {
    SKIP_IF_NO_DEVICE();
    auto workloadId = workloadContextHelperPtr->getWorkloadId();
    auto params = RemoteContext_Helper::wrapWorkloadIdToMap(workloadId);

    ASSERT_NO_THROW({ HDDL2RemoteContext context(params, config); });
    ASSERT_TRUE(workloadContextHelperPtr->isValidWorkloadContext(workloadId));
}

//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, getDeviceName_ReturnCorrectPluginName) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    const auto devicesNames = DeviceName::getDevicesNamesWithPrefix();
    auto sameNameFound = devicesNames.find(context->getDeviceName());
    EXPECT_TRUE(sameNameFound != devicesNames.end());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteContext_UnitTests Initiations - getParams
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_UnitTests, getParams_containsWorkloadId) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    const std::string workloadContextKey = IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID);

    auto params = context->getParams();
    auto iter = params.find(workloadContextKey);

    ASSERT_TRUE(iter != params.end());
}

TEST_F(HDDL2_RemoteContext_UnitTests, getParams_containsSameWorkloadId) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);
    const std::string workloadContextKey = IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID);

    auto params = context->getParams();
    auto iter = params.find(workloadContextKey);
    const uint64_t workloadId = iter->second.as<uint64_t>();

    const uint64_t correctWorkloadId = workloadContextHelperPtr->getWorkloadId();
    ASSERT_EQ(correctWorkloadId, workloadId);
}

//------------------------------------------------------------------------------
class HDDL2_RemoteContext_CreateBlob_UnitTests : public HDDL2_RemoteContext_UnitTests {
public:
    void SetUp() override;
    void TearDown() override;

    InferenceEngine::ParamMap blobParams;
    InferenceEngine::TensorDesc tensorDesc;
    const size_t sizeToAllocate = 1024 * 1024 * 4;

protected:
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteMemory_Helper _remoteMemoryHelper;
};

void HDDL2_RemoteContext_CreateBlob_UnitTests::SetUp() {
    HDDL2_RemoteContext_UnitTests::SetUp();
    if (HDDL2Metrics::isServiceAvailable()) {
        tensorDesc = _tensorDescriptionHelper.tensorDesc;
        auto remoteMemory =
            _remoteMemoryHelper.allocateRemoteMemory(workloadContextHelperPtr->getWorkloadId(), sizeToAllocate);
        blobParams = RemoteBlob_Helper::wrapRemoteMemToMap(remoteMemory);
    }
}

void HDDL2_RemoteContext_CreateBlob_UnitTests::TearDown() { _remoteMemoryHelper.destroyRemoteMemory(); }

//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreateBlob_WithParams_ReturnNotNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    auto blob = context->CreateBlob(tensorDesc, blobParams);
    ASSERT_NE(nullptr, blob);
}

TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreateBlob_NoParams_ReturnNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);
    InferenceEngine::ParamMap emptyBlobParams = {};

    auto blob = context->CreateBlob(tensorDesc, emptyBlobParams);

    ASSERT_EQ(nullptr, blob);
}

TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreateBlob_InvalidParams_ReturnNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);
    InferenceEngine::ParamMap invalidBlobParams = {{"Invalid key", "Invalid value"}};

    auto blob = context->CreateBlob(tensorDesc, invalidBlobParams);

    ASSERT_EQ(nullptr, blob);
}

// TODO Provide more information to user that this way should not be used (How?)
TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreatBlob_NotFromPointer_ReturnNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext context(params, config);

    auto blob = context.CreateBlob(tensorDesc, blobParams);
    ASSERT_EQ(nullptr, blob);
}

TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreateBlob_Default_IsRemoteBlob) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    auto blob = context->CreateBlob(tensorDesc, blobParams);
    ASSERT_TRUE(blob->is<IE::RemoteBlob>());
}

TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, CreateBlob_Default_CanAllocate) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    auto blob = context->CreateBlob(tensorDesc, blobParams);

    ASSERT_NO_FATAL_FAILURE(blob->allocate());
}

TEST_F(HDDL2_RemoteContext_CreateBlob_UnitTests, DISABLED_CreateBlob_Default_LockedMemoryNotNull) {
    SKIP_IF_NO_DEVICE();
    HDDL2RemoteContext::Ptr context = std::make_shared<HDDL2RemoteContext>(params, config);

    auto blob = context->CreateBlob(tensorDesc, blobParams);

    blob->allocate();

    auto lockedMemory = blob->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    EXPECT_NE(nullptr, data);
}
