//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

// Tests verify ARM-specific Vpual component
#if defined(__arm__) || defined(__aarch64__)

#include <gtest/gtest.h>
#include <atomic>
#include <mutex>
#include <thread>
#include "vpux/utils/core/logger.hpp"
#include "vpual_core_nn_synchronizer.hpp"

using namespace vpux;

class StubImpl {
public:
    StubImpl(Logger _test_logger): test_logger(_test_logger){};

    int RequestInferenceFunction(NnExecWithProfilingMsg& request) {
        test_logger.info("Stub RequestInferenceFunction: requested # {0}", request.inferenceID);
        return X_LINK_SUCCESS;
    }

    int PollForResponseFunction(NnExecWithProfilingResponseMsg& response) {
        response.status = MVNCI_SUCCESS;
        response.inferenceID = ++inferID;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        test_logger.info("Stub PollForResponseFunction: responding # {0}", response.inferenceID);
        return X_LINK_SUCCESS;
    }

    int getInferNumber() const {
        return inferID;
    }

protected:
    Logger test_logger;
    std::atomic<int> inferID{0};
};

class VpualCoreNNSynchronizerTests : public ::testing::Test {
public:
    VpualCoreNNSynchronizerTests()
            : test_logger("vpual_sync_tests", LogLevel::Error), stub(test_logger), sync(stub, test_logger){};

protected:
    Logger test_logger;
    StubImpl stub;
    VpualCoreNNSynchronizer<StubImpl> sync;
};

TEST_F(VpualCoreNNSynchronizerTests, Pull1GetsResponse1Pull2HasNoRequest) {
    const int inferenceID_1 = 1;
    // pushing thread submits requests
    auto push = [&]() -> void {
        NnExecWithProfilingMsg request;
        int res = sync.RequestInference(request, inferenceID_1);
        ASSERT_EQ(X_LINK_SUCCESS, res);
    };
    auto pushThread = std::thread(push);

    // pulling threads are blocked waiting for the result
    auto pull = [&]() -> void {
        pushThread.join();
        test_logger.info("Pull: pulling infer #1");
        int res = sync.WaitForResponse(inferenceID_1);
        ASSERT_EQ(MVNCI_SUCCESS, res);
    };

    auto pullThread = std::thread(pull);
    pullThread.join();

    const int inferenceID_2 = 2;
    auto pull2 = [&]() -> void {
        test_logger.info("Pull: pulling infer #2");
        int res = sync.WaitForResponse(inferenceID_2);
        ASSERT_EQ(MVNCI_INTERNAL_ERROR, res);
    };
    auto pull2Thread = std::thread(pull2);

    pull2Thread.join();
}

TEST_F(VpualCoreNNSynchronizerTests, PushReq1NoPullReq2) {
    std::mutex pullMtx;
    std::condition_variable pullCV;
    // pushing thread submits requests #1
    auto push = [&]() -> void {
        test_logger.info("Push: requesting infer #1");
        NnExecWithProfilingMsg request;
        int res = sync.RequestInference(request, 1);
        ASSERT_EQ(X_LINK_SUCCESS, res);
    };

    // pulling thread is blocked waiting for the result #1
    auto pull2 = [&]() -> void {
        test_logger.info("Pull: pulling infer #2");
        int res = sync.WaitForResponse(2);
        ASSERT_NE(MVNCI_SUCCESS, res);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        pullCV.notify_one();
    };

    auto pushThread = std::thread(push);
    pushThread.join();
    auto pullThread = std::thread(pull2);

    // pull will not happen ever since there is no matching response
    std::unique_lock<std::mutex> pullLk(pullMtx);
    std::cv_status res = pullCV.wait_for(pullLk, std::chrono::milliseconds(1000));
    ASSERT_NE(std::cv_status::timeout, res);

    test_logger.info("No match for infer #2");

    // NOTE: to unblock pull thread which has matching entry, but does not receive matching response, there is no
    // way...
    pullThread.join();
}

TEST_F(VpualCoreNNSynchronizerTests, Pull1TakesLongerThanPull2) {
    // pushing thread submits requests #1, #2
    auto push = [&]() -> void {
        NnExecWithProfilingMsg request;

        test_logger.info("Push: requesting infer #1");
        int res = sync.RequestInference(request, 1);
        ASSERT_EQ(X_LINK_SUCCESS, res);

        test_logger.info("Push: requesting infer #2");
        res = sync.RequestInference(request, 2);
        ASSERT_EQ(X_LINK_SUCCESS, res);
    };

    // pulling thread is blocked waiting for the result
    auto pull = [&](int inferenceID) -> void {
        int sleepFor = 100 / inferenceID;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepFor));
        test_logger.info("Pull: pulling infer # {0} after {1}", inferenceID, sleepFor);
        int res = sync.WaitForResponse(inferenceID);
        ASSERT_EQ(MVNCI_SUCCESS, res);
    };

    auto pushThread = std::thread(push);
    pushThread.join();

    auto pull1Thread = std::thread(pull, 1);
    auto pull2Thread = std::thread(pull, 2);

    pull1Thread.join();
    pull2Thread.join();
}

TEST_F(VpualCoreNNSynchronizerTests, HundredPushPulls) {
    const int reqNumber = 100;
    std::vector<std::shared_ptr<std::thread>> pullThreads;
    for (int i = 1; i <= reqNumber; ++i) {
        // pushing thread submits requests #1, #2
        auto push = [&, i]() -> void {
            NnExecWithProfilingMsg request;

            test_logger.info("Push: requesting infer # {0}", i);
            int res = sync.RequestInference(request, i);
            ASSERT_EQ(X_LINK_SUCCESS, res);
        };
        std::shared_ptr<std::thread> pushThread = std::make_shared<std::thread>(push);

        auto pull = [&, i, pushThread]() -> void {
            pushThread->join();

            int inferenceID = i;
            test_logger.info("Pull: pulling infer # {0}", inferenceID);
            // NOTE: if pulling thread misses the notification, it will indefinitely wait for the next one
            int res = sync.WaitForResponse(inferenceID);
            ASSERT_EQ(MVNCI_SUCCESS, res);
        };
        pullThreads.emplace_back(std::make_shared<std::thread>(pull));
    }

    test_logger.info("Finishing all pullings...");
    for (int i = 1; i <= reqNumber; ++i) {
        pullThreads[i - 1]->join();
    }

    ASSERT_EQ(stub.getInferNumber(), reqNumber);
}
#endif  // #if defined(__arm__) || defined(__aarch64__)
