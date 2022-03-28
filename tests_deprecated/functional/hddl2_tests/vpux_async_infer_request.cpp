//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <creators/creator_blob_nv12.h>
#include <ie_compound_blob.h>

#include <blob_factory.hpp>
#include <mutex>
#include <random>
#include <thread>

#include "cases/core_api.h"
#include "comparators.h"
#include "file_reader.h"
#include <hddl2_helpers/helper_calc_cpu_ref.h>
#include <tests_common.hpp>
#include "executable_network_factory.h"
#include "models/models_constant.h"

#include "vpux/utils/IE/blob.hpp"

namespace IE = InferenceEngine;

class AsyncInferRequest_Tests : public CoreAPI_Tests {
public:
    const int REQUEST_LIMIT = 10;
    const int MAX_WAIT = 60000;
    const size_t numberOfTopClassesToCompare = 3;
    const Models::ModelDesc modelToUse = Models::googlenet_v1;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

protected:
    void SetUp() override;

    std::vector<IE::InferRequest> createRequests(const int& numberOfRequests);

    static void loadCatImageToBlobForRequests(
        const std::string& blobName, std::vector<IE::InferRequest>& requests);
};

void AsyncInferRequest_Tests::SetUp() {

    executableNetworkPtr = std::make_shared<IE::ExecutableNetwork>(ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel));
}

std::vector<IE::InferRequest> AsyncInferRequest_Tests::createRequests(const int& numberOfRequests) {
    std::vector<IE::InferRequest> requests;
    for (int requestCount = 0; requestCount < numberOfRequests; requestCount++) {
        IE::InferRequest inferRequest;
        inferRequest = executableNetworkPtr->CreateInferRequest();
        requests.push_back(inferRequest);
    }
    return requests;
}

void AsyncInferRequest_Tests::loadCatImageToBlobForRequests(
    const std::string& blobName, std::vector<IE::InferRequest>& requests) {
    for (auto currentRequest : requests) {
        IE::Blob::Ptr blobPtr;
        auto inputBlob = loadCatImage();
        currentRequest.SetBlob(blobName, inputBlob);
    }
}

//------------------------------------------------------------------------------
// TODO Refactor create infer request for async inference correctly - use benchmark app approach
TEST_F(AsyncInferRequest_Tests, precommit_asyncIsFasterThenSync) {
    using Time = std::chrono::high_resolution_clock::time_point;
    Time (&Now)() = std::chrono::high_resolution_clock::now;

    Time start_sync;
    Time end_sync;
    Time start_async;
    Time end_async;
    {
        // --- Get input
        auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;

        // --- Create requests
        std::vector<IE::InferRequest> syncRequests = createRequests(REQUEST_LIMIT);
        loadCatImageToBlobForRequests(inputBlobName, syncRequests);
        // --- Sync execution
        start_sync = Now();
        for (IE::InferRequest& currentRequest : syncRequests) {
            ASSERT_NO_THROW(currentRequest.Infer());
        }
        end_sync = Now();


        // --- Create requests
        std::vector<IE::InferRequest> asyncRequests = createRequests(REQUEST_LIMIT);
        loadCatImageToBlobForRequests(inputBlobName, asyncRequests);

        // --- Specify callback
        std::mutex requestCounterGuard;
        volatile int completedRequests = 0;
        auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
            std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
            completedRequests++;
        };

        start_async = Now();
        // --- Asynchronous execution
        for (IE::InferRequest& currentRequest : asyncRequests) {
            currentRequest.SetCompletionCallback(onComplete);
            currentRequest.StartAsync();
        }

        auto waitRoutine = [&completedRequests, this]() -> void {
            std::chrono::system_clock::time_point endTime =
                std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
            while (completedRequests < REQUEST_LIMIT) {
                ASSERT_LE(std::chrono::system_clock::now(), endTime);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };

        std::thread waitThread(waitRoutine);
        waitThread.join();
        end_async = Now();
    }

    auto elapsed_sync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync);
    std::cout << "Sync inference (ms)" << elapsed_sync.count() << std::endl;

    auto elapsed_async = std::chrono::duration_cast<std::chrono::milliseconds>(end_async - start_async);
    std::cout << "Async inference (ms)" << elapsed_async.count() << std::endl;

    ASSERT_LT(elapsed_async.count(), elapsed_sync.count());
}

TEST_F(AsyncInferRequest_Tests, precommit_correctResultSameInput) {
    // --- Create requests
    std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    loadCatImageToBlobForRequests(inputBlobName, requests);

    // --- Specify callback
    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // --- Asynchronous execution
    for (IE::InferRequest& currentRequest : requests) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    auto waitRoutine = [&completedRequests, this]() -> void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    // --- Reference Blob
    IE::Blob::Ptr inputBlob = requests.at(0).GetBlob(inputBlobName);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);

    // --- Compare output with reference
    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    for (auto currentRequest : requests) {
        IE::Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = currentRequest.GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                            vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                            vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                            numberOfTopClassesToCompare));
    }
}

//------------------------------------------------------------------------------
class AsyncInferRequest_DifferentInput : public AsyncInferRequest_Tests {
public:
    struct Reference {
        explicit Reference(const bool _isNV12 = false)
            : isNV12(_isNV12) {}
        bool isNV12;
    };

    std::vector<Reference> references;
    std::string inputNV12Path;

protected:
    void SetUp() override;
};

void AsyncInferRequest_DifferentInput::SetUp() {
    AsyncInferRequest_Tests::SetUp();
    inputNV12Path = TestDataHelpers::get_data_path() + "/" + std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "/cat3.yuv";
    std::vector<Reference> availableReferences;

    availableReferences.emplace_back(Reference(false));
    availableReferences.emplace_back(Reference(true));

    const uint32_t seed = 666;
    static auto randEngine = std::default_random_engine(seed);
    const int referencesToUse = availableReferences.size();
    auto randGen = std::bind(std::uniform_int_distribution<>(0, referencesToUse - 1), randEngine);

    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        references.push_back(availableReferences.at(randGen()));
    }
}

//------------------------------------------------------------------------------
TEST_F(AsyncInferRequest_DifferentInput, precommit_correctResultShuffled_NoPreprocAndPreproc) {
    // --- Create requests
    std::vector<IE::InferRequest> requests = createRequests(REQUEST_LIMIT);
    auto inputBlobName = executableNetworkPtr->GetInputsInfo().begin()->first;
    IE::Blob::Ptr refRgbBlob = nullptr;
    IE::Blob::Ptr refNV12Blob = nullptr;

    // --- Load random reference images
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        if (references.at(i).isNV12) {
            // ----- Load NV12 input
            std::vector<uint8_t> nv12InputBlobMemory;
            nv12InputBlobMemory.resize(inputWidth * inputHeight * 3 / 2);
            IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

            // Preprocessing
            IE::PreProcessInfo preprocInfo = requests.at(i).GetPreProcess(inputBlobName);
            preprocInfo.setColorFormat(IE::ColorFormat::NV12);

            // ---- Set NV12 blob with preprocessing information
            requests.at(i).SetBlob(inputBlobName, nv12InputBlob, preprocInfo);

            if (refNV12Blob == nullptr) {
                refNV12Blob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, true, &preprocInfo);
            }
        } else {
            auto inputBlob = loadCatImage();
            ASSERT_NO_THROW(requests.at(i).SetBlob(inputBlobName, inputBlob));

            if (refRgbBlob == nullptr) {
                refRgbBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);
            }
        }
    }

    // --- Specify callback
    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // --- Asynchronous execution
    for (IE::InferRequest& currentRequest : requests) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    auto waitRoutine = [&completedRequests, this]() -> void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);
        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    // --- Compare output with reference
    auto outputBlobName = executableNetworkPtr->GetOutputsInfo().begin()->first;
    IE::Blob::Ptr refBlob;
    IE::Blob::Ptr outputBlob;
    for (int i = 0; i < REQUEST_LIMIT; ++i) {
        refBlob = references.at(i).isNV12 ? refNV12Blob : refRgbBlob;
        ASSERT_NO_THROW(outputBlob = requests.at(i).GetBlob(outputBlobName));
        ASSERT_NO_THROW(Comparators::compareTopClassesUnordered(
                            vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                            vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                            numberOfTopClassesToCompare));
    }
}
