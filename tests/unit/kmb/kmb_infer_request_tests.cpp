//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#if 0
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vpux_infer_request.h>
#include <vpual_config.hpp>
#include <vpux.hpp>
#include <vpux_private_config.hpp>

#include "creators/creator_blob.h"
#include "creators/creator_blob_nv12.h"

namespace ie = InferenceEngine;

using namespace ::testing;
using namespace vpu::KmbPlugin;

class kmbInferRequestConstructionUnitTests : public ::testing::Test {
protected:
    ie::InputsDataMap setupInputsWithSingleElement() {
        std::string inputName = "input";
        ie::TensorDesc inputDescription = ie::TensorDesc(ie::Precision::U8, {1, 3, 224, 224}, ie::Layout::NHWC);
        ie::DataPtr inputData = std::make_shared<ie::Data>(inputName, inputDescription);
        ie::InputInfo::Ptr inputInfo = std::make_shared<ie::InputInfo>();
        inputInfo->setInputData(inputData);
        ie::InputsDataMap inputs = {{inputName, inputInfo}};

        return inputs;
    }

    ie::OutputsDataMap setupOutputsWithSingleElement() {
        std::string outputName = "output";
        ie::TensorDesc outputDescription = ie::TensorDesc(ie::Precision::U8, {1000}, ie::Layout::C);
        ie::DataPtr outputData = std::make_shared<ie::Data>(outputName, outputDescription);
        ie::OutputsDataMap outputs = {{outputName, outputData}};

        return outputs;
    }
};

constexpr int defaultDeviceId = 0;
class MockNetworkDescription : public vpux::INetworkDescription {
    const vpux::DataMap& getInputsInfo() const override { return inputs; }

    const vpux::DataMap& getOutputsInfo() const override { return outputs; }

    const vpux::DataMap& getDeviceInputsInfo() const override { return inputs; }

    const vpux::DataMap& getDeviceOutputsInfo() const override { return outputs; }

    const std::vector<char>& getCompiledNetwork() const override { return network; }

    const void* getNetworkModel() const override { return network.data(); }

    std::size_t getNetworkModelSize() const override { return network.size(); }

    const std::string& getName() const override { return name; }

private:
    std::string name;
    vpux::DataMap inputs;
    vpux::DataMap outputs;
    std::vector<char> network;
};

class MockExecutor : public vpux::Executor {
public:
    MOCK_METHOD2(push, void(const ie::BlobMap&, const vpux::PreprocMap&));
    MOCK_METHOD1(push, void(const ie::BlobMap&));
    MOCK_METHOD1(pull, void(ie::BlobMap&));

    void setup(const InferenceEngine::ParamMap&) {}

    bool isPreProcessingSupported(const vpux::PreprocMap&) const { return false; }
    std::map<std::string, ie::InferenceEngineProfileInfo> getLayerStatistics() {
        return std::map<std::string, ie::InferenceEngineProfileInfo>();
    }

    ie::Parameter getParameter(const std::string&) const { return ie::Parameter(); }
};

class MockAllocator : public vpux::Allocator {
public:
    void* lock(void* /*handle*/, InferenceEngine::LockOp) noexcept override {
        return reinterpret_cast<uint8_t*>(buffer);
    }

    void unlock(void* /*handle*/) noexcept override {}

    virtual void* alloc(size_t /*size*/) noexcept override { return reinterpret_cast<uint8_t*>(buffer); }

    virtual bool free(void* /*handle*/) noexcept override { return true; }

    unsigned long getPhysicalAddress(void* /*handle*/) noexcept override { return 0; }

    virtual bool isValidPtr(void* /*ptr*/) noexcept { return false; }

    void* wrapRemoteMemoryHandle(
        const int& /*remoteMemoryFd*/, const size_t /*size*/, void* /*memHandle*/) noexcept override {
        return nullptr;
    }
    void* wrapRemoteMemoryOffset(const int& /*remoteMemoryFd*/, const size_t /*size*/,
        const size_t& /*memOffset*/) noexcept override {
        return nullptr;
    }

private:
    uint8_t buffer[10];
};

TEST_F(kmbInferRequestConstructionUnitTests, cannotCreateInferRequestWithEmptyInputAndOutput) {
    vpux::VpualConfig config;

    auto executor = std::make_shared<MockExecutor>();
    vpux::InferRequest::Ptr inferRequest;

    auto allocator = std::make_shared<MockAllocator>();
    ASSERT_THROW(inferRequest = std::make_shared<vpux::InferRequest>(ie::InputsDataMap(), ie::OutputsDataMap(),
                     executor, config, "networkName", allocator),
        ie::Exception);
}

TEST_F(kmbInferRequestConstructionUnitTests, canCreateInferRequestWithValidParameters) {
    vpux::VpualConfig config;
    auto executor = std::make_shared<MockExecutor>();
    auto inputs = setupInputsWithSingleElement();
    auto outputs = setupOutputsWithSingleElement();

    auto allocator = std::make_shared<MockAllocator>();
    vpux::InferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(inferRequest = std::make_shared<vpux::InferRequest>(inputs, outputs,
                        executor, config, "networkName", allocator));
}
class TestableKmbInferRequest : public vpux::InferRequest {
public:
    TestableKmbInferRequest(const ie::InputsDataMap& networkInputs, const ie::OutputsDataMap& networkOutputs,
        const std::vector<vpu::StageMetaInfo>& /*blobMetaData*/, const vpux::VpualConfig& kmbConfig,
        const std::shared_ptr<vpux::Executor>& executor, const std::shared_ptr<ie::IAllocator>& allocator)
        // TODO blobMetaData removed - investigate usage [Track number: C#41602]
        : vpux::InferRequest(networkInputs, networkOutputs, executor, kmbConfig, "networkName", allocator){};

public:
    MOCK_METHOD7(execKmbDataPreprocessing, void(ie::BlobMap&, std::map<std::string, ie::PreProcessDataPtr>&,
                                               ie::InputsDataMap&, ie::ColorFormat, unsigned int, unsigned int, unsigned int));
    MOCK_METHOD2(execDataPreprocessing, void(ie::BlobMap&, bool));
};

// FIXME: cannot be run on x86 the tests below use vpusmm allocator and requires vpusmm driver instaled
// can be enabled with other allocator
// [Track number: S#28136]
// TODO Compile these tests, but do not run [Track number: S#37579]
#if defined(__arm__) || defined(__aarch64__)
class kmbInferRequestUseCasesUnitTests : public kmbInferRequestConstructionUnitTests {
protected:
    vpux::VpualConfig config;
    ie::InputsDataMap _inputs;
    ie::OutputsDataMap _outputs;
    std::shared_ptr<MockExecutor> _executor;
    std::shared_ptr<TestableKmbInferRequest> _inferRequest;

protected:
    void SetUp() override {
        _executor = std::make_shared<MockExecutor>();

        _inputs = setupInputsWithSingleElement();
        _outputs = setupOutputsWithSingleElement();

        _allocator = std::make_shared<MockAllocator>();
        _inferRequest = std::make_shared<TestableKmbInferRequest>(
            _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, _allocator);
    }

    ie::Blob::Ptr createVPUBlob(const ie::SizeVector dims, const ie::Layout layout = ie::Layout::NHWC) {
        if (dims.size() != 4) {
            IE_THROW() << "Dims size must be 4 for createVPUBlob method";
        }

        ie::TensorDesc desc = {ie::Precision::U8, dims, layout};

        auto blob = ie::make_shared_blob<uint8_t>(desc, _allocator);
        blob->allocate();

        return blob;
    }

    ie::NV12Blob::Ptr createNV12VPUBlob(const std::size_t width, const std::size_t height) {
        nv12Data = reinterpret_cast<uint8_t*>(_allocator->alloc(height * width * 3 / 2));
        return NV12Blob_Creator::createBlob(width, height, nv12Data);
    }

    void TearDown() override {
        // nv12Data can be allocated in two different ways in the tests below
        // that why we need to branches to handle removing of memory
        if (nv12Data != nullptr) {
            if (_allocator->isValidPtr(nv12Data)) {
                _allocator->free(nv12Data);
            }
        }
    }

private:
    uint8_t* nv12Data = nullptr;
    std::shared_ptr<MockAllocator> _allocator;
};

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesTheSameInputForInferenceAsGetBlobReturns) {
    auto inputName = _inputs.begin()->first.c_str();

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);
    ie::BlobMap inputs = {{inputName, input}};
    EXPECT_CALL(*_executor, push(inputs)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, requestUsesExternalShareableBlobForInference) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto vpuBlob = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    ie::BlobMap inputs = {{inputName, vpuBlob}};
    EXPECT_CALL(*_executor, push(inputs)).Times(1);

    _inferRequest->SetBlob(inputName, vpuBlob);

    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, DISABLED_requestUsesNonSIPPPPreprocIfResize) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto largeInput = Blob_Creator::createBlob({dims[0], dims[1], dims[2] * 2, dims[3] * 2});

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, largeInput, preProcInfo);

    // TODO: enable this check after execDataPreprocessing become virtual
    // EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(inferRequest.get()), execDataPreprocessing(_, _));
    EXPECT_CALL(*_executor, push(_)).Times(1);
    ASSERT_NO_THROW(_inferRequest->InferAsync());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetNV12Blob) {
    auto nv12Input = NV12Blob_Creator::createBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);
    ASSERT_EQ(nv12Input->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetVPUBlob) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto vpuInput = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->SetBlob(inputName, vpuInput);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);

    ASSERT_EQ(vpuInput->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, DISABLED_CanGetTheSameBlobAfterSetLargeVPUBlob) {
    auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    dims[2] *= 2;
    dims[3] *= 2;
    auto vpuInput = createVPUBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, vpuInput, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);

    ASSERT_EQ(vpuInput->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, CanGetTheSameBlobAfterSetOrdinaryBlobMatchedNetworkInput) {
    const auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    auto inputToSet = Blob_Creator::createBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    _inferRequest->SetBlob(inputName, inputToSet);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);

    ASSERT_EQ(inputToSet->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, DISABLED_CanGetTheSameBlobAfterSetOrdinaryBlobNotMatchedNetworkInput) {
    auto dims = _inputs.begin()->second->getTensorDesc().getDims();
    dims[2] *= 2;
    dims[3] *= 2;
    auto inputToSet = Blob_Creator::createBlob(dims);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    _inferRequest->SetBlob(inputName, inputToSet, preProcInfo);

    EXPECT_CALL(*_executor, push(_)).Times(1);

    ASSERT_NO_THROW(_inferRequest->InferAsync());

    ie::Blob::Ptr input = _inferRequest->GetBlob(inputName);

    ASSERT_EQ(inputToSet->buffer().as<void*>(), input->buffer().as<void*>());
}

TEST_F(kmbInferRequestUseCasesUnitTests, BGRIsDefaultColorFormatForSIPPPreproc) {
    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()),
        execKmbDataPreprocessing(_, _, _, ie::ColorFormat::BGR, _, _, _));

    _inferRequest->InferAsync();
}

class kmbInferRequestOutColorFormatSIPPUnitTests :
    public kmbInferRequestUseCasesUnitTests,
    public testing::WithParamInterface<const char*> {};

TEST_P(kmbInferRequestOutColorFormatSIPPUnitTests, preprocessingUseRGBIfConfigIsSet) {
    vpux::VpualConfig config;
    const auto configValue = GetParam();
    config.update({{VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT), configValue}});

    auto allocator = std::make_shared<MockAllocator>();
    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, allocator);

    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    auto expectedColorFmt = [](const std::string colorFmt) {
        if (colorFmt == "RGB") {
            return ie::ColorFormat::RGB;
        } else if (colorFmt == "BGR") {
            return ie::ColorFormat::BGR;
        }

        return ie::ColorFormat::RAW;
    };
    EXPECT_CALL(*dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()),
        execKmbDataPreprocessing(_, _, _, expectedColorFmt(configValue), _, _, _));

    _inferRequest->InferAsync();
}

INSTANTIATE_TEST_SUITE_P(
    SupportedColorFormats, kmbInferRequestOutColorFormatSIPPUnitTests, testing::Values("RGB", "BGR"));

class kmbInferRequestSIPPPreprocessing :
    public kmbInferRequestUseCasesUnitTests,
    public testing::WithParamInterface<std::string> {};

TEST_F(kmbInferRequestSIPPPreprocessing, DISABLED_canDisableSIPP) {
    vpux::VpualConfig config;
    config.update({{"VPUX_USE_SIPP", CONFIG_VALUE(NO)}});

    auto allocator = std::make_shared<MockAllocator>();
    _inferRequest = std::make_shared<TestableKmbInferRequest>(
        _inputs, _outputs, std::vector<vpu::StageMetaInfo>(), config, _executor, allocator);

    auto nv12Input = createNV12VPUBlob(1080, 1080);

    auto inputName = _inputs.begin()->first.c_str();
    auto preProcInfo = _inputs.begin()->second->getPreProcess();
    preProcInfo.setResizeAlgorithm(ie::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcInfo.setColorFormat(ie::ColorFormat::NV12);
    _inferRequest->SetBlob(inputName, nv12Input, preProcInfo);

    EXPECT_CALL(
        *dynamic_cast<TestableKmbInferRequest*>(_inferRequest.get()), execKmbDataPreprocessing(_, _, _, _, _, _, _))
        .Times(0);

    _inferRequest->InferAsync();
}

#endif  //  __arm__

#endif
