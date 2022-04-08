//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <helper_ie_core.h>

#include <blob_factory.hpp>
#include <vpux_private_config.hpp>
#include "executable_network_factory.h"

#include <hddl2_helpers/helper_calc_cpu_ref.h>
#include "comparators.h"
#include "creators/creator_blob_nv12.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "models/models_constant.h"
#include "tests_common.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "vpux/utils/IE/blob.hpp"

namespace IE = InferenceEngine;

class ImageWorkload_Tests : public ::testing::Test {
public:
    Models::ModelDesc modelForNoPreprocess = Models::squeezenet1_1;
    // TODO For preprocessing model with width and height % 2 == 0 required
    Models::ModelDesc modelForPreprocessing = Models::googlenet_v1;

    const size_t numberOfTopClassesToCompare = 5;
};

//------------------------------------------------------------------------------
using ImageWorkload_WithoutPreprocessing = ImageWorkload_Tests;
TEST_F(ImageWorkload_WithoutPreprocessing, precommit_SyncInference) {
    const Models::ModelDesc modelToUse = modelForNoPreprocess;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}

/** @brief Validate repacking from NCHW to NHWC */
TEST_F(ImageWorkload_WithoutPreprocessing, precommit_SyncInferenceNCHWInput) {
    const Models::ModelDesc modelToUse = modelForNoPreprocess;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;

    // Load image in different layout to validate repacking
    auto inputNCHWBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NCHW, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputNCHWBlob));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto inputNHWCBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, true);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputNHWCBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}

#ifdef USE_OPENCV

// Case: Image with 256B alignment - not supported w/o preprocessing + VideoWorkload
TEST_F(ImageWorkload_WithoutPreprocessing, precommit_ImageWithStrides_ThrowException) {
     const Models::ModelDesc modelToUse = modelForNoPreprocess;

    // ---- Load image and add strides (227x227 image)
    const size_t imageWidth = modelToUse.width;
    const size_t imageHeight = modelToUse.height;
    const size_t alignFactor = 256;
    const size_t paddingWidth = alignFactor - imageWidth;
    const size_t planes = 3;
    const std::string imagePath = TestDataHelpers::get_data_path() + "/"+  std::to_string(imageWidth) + "x" + std::to_string(imageHeight) + "/cat3.bmp";
    cv::Mat originalImage = cv::imread(imagePath);
    if(originalImage.empty()) {
        IE_THROW() << "Could not read the image: " << imagePath;
    }

    // ---- Add padding to image at right side
    const size_t blobSize = (imageWidth + paddingWidth) * imageHeight * planes;
    std::vector<uint8_t> output(blobSize);
    cv::Mat dstImg(imageHeight, imageWidth + paddingWidth, originalImage.type(), output.data());
    cv::copyMakeBorder(originalImage, dstImg, 0, 0, 0, paddingWidth, cv::BORDER_WRAP);

    // ---- Create blob with NHWC layout
    const IE::SizeVector dims {1, planes, imageHeight, imageWidth};
    const IE::SizeVector strides { imageHeight * (imageWidth + paddingWidth) * planes, (imageWidth + paddingWidth) * planes, planes, 1};
    IE::BlockingDesc blockingDesc({1,imageHeight, imageWidth, planes}, {0, 2, 3, 1}, 0, {0,0,0,0}, strides);
    IE::TensorDesc tensorDesc(IE::Precision::U8, dims, blockingDesc);
    auto inputBlob = IE::make_shared_blob<uint8_t>(tensorDesc);
    inputBlob->allocate();
    {
        auto blobPtr = inputBlob->buffer().as<uint8_t*>();
        std::copy_n(output.data(), inputBlob->size(), blobPtr);
    }

    // ---- Create executable network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- Set input blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Run the request synchronously
    ASSERT_ANY_THROW(inferRequest.Infer());
}

#endif

//------------------------------------------------------------------------------
class ImageWorkload_WithPreprocessing : public ImageWorkload_Tests {
public:
    const std::string inputNV12Path = TestDataHelpers::get_data_path() + "/" +
                                      std::to_string(modelForPreprocessing.width) + "x" +
                                      std::to_string(modelForPreprocessing.height) + "/cat3.yuv";
};

TEST_F(ImageWorkload_WithPreprocessing, precommit_SyncInference) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Load NV12 Image and create blob from it
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;

    // ----- Load NV12 input
    std::vector<uint8_t> nv12InputBlobMemory;
    nv12InputBlobMemory.resize(inputWidth * inputHeight * 3 / 2);
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                                        inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

    // Preprocessing
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set NV12 blob with preprocessing information
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, true, &preprocInfo);

    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}

TEST_F(ImageWorkload_WithPreprocessing, precommit_SyncInference_RGBToBGR) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    // ---- Import or load network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    // ---- Set RGB input
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    const auto isBGR = false;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, isBGR);

    // ---- Preprocessing
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::PreProcessInfo preProcInfo = inferRequest.GetPreProcess(inputName);
    preProcInfo.setColorFormat(IE::ColorFormat::RGB);

    // ---- Set RGB blob with preprocessing information
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preProcInfo));

    // ---- Run the request synchronously
    ASSERT_NO_THROW(inferRequest.Infer());

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob - Same image, BGR instead of RGB
    const bool isBGRforCPU = true;
    auto inputBlobCPU = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, isBGRforCPU);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlobCPU);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}

#ifdef USE_OPENCV

// Case: Image with 256B alignment - not supported w/o preprocessing + VideoWorkload
TEST_F(ImageWorkload_WithPreprocessing, precommit_ImageWithStrides_ThrowException) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;

    // ----- Load NV12 input image
    const size_t imageWidth = modelToUse.width;
    const size_t imageHeight = modelToUse.height;
    const size_t alignFactor = 256;
    const size_t paddingWidth = alignFactor - imageWidth;
    const size_t yPlanes = 1;
    const size_t uvPlanes = 2;
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(
                                            inputNV12Path, imageWidth, imageHeight);

    // Create OpenCV image from Y plane
    std::vector <uint8_t> yPlaneOrigData;
    {
    const auto lockedMemory = IE::as<IE::MemoryBlob>(nv12InputBlob->y())->rmap();
    const auto data = lockedMemory.as<uint8_t*>();
    yPlaneOrigData.assign(data, data + nv12InputBlob->y()->byteSize());
    }
    cv::Mat originalImage = cv::Mat(imageHeight, imageWidth, CV_8UC1, yPlaneOrigData.data());

    // Add padding to Y plane at right side
    const size_t blobSize = (imageWidth + paddingWidth) * imageHeight * yPlanes;
    std::vector<uint8_t> yPlaneDstData(blobSize);
    cv::Mat dstImg(imageHeight, imageWidth + paddingWidth, CV_8UC1, yPlaneDstData.data());
    cv::copyMakeBorder(originalImage, dstImg, 0, 0, 0, paddingWidth, cv::BORDER_WRAP);

    // Create blob for Y plane with padding
    IE::SizeVector yPlaneDims {1, yPlanes, imageHeight, imageWidth + paddingWidth};
    IE::TensorDesc yPlaneTensorDesc(IE::Precision::U8, yPlaneDims, IE::Layout::NHWC);
    IE::Blob::Ptr yPlaneInputBlob = IE::make_shared_blob<uint8_t>(yPlaneTensorDesc);
    yPlaneInputBlob->allocate();
    {
        auto blobPtr = yPlaneInputBlob->buffer().as<uint8_t*>();
        std::copy_n(yPlaneDstData.data(), yPlaneInputBlob->size(), blobPtr);
    }

    // Make fictive gray-scale uvPlane with padding
    IE::SizeVector uvPlaneDims {1, uvPlanes, imageHeight / 2, (imageWidth + paddingWidth) / 2};
    IE::TensorDesc uvPlaneTensorDesc(IE::Precision::U8, uvPlaneDims, IE::Layout::NHWC);
    const int64_t grayConst = 0x80;
    IE::Blob::Ptr uvPlaneInputBlob = vpux::makeSplatBlob(uvPlaneTensorDesc, grayConst);

    // ---- Create NV12 blob with ROI for strides processing
    IE::ROI yPlaneRoi {0, 0, 0, imageWidth, imageHeight};
    IE::ROI uvPlaneRoi {0, 0, 0, imageWidth / 2, imageHeight / 2};
    IE::Blob::Ptr yRoiBlob = IE::make_shared_blob(yPlaneInputBlob, yPlaneRoi);
    IE::Blob::Ptr uvRoiBlob = IE::make_shared_blob(uvPlaneInputBlob, uvPlaneRoi);
    IE::Blob::Ptr inputBlob = IE::make_shared_blob<IE::NV12Blob>(yRoiBlob, uvRoiBlob);

    // ---- Create executable network
    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);

    // ---- Create infer request
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- Preprocessing
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputBlobName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);

    // ---- Set NV12 blob with preprocessing information
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preprocInfo));

    // ---- Run the request synchronously
    ASSERT_ANY_THROW(inferRequest.Infer());
}

#endif

//------------------------------------------------------------------------------
class ImageWorkload_SpecificCases : public ImageWorkload_Tests {
public:
    const std::string inputNV12Path = TestDataHelpers::get_data_path() + "/" +
                                      std::to_string(modelForPreprocessing.width) + "x" +
                                      std::to_string(modelForPreprocessing.height) + "/cat3.yuv";
};

/** @brief Execute inference with preprocessing and after that without preprocessing */
TEST_F(ImageWorkload_SpecificCases, precommit_WithoutPreprocessingAndPreprocessing) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- Without preprocessing - set blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, true);
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob));

    // ---- Without preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);
    IE::Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);
    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));


    // ---- With preprocessing - set blob
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    std::vector<uint8_t> nv12InputBlobMemory;
    nv12InputBlobMemory.resize(inputWidth * inputHeight * 3 / 2);
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- With preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    outputBlob = inferRequest.GetBlob(outputBlobName);
    refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, true, &preprocInfo);
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));;
}

/** @brief Execute inference without preprocessing and after that with preprocessing  */
TEST_F(ImageWorkload_SpecificCases, precommit_PreprocessingAndWithoutPreprocessing) {
    const Models::ModelDesc modelToUse = modelForPreprocessing;
    const size_t inputWidth = modelToUse.width;
    const size_t inputHeight = modelToUse.height;

    IE::ExecutableNetwork executableNetwork =
            ExecutableNetworkFactory::createExecutableNetwork(modelToUse.pathToModel);
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    // ---- With preprocessing - set blob
    auto inputName = executableNetwork.GetInputsInfo().begin()->first;
    std::vector<uint8_t> nv12InputBlobMemory;
    nv12InputBlobMemory.resize(inputWidth * inputHeight * 3 / 2);
    IE::NV12Blob::Ptr nv12InputBlob = NV12Blob_Creator::createFromFile(inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());
    IE::PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setColorFormat(IE::ColorFormat::NV12);
    inferRequest.SetBlob(inputName, nv12InputBlob, preprocInfo);

    // ---- With preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);
    auto refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, nv12InputBlob, true, &preprocInfo);
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));;

    // ---- Without preprocessing - set blob
    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = IE_Core_Helper::loadImage("cat3.bmp", inputWidth, inputHeight, IE::NHWC, true);
    IE::PreProcessInfo preprocInfoDefault;
    ASSERT_NO_THROW(inferRequest.SetBlob(inputBlobName, inputBlob, preprocInfoDefault));

    // ---- Without preprocessing - Infer and compare result
    ASSERT_NO_THROW(inferRequest.Infer());
    outputBlob = inferRequest.GetBlob(outputBlobName);
    refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelToUse.pathToModel, inputBlob);
    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_NO_THROW(
        Comparators::compareTopClassesUnordered(
                    vpux::toFP32(IE::as<IE::MemoryBlob>(outputBlob)),
                    vpux::toFP32(IE::as<IE::MemoryBlob>(refBlob)),
                    numberOfTopClassesToCompare));
}
