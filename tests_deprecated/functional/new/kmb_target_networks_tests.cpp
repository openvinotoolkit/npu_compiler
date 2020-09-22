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

#include <file_utils.h>

#include "test_model/kmb_test_base.hpp"

//
// ResNet50 FP16 IRv10
//
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_fp16_IRv10) {
    // [Track number: D#3222]
    SKIP_ON("KMB", "HDDL2", "VPUX", "MemoryAllocator:VPU_DDR_Heap - ArgumentError");
    runTest(
        TestNetworkDesc("KMB_models/FP16/resnet_50_pytorch/resnet-50-pytorch.xml")
            .setUserInputPrecision("input", Precision::FP16)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_ResNet_50) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet-50-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 2.5f);
}

TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_PyTorch_IRv7_ResNet_50) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet50_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/husky.bmp", ImageFormat::RGB),
        1, 0.7f);
}

// KMB : Hangs on infer stage [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV1_TF_IRv7_ResNet_50) {  // 28.4% sparsity
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v1-tf-0001.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// KMB : Hangs on infer stage
// [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_TF_IRv7_ResNet_50) {  // 60.4% sparsity
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// KMB : Bad inference results.
// Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_ONNX_IRv7_ResNet_50) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "Compiler Error: min > max");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-onnx/resnet50-int8-sparse-v2.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

//
// MobileNetV2
//

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-caffe-IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16)
            .setUserOutputLayout("output", Layout::NHWC),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        2, 0.7f);
}

// CPU : Supported primitive descriptors list is empty for node: Add1_/Fused_Add_
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        3, 2.15f);
}

TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_PyTorch_IRv7_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_SparseV1_TF_IRv7_MobileNet_V2) {  // 30.8% sparsity
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v1-tf-0001.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// KMB : Bad inference results.
// [Track number: D#2246 D#2691]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_TF_IRv7_MobileNet_V2) {  // 59.3% sparsity
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results, mixed up top2 classes");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// KMB : Bad inference results.
// [Track number: D#2246]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_ONNX_IRv7_MobileNet_V2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
	    TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 0.05f);
}

//
// InceptionV1
//

// KMB : Op:pool5/7x7_s1 - OpError: Invalid input data (0) - Filter kernel width (7) exceeds the padded input width (6)
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_Inception_V1) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1-caffe-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        3, 1e-2f);
}

// KMB : Test on caffe based inception_v1 fails on IE to mcmCompiler parsing stage
// KMB : C++ exception with description "Op:pool5/7x7_s1 - OpError: Invalid input data (0) -
// KMB : Filter kernel width (7) exceeds the padded input width (6)
// [Track number: S#25483/D#2374]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv7_Inception_V1) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// KMB : Hangs on infer stage
// [Track number: D#2293]
TEST_F(KmbClassifyNetworkTest, INT8_Sparse_TF_IRv7_Inception_V1) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

//
// InceptionV3
//

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_Inception_V3) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 1e-1f);
}

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv10_Inception_V3) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-tf-frozen-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 1e-1f);
}

TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_TF_IRv7_Inception_V3) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.05f);
}

//
// SqueezeNet 1.1
//

// FIXME: Missing IR in models-ir repository
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_Caffe2_IRv10_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1-caffe2-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        3, 1e-1f);
}

// KMB : Following test on caffe based squeezenet1_1 fails on IE to mcmCompiler parsing stage
// KMB : with message
// KMB : C++ exception with description "Op:pool10 - OpError: Invalid input data (0) -
// KMB : Filter kernel width (14) exceeds the padded input width (13)
// [Track number: S#25483/D#2374]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv7_SqueezeNet_1_1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16)
            .setUserOutputLayout("input", Layout::NHWC),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_SqueezeNet_1_1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 2.f);
}

// [Track number: D#3052]
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Sparse_ONNX_IRv7_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

//
// TinyYolo V1
//

TEST_F(KmbYoloV1NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/tiny_yolo_v1/tiny_yolo_v1_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, true);
}

//
// TinyYolo V2
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV2) {
    // Track number: H#18012088819
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

//
// TinyYolo V2 with custom region
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV2_Custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
        getIELibraryPath() + "/kmb_custom_kernels/yolov2.xml");

    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({customLayers}),
        TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

//
// Yolo V2
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_YoloV2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPU", "bad results");  // TODO: create JIRA ticket
    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

//
// Yolo V2 with custom region & reorg
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_YoloV2_Custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
        getIELibraryPath() + "/kmb_custom_kernels/yolov2.xml");
    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({customLayers}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}


//////////////////////////////////////////
// Start of test-set for KMB-alpha IRv10
//////////////////////////////////////////

TEST_F(KmbYoloV2NetworkTest, yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

// KMB : Bad inference results. Possible bug in test system.
// [Track number: S#28790]
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_tf_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

const static std::vector<InferenceEngine::Layout> specificLayout = {
        InferenceEngine::Layout::NHWC,
        InferenceEngine::Layout::NCHW};

class KmbClassifyNetworkTestWithSpecificLayout : public KmbClassifyNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};

TEST_P(KmbClassifyNetworkTestWithSpecificLayout, precommit_resnet_50_pytorch_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", GetParam())
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/husky.bmp", ImageFormat::RGB),
        1, 0.7f);
}

INSTANTIATE_TEST_CASE_P(precommit, KmbClassifyNetworkTestWithSpecificLayout, ::testing::ValuesIn(specificLayout));

TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("224x224/husky.bmp", ImageFormat::RGB),
        1, 0.7f);
}

TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 7.0f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 7.0f);
}


TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v1_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v1_tf_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v3_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.05f);
}


TEST_F(KmbClassifyNetworkTest, googlenet_v3_tf_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.1f);
}

TEST_F(KmbClassifyNetworkTest, precommit_squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        1, 2.0f);
}

// C++ exception with description "propagateParameters ERROR: inputs of the Eltwise/Concat do not have the same QuantParams"
// [Track number: S#31766]
TEST_F(KmbClassifyNetworkTest, squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_ngraph) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(YES)}}),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        1, 2.0f);
}
//////////////////////////////////////////
// End of test-set for KMB-alpha IRv10
//////////////////////////////////////////


//////////////////////////////////////////
// Start of test-set for KMB-beta IRv10
//////////////////////////////////////////

// C++ exception with description "Caught exception during unit run: Wrong strategy generated:
// tensor fire6/squeeze1x1:0_crop:0_align:0 needs sparsity but it can't be sparsified" thrown in the test body.
// [Track number: D#3467]
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nchw) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "Compilation fails");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW),
            TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nhwc) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC),
            TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

// C++ exception with description "Caught exception during unit run: Wrong strategy generated:
// tensor fire6/squeeze1x1:0_crop:0_align:0 needs sparsity but it can't be sparsified" thrown in the test body.
// [Track number: D#3467]
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nchw_fuse_scale_input_accuracy_drop) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "Compilation fails");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nhwc_fuse_scale_input_accuracy_drop) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            // [Track number: D#3634]
            .setCompileConfig({{"VPU_COMPILER_SCALE_FUSE_INPUT", CONFIG_VALUE(NO)}}),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

TEST_F(KmbSSDNetworkTest, precommit_ssd512_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd512/ssd512_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "Only single input is supported currently
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:785
// [Track number: D#2723]
TEST_F(KmbDetectionNetworkTest, precommit_faster_rcnn_resnet101_coco_tf_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("1024x600/frankfurt_001016.jpg", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v4_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            1, 0.06f);
}

TEST_F(KmbSSDNetworkTest, precommit_ssd_mobilenet_v1_coco_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

// Interrupted by signal 6: SIGABRT
// KmbFunctionalTests: kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1635:
// void vpu::FrontEndMcm::parseNormalize(const CNNLayerPtr&, const McmNodeVector&):
// Assertion `(dims[1] == weightsSize)' failed.
// [Track number: D#2918]
TEST_F(KmbClassifyNetworkTest, precommit_facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/facenet-20180408-102900/facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("160x160/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// C++ exception with description "ELU layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1604
// [Track number: D#2725]
TEST_F(KmbDetectionNetworkTest, precommit_person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("1024x1024/frankfurt_001016.png", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbDetectionNetworkTest, precommit_vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("736x416/dss_val_05.png", ImageFormat::BGR),
            0.3f,
            0.1f, 0.3f);
}

// FIXME change adapter to Yolo V3 when available
// [Track number: H#1801262299]
TEST_F(KmbYoloV2NetworkTest, person_vehicle_bike_detection_crossroad_yolov3_1020) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "output strides are set to represent NHWC");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/"
                    "person-vehicle-bike-detection-crossroad-yolov3-1020/"
                    "person-vehicle-bike-detection-crossroad-yolov3-1020.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("500x500/car_fcn8.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

// C++ exception with description "PriorBoxClustered layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1779
// [Track number: D#2727]
TEST_F(KmbDetectionNetworkTest, precommit_face_detection_retail_0004_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/face_detection_retail_0004_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbClassifyNetworkTest, precommit_resnet_101_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, precommit_resnet_152_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// C++ exception with description "Op:conv2 - OpError: Invalid input weights (1) -
// Does not match the channel dimension of input 96
// [Track number: D#2799]
TEST_F(KmbClassifyNetworkTest, precommit_alexnet_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/alexnet/alexnet_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// Compilation time is very long in comparison with other networks
// [Track number: S#28620]
TEST_F(KmbClassifyNetworkTest, precommit_vgg16_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "very long compile time");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

TEST_F(KmbRetinaFaceNetworkTest, precommit_retinaface_mobilenetv2_0_25_modified) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/private/retinaface-mobilenetv2-0.25-modified/retinaface-mobilenetv2-0.25-modified.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_kmb_retinaface"}}),
            "data",
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB));
}

//////////////////////////////////////////
// End of test-set for KMB-beta IRv10
//////////////////////////////////////////

////////////////////////////////////////////////////////////
// Start of test-set for IRv10 FP16 to INT8 quantization
////////////////////////////////////////////////////////////

// C++ exception with description "Layer Power_123537 supports only power = 1
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1464
// [Track number: D#2809]
TEST_F(KmbYoloV2NetworkTest, yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")

                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbClassifyNetworkTest, resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.62f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setUserOutputLayout("output", Layout::NHWC),
            TestImageDesc("227x227/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, DISABLED_googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, resnet_101_caffe_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, resnet_152_caffe_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

// Compilation time is very long - more than 40 minutes.
// The same situation as for vgg16_caffe_dense_int8_IRv10
// [Track number: S#28620]
TEST_F(KmbClassifyNetworkTest, vgg16_caffe_dense_int8_IRv10_fp16_to_int8) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "very long compile time");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// Compilation fails with exception:
// "Caught exception during unit run: propagateParameters ERROR:
// inputs of the Eltwise/Concat do not have the same QuantParams"
// [Track number: D#3453]
TEST_F(KmbRFCNNetworkTest, rfcn_resnet50_caffe_IRV10_fp16_int8) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "Compilation fails");
    // [Track number: S#3331]
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    const std::string data_name    = "data";
    const std::string im_info_name = "im_info";

    runTest(
        TestNetworkDesc("KMB_models/INT8/private/rfcn-resnet50/caffe/FP16-INT8/rfcn-resnet50_ww22.xml")
            .setUserInputPrecision(data_name, Precision::U8)
            .setUserInputLayout(data_name, Layout::NCHW)
            .setUserInputPrecision(im_info_name, Precision::FP32)
            .setUserInputLayout(im_info_name, Layout::NC)
            .setUserOutputPrecision("cls_prob_reshape",  Precision::FP32)
            .setUserOutputPrecision("bbox_pred_reshape", Precision::FP32),
        "data",
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        "im_info",
        {224.f, 224.f, 1.f});
}
////////////////////////////////////////////////////////////
// End of test-set for IRv10 FP16 to INT8 quantization
////////////////////////////////////////////////////////////

TEST_F(KmbClassifyNetworkTest, emotion_recognition_retail_0003) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/emotions-recognition-retail-0003/emotions-recognition-retail-0003_int8_from_fp16.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "vpu/emotions-recognition-retail-0003.png",
        2, 0.1f);
}

TEST_F(KmbSegmentationNetworkTest, icnet_camvid_ava_0001) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/icnet-camvid-ava-tf-0001/icnet_camvid_ava_tf_0001_tf_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::CHW)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("1024x1024/frankfurt_001016.png", ImageFormat::RGB),
        0.3f);  // mean intersection over union tolerance
}

TEST_F(UnetNetworkTest, precommit_unet_camvid_ava_0001_NHWC_NCHW) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/unet-camvid-onnx-0001/caffe2/FP16-INT8/unet_camvid_onnx_0001_WW34.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::NCHW),
        TestImageDesc("480x360/0016E5_07959.png", ImageFormat::RGB),
        0.3f);  // mean intersection over union tolerance
}

TEST_F(UnetNetworkTest, unet_camvid_ava_0001_NCHW_NCHW) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/unet-camvid-onnx-0001/caffe2/FP16-INT8/unet_camvid_onnx_0001_WW34.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputLayout("output", Layout::NCHW),
        TestImageDesc("480x360/0016E5_07959.png", ImageFormat::RGB),
        0.3f);  // mean intersection over union tolerance
}

TEST_F(GazeEstimationNetworkTest, gaze_estimation_adas_0002) {
    const auto left_eye_input_name = "left_eye_image";
    const auto right_eye_input_name = "right_eye_image";
    const auto head_pos_input_name = "head_pose_angles";

    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/gaze-estimation-adas-0002/gaze_estimation_adas_0002_int8_from_fp16_ww22.xml")
            .setUserInputPrecision(left_eye_input_name, Precision::U8)
            .setUserInputLayout(left_eye_input_name, Layout::NHWC)
            .setUserInputPrecision(right_eye_input_name, Precision::U8)
            .setUserInputLayout(right_eye_input_name, Layout::NHWC)
            .setUserInputPrecision(head_pos_input_name, Precision::FP32)
            .setUserInputLayout(head_pos_input_name, Layout::NC)
            .setUserOutputPrecision("output", Precision::FP32),
        left_eye_input_name,
        "vpu/gm_0000_left.png",
        right_eye_input_name,
        "vpu/gm_0000_right.png",
        head_pos_input_name,
        std::vector<float>{-2.076815605163574, -2.1021695137023926, 0.13159990310668945});
}

TEST_F(SmokeNetworkTest, openpose_pose_cf_NHWC) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/OpenPose/FP16-INT8/openpose-pose_cf_ww22.xml")
            .setUserInputPrecision("image", Precision::U8)
            .setUserInputLayout("image", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32));
}

TEST_F(SmokeNetworkTest, DISABLED_openpose_pose_cf_NCHW) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/OpenPose/FP16-INT8/openpose-pose_cf_ww22.xml")
            .setUserInputPrecision("image", Precision::U8)
            .setUserInputLayout("image", Layout::NCHW)
            .setUserOutputPrecision("output", Precision::FP32));
}

TEST_F(AgeGenderNetworkTest, precommit_age_gender_retail_0013) {
    const std::string input_name = "input";

    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/age-gender-recognition-retail-0013/caffe/FP16-INT8/age-gender-recognition-retail-0013_ww22.xml")
            .setUserInputPrecision(input_name, Precision::U8),
        TestImageDesc("62x62/face62.bmp", ImageFormat::RGB),
        0.1f);
}

// C++ exception with description "Cannot convert layer "GatherIE_6126" due to unsupported layer type "Gather"
// [Track number: S#31241]
TEST_F(PersonAttrRecNetworkTest, person_attributes_recognition_crossroad_0234) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0234.xml"),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR),
        0.1f);
}

// C++ exception with description "Cannot convert layer "GatherIE_6126" due to unsupported layer type "Gather"
// [Track number: S#31241]
TEST_F(PersonAttrRecNetworkTest, person_attributes_recognition_crossroad_0238) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0238.xml"),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR),
        0.1f);
}

// [Track number: D#3604]
TEST_F(KmbSSDNetworkTest, ssdlite_mobilenet_v2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.xml")
            .setUserInputPrecision("image_tensor", Precision::U8),
        TestImageDesc("300x300/dog.bmp", ImageFormat::BGR),
        0.3f,
        0.1f, 0.3f);
}

TEST_F(VehicleAttrRecNetworkTest, vehicle_attributes_recognition_barrier_0042) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/"
                        "vehicle-attributes-recognition-barrier-0042/"
                        "vehicle-attributes-recognition-barrier-0042.xml")
            .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("500x500/test.bmp", ImageFormat::BGR),
        0.25f);
}

// C++ exception with description "Op:L0067_AddBackward1 - OpError: Invalid input inputs (0) -
// All the inputs of eltwise ops have to share the same size
// or the other inputs must have size 1 and be populated
// [Track number: D#3627]
TEST_F(KmbSegmentationNetworkTest, road_segmentation_adas_0001) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/road_segmentation_adas_0001/road-segmentation-adas-0001.xml"),
        TestImageDesc("512x896/road-segmentation-adas-0001.png", ImageFormat::BGR),
        0.3f);
}

// C++ exception with description "Caught exception during unit run: MemoryAllocator:VPU_CMX_NN - ArgumentError:
// conv4_3_0_norm_mbox_locNeutral_copy0conv4_3_0_norm_mbox_locNeutral_copyDMAconv5_5/sep/bn/variance/Fused_Add_:0:0:0::paddedShape[2]
// 192 - Does not match the dimension 184 of the tensor conv4_3_0_norm_mbox_locNeutral:0 already allocated in the given buffer
// // [Track number: D#3656]
TEST_F(KmbDetectionNetworkTest, face_detection_adas_0001) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/face-detection-adas-0001/face-detection-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8)
	    .setUserInputLayout("input", Layout::NHWC),
        TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::BGR),
        0.3f,
        1.f, 0.3f);
}

TEST_F(HeadPoseEstimationNetworkTest, head_pose_estimation_adas_0001) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/head_pose_estimation_adas_0001/head-pose-estimation-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("60x60/head-pose-estimation-adas-0001.png", ImageFormat::BGR),
        0.1f);
}

// C++ exception with description "Tile layer is not supported by kmbPlugin
// [Track number: D#3657]
TEST_F(KmbClassifyNetworkTest, license_plate_recognition_barrier_0007) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "compile error");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("24x94/000000.bmp", ImageFormat::BGR),
        7,
        0.3f);
}
