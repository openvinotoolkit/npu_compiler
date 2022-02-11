//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/hwtest/test_case_json_parser.hpp"

#include <gtest/gtest.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <cstdlib>

void createCaseGeneratorHeaderJson(llvm::json::OStream& j) {
    j.attribute("case_type", "ZMajorConvolution");
    j.attribute("network", "");
    j.attribute("layer_name", "conv2d_u8_to_u8_unit_test");
}

void createCaseGeneratorInputJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {1, 256, 16, 16};
    std::string dtype = "uint8";
    double scale = 0.01;
    unsigned long int zeropoint = 127;
    unsigned long int low_range = 0;
    unsigned long int high_range = 255;

    j.attributeBegin("input");
    {
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attribute("scale", scale);
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorWeightsJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {64, 256, 1, 1};
    std::string dtype = "uint8";
    double scale = 0.01;
    unsigned long int zeropoint = 0;
    unsigned long int low_range = 1;
    unsigned long int high_range = 1;

    j.attributeBegin("weight");
    {
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attribute("scale", scale);
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorOutputJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {1, 64, 16, 16};
    std::string dtype = "uint8";
    double scale = 0.01;
    unsigned long int zeropoint = 0;
    unsigned long int low_range = 1;
    unsigned long int high_range = 1;

    j.attributeBegin("output");
    {
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attribute("scale", scale);
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorConvJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 2> stride = {1, 1};
    std::array<unsigned long int, 2> pad = {0, 0};
    unsigned long int group = 1;
    unsigned dilation = 1;

    j.attributeBegin("conv_op");
    {
        j.objectBegin();

        j.attributeBegin("stride");
        {
            j.arrayBegin();
            for (auto s : stride)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attributeBegin("pad");
        {
            j.arrayBegin();
            for (auto s : pad)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("group", group);
        j.attribute("dilation", dilation);

        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorODUPermutationJson(llvm::json::OStream& j) {
    std::string order("nhwc");
    j.attribute("output_order", order);
}

void createCaseGeneratorActivationJson(llvm::json::OStream& j) {
    j.attributeBegin("activation");
    {
        j.objectBegin();
        j.attribute("name", nullptr);
        j.objectEnd();
    }
    j.attributeEnd();
}

void createAndRunConvTest() {
    auto testConfigFile = "conv_test.json";

    auto createNumericsBenchJsonSpec = [&]() {
        std::error_code ec;
        llvm::raw_fd_ostream jsonFd(testConfigFile, ec);
        llvm::json::OStream j(jsonFd);

        j.objectBegin();

        createCaseGeneratorHeaderJson(j);
        createCaseGeneratorInputJson(j);
        createCaseGeneratorWeightsJson(j);
        createCaseGeneratorConvJson(j);
        createCaseGeneratorOutputJson(j);
        createCaseGeneratorODUPermutationJson(j);
        createCaseGeneratorActivationJson(j);

        j.objectEnd();
    };

    createNumericsBenchJsonSpec();

    std::ifstream in_file(testConfigFile);
    std::stringstream in_file_buffer;
    in_file_buffer << in_file.rdbuf();

    nb::TestCaseJsonDescriptor desc(in_file_buffer.str());

    nb::InputLayer input = desc.getInputLayer();
    ASSERT_EQ(nb::to_string(input.dtype), "uint8");

    nb::WeightLayer weight = desc.getWeightLayer();
    ASSERT_EQ(weight.qp.scale, 0.01);
}

TEST(MTL_JSON_Parser, conv_test) {
    createAndRunConvTest();
}
