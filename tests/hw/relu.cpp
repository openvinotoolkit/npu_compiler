#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"

TEST(mvtensor_serialization, blob_relu1)
{
    mv::CompilationUnit unit("relu1");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({32, 32, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto relu = om.relu(input);
    auto output = om.output(relu);

    std::string outputName = "test_relu1";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string(outputName+".blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName+".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    mv::ReturnCodes results = mv::HWTest(unit, outputName);
    ASSERT_EQ(results.mcmBlobOnHardware, 0);
    ASSERT_EQ(results.fathomCompilation, 0);
    ASSERT_EQ(results.fathomVsCaffe, 0);
    ASSERT_EQ(results.fathomVsMcm, 0);

}

TEST(mvtensor_serialization, blob_relu2)
{
    mv::CompilationUnit unit("relu2");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto relu = om.relu(input);
    auto output = om.output(relu);

    std::string outputName = "test_relu2";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string(outputName+".blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName+".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    mv::ReturnCodes results = mv::HWTest(unit, outputName);
    ASSERT_EQ(results.mcmBlobOnHardware, 0);
    ASSERT_EQ(results.fathomCompilation, 0);
    ASSERT_EQ(results.fathomVsCaffe, 0);
    ASSERT_EQ(results.fathomVsMcm, 0);

}

TEST(mvtensor_serialization, blob_relu3)
{
    mv::CompilationUnit unit("relu3");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({224, 200, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto relu = om.relu(input);
    auto output = om.output(relu);

    std::string outputName = "test_relu3";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = std::string(outputName+".blob");
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName+".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string(outputName + ".prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string(outputName + ".caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    mv::ReturnCodes results = mv::HWTest(unit, outputName);
    ASSERT_EQ(results.mcmBlobOnHardware, 0);
    ASSERT_EQ(results.fathomCompilation, 0);
    ASSERT_EQ(results.fathomVsCaffe, 0);
    ASSERT_EQ(results.fathomVsMcm, 0);

}

