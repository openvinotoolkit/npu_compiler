#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

// This example demonstrates the DPUConvolution pass:
// which replaces all convolution operations with DPU tasks,
// and adds appropriate DMA tasks (for DDR-to-CMX and back),
// and de-allocation tasks for the temporary CMX buffers.

int main()
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({56, 56, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float8"), mv::Order::getZMajorID(4));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> biasWeightsData = mv::utils::generateSequence<double>(64);
    auto biasWeights = om.constant(biasWeightsData, {64}, mv::DType("Float8"), mv::Order::getColMajorID(1));

    auto bias = om.bias(conv, biasWeights);

    om.output(bias);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}
