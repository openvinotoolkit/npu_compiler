#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ConvLeakyReluModel");
    mv::OpModel& om = unit.model();

    double alpha = 0.1;
    //Input full of -0.5s
    auto input0 = om.input("input#170", {16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{255},{0.00196078431372549},{},{}});

    std::vector<int64_t> weightsData0 = {   255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255};

    //Wights first input channel of output channel full of 1s
    auto weights0 = om.constantInt("", weightsData0, {1,1,16,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights0->setQuantParams({{0},{0.00392156862745098},{},{}});
    auto conv0 = om.conv("conv", input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv0->setQuantParams({{255},{0.00196078431372549},{},{}});
    auto leakyRelu0 = om.leakyRelu("leakyRelu0", conv0, alpha);
    leakyRelu0->setQuantParams({{255},{0.00019607843137254904},{},{}});

    om.output("", leakyRelu0, mv::DType("Float16"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
