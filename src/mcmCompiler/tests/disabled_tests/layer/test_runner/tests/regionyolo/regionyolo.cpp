//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/print_info_pass.hpp"

int main()
{
    mv::CompilationUnit unit("RegionYolo");
    mv::OpModel& om = unit.model();
    auto input0 = om.input("input0", {13,13,125,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    // Define Params
    unsigned coords = 4;
    unsigned classes = 20;
    bool do_softmax = true;
    unsigned num = 5;
    std::vector<unsigned> mask;
    auto regionyolo0 = om.regionYolo("", input0, coords, classes, do_softmax, num, mask);
    om.output("", regionyolo0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Silent")));
    unit.compilationDescriptor().addToGroup("serialize", "PrintInfo", "Singular", false);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
