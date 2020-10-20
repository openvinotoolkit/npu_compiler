#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("FlattenModel");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({300,300,3,1});
    auto inputOrder = mv::Order("NHWC");

    // outputShape = {270000,1,1,1}
    //auto axis = 1;
    //auto end_axis = -1;

    // outputShape = {300,900,1,1}
    auto axis = 1;
    auto end_axis = 2;

    // outputShape = {300,300,3,1}
    //auto axis = 1;
    //auto end_axis = 1;

    // outputShape = {90000,3,1,1}
    //auto axis = 2;
    //auto end_axis = 3;

    // outputShape = {300,300,3,1}
    //auto axis = 2;
    //auto end_axis = 2;

    // invalid axis
    //auto axis = 4;
    //auto end_axis = -1;

    // invalid end_axis
    //auto axis = 1;
    //auto end_axis = -4;

    // invalid combo (axis !<= end_axis)
    //auto axis = 3;
    //auto end_axis = 2;

    // Unrealistic implicit reshape case
    //inputShape = mv::Shape({270000,1,1,1});
    //auto axis = 1;
    //auto end_axis = -1;


    auto input0 = om.input("input0", inputShape, mv::DType("Float16"), inputOrder);
    auto flatten0 = om.flatten("", input0, axis, end_axis);
    om.output("", flatten0);
    
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
