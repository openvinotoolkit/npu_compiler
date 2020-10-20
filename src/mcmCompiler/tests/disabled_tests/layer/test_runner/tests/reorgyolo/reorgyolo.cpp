#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/print_info_pass.hpp"

int main()
{
    mv::CompilationUnit unit("ReorgYolo");
    mv::OpModel& om = unit.model();
    auto input0 = om.input("input0", {26,26,64,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));

    unsigned stride = 2;

    auto reorgyolo0 = om.reorgYolo("", input0, stride);
    om.output("", reorgyolo0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Silent")));
    unit.compilationDescriptor().addToGroup("serialize", "PrintInfo", "Singular", false);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
