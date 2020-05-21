//This file is the parsed network which is created through python.

// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const auto Parameter_0 = model.input({32, 32, 16, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, true, "Parameter_0");
    const auto Parameter_1 = model.input({32, 32, 32, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, true, "Parameter_1");
    auto identity_maxPool_0 = model.maxPool(Parameter_0, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "identity_maxpool_0");
    auto identity_maxPool_1 = model.maxPool(Parameter_1, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "identity_maxpool_1");
    const auto Concat_0 = model.concat({identity_maxPool_0, identity_maxPool_1}, "C", mv::DType("Default"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "Concat_0");
    const auto output = model.output(Concat_0);

}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
