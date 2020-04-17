
// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"
#include "conv_batch_fasterRCNN.data.inc"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const auto input_3_0 = model.input({7, 7, 2048, 100}, mv::DType("UInt8"), mv::Order("NHWC"), {{128},{0.007843137718737},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input#3");
    const auto conv1_conv1_0_weights_1_0 = model.constantInt(conv1_conv1_0_weights_1_0_data, {1, 1, 2048, 512}, mv::DType("UInt8"), mv::Order("NCHW"), {{127},{0.003638133406639},{-0.463110893964767},{0.464613109827042},{0},{1}}, "conv1/conv1#0_weights#1");
    const auto conv1_conv1_4_0 = model.conv(input_3_0, conv1_conv1_0_weights_1_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv1/conv1#4");
    const auto conv1_conv1_0_bias_2weights_0 = model.constantInt(conv1_conv1_0_bias_2weights_0_data, {512}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000028534379453},{-inf},{inf},{0},{1}}, "conv1/conv1#0_bias#2weights");
    const auto conv1_conv1_0_bias_2_0 = model.bias(conv1_conv1_4_0, conv1_conv1_0_bias_2weights_0, mv::DType("UInt8"), {{0},{0.000028534379453},{-inf},{inf},{0},{1}}, "conv1/conv1#0_bias#2");
    const auto output = model.output(conv1_conv1_0_bias_2_0, mv::DType("Default"), {{},{},{},{}}, true, "");
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_SC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

