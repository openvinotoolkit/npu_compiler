#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void concatAsImplicitFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConcatAsImplicit)
            .setFunc(concatAsImplicitFcn)
            .setDescription(
                "Replaces all concats with implicits concats");
    }
}

void concatAsImplicitFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    auto concats = om.getOps("Concat");

    for(auto& concat: concats)
    {
        auto inputs = concat->getInputTensor();
        auto axis = concat->get<std::string>("axis");
        auto name = concat->getName();
        mv::QuantizationParams quantParams = {{}, {}, {}, {}};
        if(concat->hasAttr("quantParams"))
            quantParams = concat->get<mv::QuantizationParams>("quantParams");
        auto outputLocation = concat->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto opId = concat->get<unsigned>("opId");
        auto outputFlows = mv::getOutputDataFlow(om, concat);
        auto implicitConcat = om.implicitConcat(inputs, axis, quantParams, name);
        om.getSourceOp(implicitConcat)->set<unsigned>("opId", opId);
        implicitConcat->set<mv::Tensor::MemoryLocation>("Location", outputLocation);

        mv::setOutputDataFlow(om, implicitConcat, outputFlows);
    }
}
