#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void assignInputFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                           mv::Element&, mv::Element&);

namespace mv {

namespace pass {

MV_REGISTER_PASS(AssignInputNode)
        .setFunc(assignInputFcn)
        .setDescription(
                "This pass assigns the input node of the computational data graph, to support multiple inputs.");
}
}  // namespace mv

void assignInputFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                    mv::Element&) {
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    if (om.getNumNetworkInputs() == 0) {
        pass.log(mv::Logger::MessageType::Error, "zero network inputs specified");
        return;
    }

    if (om.getNumNetworkInputs() == 1) {
        om.setInputNode(om.getNetworkInput(0));
        return;
    }

    // Create the virtual graph input node. Size doesn't really matter.
    auto inputTensor = om.input("", mv::Shape({64, 64, 3, 1}), mv::DType("UInt8"), mv::Order("NHWC"), false);

    auto networkInputs = om.getNetworkInputs();
    for (size_t i = 0; i < networkInputs.size(); i++) {
        auto networkInput = networkInputs[i]->getOutputTensor(0);

        auto networkInputOrder = networkInputs[i]->get<mv::Order>("order");

        const auto quantParams = networkInput->getQuantParams();
        const auto tensorDType = networkInput->getDType();
        const auto targetDType = networkInputs[i]->get<mv::DType>("dType");

        networkInput->setDType(targetDType);

        // Create an implicit Input slice op, connect respective network input to that op, and attach.
        auto implicitInputSlice = om.implicitInputSlice("", inputTensor);

        implicitInputSlice->setShape(networkInput->getShape());
        implicitInputSlice->setDType(targetDType);
        implicitInputSlice->setOrder(networkInputOrder);
        implicitInputSlice->setQuantParams(quantParams);

        auto networkInputOp = om.getSourceOp(networkInput);
        // Assumes one input per outputNode
        auto implicitInput =
                om.implicitInput(networkInput->getName() + "_implicit", implicitInputSlice, networkInput->getShape(),
                                 targetDType, networkInputOrder);
        implicitInput->setQuantParams(quantParams);
        implicitInput->setDType(tensorDType);

        implicitInput->set<uint8_t>("inputIndex", i);

        implicitInput->setOrder(mv::Order("NHWC"));

        // connect the output of this implicitInput to be the same as the
        // output of the original input op
        auto parentOp = om.getSourceOp(implicitInputSlice);

        linkNewOperationsReplacement(parentOp, implicitInput, om, networkInputOp);

        om.replaceNetworkInputAtIdx(i, om.getSourceOp(implicitInput));
    }
}
