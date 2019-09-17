#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

void generateDotFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateDot)
        .setFunc(generateDotFcn)
        .defineArg(json::JSONType::String, "output")
        .defineArg(json::JSONType::String, "scope")
        .defineArg(json::JSONType::String, "content")
        .defineArg(json::JSONType::Bool, "html")
        .setLabel("Debug")
        .setDescription(
            "Generates the DOT representation of computation model"
        );

    }

}

void generateDotFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    if (!passDesc.hasAttr("output") || passDesc.get<std::string>("output").empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate dot pass");

    bool verbose = false;
    if (passDesc.hasAttr("verbose"))
        verbose = passDesc.get<bool>("verbose");

    std::string outputScope = passDesc.get<std::string>("scope");

    if (outputScope != "OpModel" && outputScope != "ExecOpModel" && outputScope != "ControlModel" &&
        outputScope != "OpControlModel" && outputScope != "ExecOpControlModel" && outputScope != "DataModel")
        throw ArgumentError(model, "scope", outputScope, "Invalid model scope");

    std::string contentLevel = passDesc.get<std::string>("content");
//    if (contentLevel != "full" && outputScope != "name")
//        throw ArgumentError(model, "content", contentLevel, "Invalid content scope");

    bool htmlLike = passDesc.get("html");

    std::ofstream ostream;
    std::string outputFile = passDesc.get<std::string>("output");

    mv::utils::validatePath(outputFile);

    ostream.open(outputFile, std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError(model, "output", outputFile, "Unable to open output file");

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    if (outputScope != "DataModel")
    {
        OpModel opModel(model);

        for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
        {
            if (!(outputScope == "ControlModel" || outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel")
                || (opIt->hasTypeTrait("executable") || opIt->getOpType() == "Input" || opIt->getOpType() == "Output"))
            {
                std::string nodeDef = "\t\"" + opIt->getName() + "\" [shape=box,";

                if(opIt->getOpType() == "DMATask")
                {
                    auto direction = opIt->get<mv::DmaDirection>("direction");
                    if(direction == mv::DmaDirectionEnum::DDR2NNCMX ||
                       direction == mv::DmaDirectionEnum::DDR2UPACMX)
                        nodeDef += " style=filled, fillcolor=green,";
                    else if(direction == mv::DmaDirectionEnum::NNCMX2UPACMX ||
                            direction == mv::DmaDirectionEnum::UPACMX2NNCMX)
                        nodeDef += " style=filled, fillcolor=yellow,";
                    else
                        nodeDef += " style=filled, fillcolor=red,";
                }
                if(opIt->getOpType() == "Deallocate")
                {
                    auto location = opIt->get<mv::Tensor::MemoryLocation>("Location");
                    if (location == mv::Tensor::MemoryLocation::NNCMX)
                        nodeDef += " style=filled, fillcolor=orange,";
                    else
                        nodeDef += " style=filled, fillcolor=blue,";
                }
                if(opIt->getOpType() == "BarrierTask")
                    nodeDef += " style=filled, fillcolor=cyan,";

                if (htmlLike)
                {
                    nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\">\
                                <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" \
                                + opIt->getName()
                                + "</B></FONT></TD></TR>";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(opIt->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)\
                        {
                            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                                        + *attrIt
                                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">";

                            auto attrTypeID = opIt->get(*attrIt).getTypeID();
                            bool largeData = mv::attr::AttributeRegistry::hasTypeTrait(attrTypeID, "large");
                            if (verbose && largeData)
                                nodeDef += opIt->get(*attrIt).toLongString();
                            else
                                nodeDef += opIt->get(*attrIt).toString();

                            nodeDef += "</FONT></TD></TR>";
                        }
                    }
                    else
                    {
                        nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + opIt->getOpType() + "</FONT></TD></TR>";
                    }
                    nodeDef += "</TABLE>>";
                }
                else
                {
                    nodeDef += " label=\"" + opIt->getName() + "\\n";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(opIt->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += *attrIt + ": " + opIt->get(*attrIt).toString() + "\\n";
                    }
                    nodeDef += "\"";
                }

                ostream << nodeDef << "];\n";

            }

        }

        if (outputScope == "OpModel" || outputScope == "ExecOpModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {

            DataModel dataModel(model);

            for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
            {
                if (!(outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel")
                    || (opIt->hasTypeTrait("executable") || opIt->getOpType() == "Input" || opIt->getOpType() == "Output"))
                {
                    for (auto dataIt = opIt.leftmostOutput(); dataIt != dataModel.flowEnd(); ++dataIt)
                    {

                        std::string edgeDef = "\t\"" + opIt->getName() + "\" -> \"" + dataIt.sink()->getName() + "\"";
                        if (htmlLike)
                        {
                            edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" \
                                         CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"> \
                                         <FONT POINT-SIZE=\"14.0\"><B>"
                                        + dataIt->getTensor()->getName()
                                        + "</B></FONT></TD></TR>";
                            if (contentLevel == "full")
                            {
                                std::vector<std::string> attrKeys(dataIt->getTensor()->attrsKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    if (*attrIt != "flows")
                                        edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                                                + *attrIt
                                                + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                                + dataIt->getTensor()->get(*attrIt).toString()
                                                + "</FONT></TD></TR>";
                            }
                            else
                            {
                                edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                            + dataIt->getTensor()->getShape().toString()
                                            + "</FONT></TD></TR>";
                            }
                            edgeDef += "</TABLE>>];";
                        }
                        else
                        {
                            edgeDef += " [label=\"" + dataIt->getTensor()->getName() + "\\n";
                            if (contentLevel == "full")
                            {
                                std::vector<std::string> attrKeys(dataIt->getTensor()->attrsKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    if (*attrIt != "flows")
                                        edgeDef += *attrIt + ": " + dataIt->getTensor()->get(*attrIt).toString() + "\\n";
                            }
                            edgeDef += "\"];";
                        }

                        ostream << edgeDef << "\n";

                    }

                }

            }

        }

        if (outputScope == "ControlModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {
            ControlModel controlModel(model);

            for (auto opIt = controlModel.getFirst(); opIt != controlModel.opEnd(); ++opIt)
            {

                for (auto controlIt = opIt.leftmostOutput(); controlIt != controlModel.flowEnd(); ++controlIt)
                {
                    std::string edgeDef = "\t\"" + opIt->getName() + "\" -> \"" + controlIt.sink()->getName() + "\"";
                    if (htmlLike)
                    {
                        if(contentLevel != "full") {
                            if(controlIt->hasAttr("MemoryRequirement")) {

                                edgeDef += " [penwidth=2.0, style=dashed label=<<TABLE BORDER=\"0\" \
                                CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" \
                                COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>"
                                + std::to_string(controlIt->get<int>("MemoryRequirement"))
                                + "</B></FONT></TD></TR>";
                                edgeDef += "</TABLE>>];";
                            }
                            else
                                edgeDef += " [penwidth=2.0, style=dashed]";
                        }
                        else
                        {
                            edgeDef += " [penwidth=2.0, style=dashed label=<<TABLE BORDER=\"0\" \
                                    CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" \
                                    COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>"
                                    + controlIt->getName()
                                    + "</B></FONT></TD></TR>";
                            std::vector<std::string> attrKeys(controlIt->attrsKeys());
                            for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                                            + *attrIt
                                            + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                            + controlIt->get(*attrIt).toString() + "</FONT></TD></TR>";

                            edgeDef += "</TABLE>>];";
                        }
                    }
                    else
                    {
                        edgeDef += " [label=\"" + controlIt->getName() + "\\n";
                        if (contentLevel == "full")
                        {
                            std::vector<std::string> attrKeys(controlIt->attrsKeys());
                            for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                edgeDef += *attrIt + ": " + controlIt->get(*attrIt).toString() + "\\n";
                        }
                        edgeDef += "\"];";
                    }

                    ostream << edgeDef << "\n";

                }

            }
        }

    }
    else
    {

        DataModel dataModel(model);

        for (auto tIt = dataModel.tensorBegin(); tIt != dataModel.tensorEnd(); ++tIt)
        {

            std::string nodeDef = "\t\"" + tIt->getName() + "\" [shape=box,";

            if (htmlLike)
            {
                nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                            <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>"
                            + tIt->getName()
                            + "</B></FONT></TD></TR>";
                if (contentLevel == "full")
                {
                    std::vector<std::string> attrKeys(tIt->attrsKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                                    + *attrIt
                                    + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                    + tIt->get(*attrIt).toString() + "</FONT></TD></TR>";
                }
                else
                {
                    nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                + tIt->getShape().toString()
                                + "</FONT></TD></TR>";
                }
                nodeDef += "</TABLE>>";
            }
            else
            {
                nodeDef += " label=\"" + tIt->getName() + "\\n";
                if (contentLevel == "full")
                {
                    std::vector<std::string> attrKeys(tIt->attrsKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += *attrIt + ": " + tIt->get(*attrIt).toString() + "\\n";
                }
                nodeDef += "\"";
            }

            ostream << nodeDef << "];\n";

        }

        for (auto flowIt = dataModel.flowBegin(); flowIt != dataModel.flowEnd(); ++flowIt)
        {

            if (flowIt.childrenSize() > 0)
            {

                std::string edgeDef = "\t\""
                                    + flowIt->getTensor()->getName()
                                    + "\" -> \""
                                    + flowIt.leftmostChild()->getTensor()->getName()
                                    + "\"";
                if (htmlLike)
                {
                    edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" \
                                CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"> \
                                <FONT POINT-SIZE=\"14.0\"><B>" \
                                + flowIt.sink()->getName()
                                + "</B></FONT></TD></TR>";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(flowIt.sink()->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                                        + *attrIt
                                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                        + flowIt.sink()->get(*attrIt).toString()
                                        + "</FONT></TD></TR>";
                    }
                    else
                    {
                        edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                                    + flowIt.sink()->getOpType()
                                    + "</FONT></TD></TR>";
                    }

                    edgeDef += "</TABLE>>];";
                }
                else
                {
                    edgeDef += " [label=\"" + flowIt.sink()->getName() + "\\n";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(flowIt.sink()->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            edgeDef += *attrIt + ": " + flowIt.sink()->get(*attrIt).toString() + "\\n";
                    }
                    edgeDef += "\"];";
                }

                ostream << edgeDef << "\n";

            }

        }

    }

    ostream << "}\n";
    ostream.close();
}
