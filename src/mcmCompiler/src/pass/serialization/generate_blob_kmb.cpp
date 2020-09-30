#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

static void buildGraphFileKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
static void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(BuildGraphFileKmb)
        .setFunc(buildGraphFileKmbFcn)
        .setDescription("Builds the graphfile according to the schema");
    }

    namespace pass
    {
        MV_REGISTER_PASS(GenerateBlobKmb)
        .setFunc(generateBlobKmbFcn)
        .setDescription("Dumps the graphfile to disk as an executable blob file for KMB");
    }
}

void buildGraphFileKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{   
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);
    rm.buildGraphFile(model, passDesc);
}

void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{   
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE)
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(td);

    if (passDesc.hasAttr("metaInfoSerializer"))
    {
        std::function<void(void*)> metaInfoFcn = passDesc.get<std::function<void(void*)>>("metaInfoSerializer");
        rm.serializeHelper(metaInfoFcn);
    }

    if (passDesc.hasAttr("output")) // if attribute missing, blob file not written
    {
        auto output = passDesc.get<std::string>("output");
        mv::utils::validatePath(output);

        rm.serialize(output);
    }
}
