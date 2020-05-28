#include "include/mcm/compiler/compilation_unit.hpp"

#ifdef _MSC_VER
// Force linking relevant global objects (registry entries),
// by putting linker directives inside the compilation_unit object file
#include "msvc_registry_init_syms.h"
#endif

const std::string mv::CompilationUnit::ma2490DefTargetDescPath_ = "/config/target/release_kmb.json";
const std::string mv::CompilationUnit::ma3100DefTargetDescPath_ = "/config/target/release_thb.json";
const std::string mv::CompilationUnit::compositionalModelRecordingsPath_ = "/recordings/";
const std::string mv::CompilationUnit::ma2490DefCompDescPath_ = "/config/compilation/release_kmb.json";
const std::string mv::CompilationUnit::ma3100DefCompDescPath_ = "/config/compilation/release_kmb.json";

mv::CompilationUnit::CompilationUnit(const std::string& modelName) :
model_(new OpModel(modelName))
{
    MV_PROFILER_START;
}

mv::CompilationUnit::~CompilationUnit()
{
    MV_PROFILER_FINISH("profiler_output.prof");
    delete model_;
}

bool mv::CompilationUnit::loadTargetDescriptor(const std::string& path)
{
    try
    {
        return targetDescriptor_.load(path);
    }
    catch (ArgumentError& e)
    {
        log(Logger::MessageType::Error, e.what());
        return false;
    }

    return true;

}

bool mv::CompilationUnit::loadCompilationDescriptor(const std::string& filePath)
{
    try
    {
        mv::json::Object jsonDesc = mv::CompilationDescriptor::load(filePath);
        compDescriptor_ = CompilationDescriptor(jsonDesc);
    }
    catch (ParsingError& e)
    {
        return false;
    }

    // query recorded model settings
    std::vector<mv::Element> passList = compDescriptor_.serializePassList();
    mv::Element globalParams = passList[0];
    if (globalParams.hasAttr("recorded_model") )
    {
        bool recordModel = globalParams.get<bool>("recorded_model");
        if (recordModel)
        {
            bool recordWeightsAsText = false;
            if (globalParams.hasAttr("weights_form") )
            {
                std::string weights = globalParams.get<std::string>("weights_form");
                if ( (weights == "text") || (weights == "Text") || (weights == "TEXT") )
                    recordWeightsAsText = true;
            }
            model_->initRecordingFile("templateExampleNew.cpp", recordWeightsAsText);
        }
    }
    return true;
}

bool mv::CompilationUnit::loadCompilationDescriptor(Target target)
{
    std::string descPath;

    switch (target)
    {
        case Target::ma2490:
        {
            descPath = utils::projectRootPath() + ma2490DefCompDescPath_;
            break;
        }
        case Target::ma3100:
        {
            descPath = utils::projectRootPath() + ma3100DefCompDescPath_;
            break;
        }
        default:
            return false;
    }

    return loadCompilationDescriptor(descPath);
}

bool mv::CompilationUnit::loadTargetDescriptor(Target target)
{

    switch (target)
    {

        case Target::ma2490:
        {
            std::string descPath = utils::projectRootPath() + ma2490DefTargetDescPath_;
            return loadTargetDescriptor(descPath);
        }

        case Target::ma3100:
        {
            std::string descPath = utils::projectRootPath() + ma3100DefTargetDescPath_;
            return loadTargetDescriptor(descPath);
        }

        default:
            return false;

    }

    return false;

}

mv::CompilationDescriptor& mv::CompilationUnit::compilationDescriptor()
{
    return compDescriptor_;
}

mv::OpModel& mv::CompilationUnit::model()
{
    return *model_;
}

bool mv::CompilationUnit::initialize()
{

    if (!passManager_.initialize(*model_, targetDescriptor_, compDescriptor_))
        return false;
    // Initialize resouces
    for (auto it = targetDescriptor_.memoryDefs().begin(); it != targetDescriptor_.memoryDefs().end(); ++it)
    {
        mv::DataModel dm(*model_);
        dm.addAllocator(it->first, it->second.size, it->second.alignment);
    }

    return true;

}

mv::Element mv::CompilationUnit::runStep()
{
    return passManager_.step();
}

mv::Element mv::CompilationUnit::run()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE);
    Element output("CompilationOutput");
    output.set<std::string>("ModelName", model_->getName());
    std::vector<mv::Element> passList = compDescriptor_.serializePassList();
    passManager_.loadPassList(passList);

    while (!passManager_.completed())
    {
        MV_PROFILE_VIRTUAL_MEM;
        MV_PROFILE_PHYSICAL_MEM;
        #ifdef MV_PROFILER_ENABLED
        std::size_t ops = model_->opsCount();
        std::size_t dataFlows = model_->dataFlowsCount();
        ControlModel cm(*model_);
        std::size_t controlFlows = cm.controlFlowsCount();
        DataModel dm(*model_);
        std::size_t tensors = dm.tensorsCount();
        std::size_t populatedSize = dm.populatedTotalSize();
        std::size_t unpopulatedSize = dm.unpopulatedTotalSize();
        MV_PROFILED_VARIABLE(ops, MV_PROFILER_COLOR_GREEN);
        MV_PROFILED_VARIABLE(dataFlows, MV_PROFILER_COLOR_ROSE);
        MV_PROFILED_VARIABLE(controlFlows, MV_PROFILER_COLOR_RED);
        MV_PROFILED_VARIABLE(tensors, MV_PROFILER_COLOR_BLUE);
        MV_PROFILED_VARIABLE(populatedSize, MV_PROFILER_COLOR_ORANGE);
        MV_PROFILED_VARIABLE(unpopulatedSize, MV_PROFILER_COLOR_LIME);
        #endif

        output = passManager_.step();
    }

    return output;
}

bool mv::CompilationUnit::completed() const
{
    return passManager_.completed();
}

std::string mv::CompilationUnit::getLogID() const
{
    return "CompilationUnit";
}

std::shared_ptr<std::vector<char>> mv::CompilationUnit::getBlob() const
{
    if(!completed())
        log(Logger::MessageType::Warning, "Getting a blob from compilation unit before completion");
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(targetDescriptor_);
    return rm.getBlob();
}

const mv::BufferMap& mv::CompilationUnit::getBufferMap() const
{
    if(!completed())
        throw RuntimeError(*this, "Attempt of quering the buffer map before the completion of the compilation");
    return model_->bufferMap();
}