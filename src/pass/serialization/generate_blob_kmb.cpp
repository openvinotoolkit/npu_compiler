#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

static void buildGraphFileKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
static void generateBlobKmbFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
static void DMAOrderingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);
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

    namespace pass
    {
        MV_REGISTER_PASS(DMAOrdering)
        .setFunc(DMAOrderingFcn)
        .setDescription("This pass stores an attribute (DMA-level, DPU-schedule-number) on all DMAs so that they can be serialized in the correct order");
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

    if (passDesc.hasAttr("output")) // if attribute missing, blob file not written
    {
        auto output = passDesc.get<std::string>("output");
        mv::utils::validatePath(output);

        rm.serialize(output);
    }
}

mv::Data::OpListIterator findChildDPUorUPATaskOp(mv::ComputationModel& model, mv::Data::OpChildIterator& op)
{
    mv::OpModel om(model); 
    mv::Data::OpListIterator childOp = om.getOp(op.leftmostChild()->getName());
    //std::cout << "childop of task " << op->getName() << " is " << childOp->getName() << std::endl;
    //std::cout << "childop type is " << childOp->getOpType() << std::endl;

    if(childOp->getOpType() == "Output")
            return om.getOutput();
    //std::cout << bool(childOp->getOpType() != "DPUTask") << std::endl;
    //std::cout << bool(childOp->getOpType() != "UPATask") << std::endl;

    if(childOp->getOpType() == "DPUTask" || childOp->getOpType() == "UPATask")
            return om.getOp(childOp->getName());

    while(!(childOp->getOpType() == "DPUTask") && !(childOp->getOpType() == "UPATask"))
    { 
        

        childOp = om.getOp(childOp.leftmostChild()->getName());
        //std::cout << "child op is " << childOp->getName() << std::endl;
        if(childOp->getOpType() == "DPUTask" || childOp->getOpType() == "UPATask")
            return om.getOp(childOp->getName());
        if(childOp->getOpType() == "Output")
            return om.getOutput();
        else 
            childOp = om.getOp(childOp.leftmostChild()->getName());

        //std::cout << "childop now is" << childOp->getName() << std::endl;  
    } 
    //std::cout << "returning " << childOp->getName() << std::endl; 
    return om.getOp(childOp->getName());
}

static void DMAOrderingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{ 
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS) 
    mv::OpModel om(model);
    mv::ControlModel cm(model); 
    mv::Data::OpListIterator dputask; 
    unsigned dpuTaskschedulingNumber = 0;
    unsigned dmaTasklayernumber = 0;
    unsigned maxdpuTaskschedulingNumber = 0;
//     unsigned dpulevel = 0;
//     auto sortedOps = cm.topologicalSort();

//     auto dmas = om.getOps("DMATask");

//     for(auto& dmaOp: dmas) {

//             std::cout << "DMA op is " << dmaOp->getName() << std::endl;
//             auto son = dmaOp.leftmostChild(); 
//                 auto task = om.getOp(son->getName()); 
//                 std::cout << "task op is " << task->getName() << std::endl;
//                 if(task->getOpType() != "DPUTask" && (task->getOpType() != "Output" && task->getOpType() != "UPATask"))
//                     task = findChildDPUorUPATaskOp(model, son); 
       
//                 if(task->hasAttr("schedulingNumber"))
//                 {
//                     dpuTaskschedulingNumber = task->get<unsigned>("schedulingNumber");
//                     dpulevel = task->get<unsigned>("layerNumber");
//                 }
                
//                 auto dmaTasklayernumber = dmaOp->get<unsigned>("layerNumber"); 

//                 if(dpuTaskschedulingNumber > maxdpuTaskschedulingNumber)
//                     maxdpuTaskschedulingNumber = dpuTaskschedulingNumber; 


//                 if(task->getOpType() == "Output")
//                     dpuTaskschedulingNumber= maxdpuTaskschedulingNumber+1;
                
//                 dmaOp->set<unsigned>("DPULevel",dpulevel);
//                 dmaOp->set<unsigned>("DMALevel",dmaTasklayernumber);
//                 dmaOp->set<unsigned>("DPU-schedule-number",dpuTaskschedulingNumber);
//                 dmaOp->set<std::array<unsigned short, 2>>("DMALevel-DPU-schedule-number", {dmaTasklayernumber, dpuTaskschedulingNumber});
            
//     }
// }

unsigned dpulevel = 0;
    auto sortedOps = cm.topologicalSort();
    std::vector<unsigned> dpuTaskschedulingNumbers;

    auto dmas = om.getOps("DMATask");

    for(auto& dmaOp: dmas) {

            dpuTaskschedulingNumbers.clear();
            //std::cout << "DMA name is " << dmaOp->getName() << std::endl;
            for(auto son = dmaOp.leftmostChild(); son != om.opEnd(); ++son)
            {
                auto task = om.getOp(son->getName());
                if(task->getOpType() != "DPUTask" && (task->getOpType() != "Output" && task->getOpType() != "UPATask"))
                    task = findChildDPUorUPATaskOp(model, son);
       
                if(task->hasAttr("schedulingNumber"))
                {
                    dpuTaskschedulingNumber = task->get<unsigned>("schedulingNumber");
                    //std::cout << "task name is " << task->getName() << std::endl;
                    //std::cout << "adding scheduling number to vector " << dpuTaskschedulingNumber << std::endl;
                    dpuTaskschedulingNumbers.push_back(dpuTaskschedulingNumber);
                    dpulevel = task->get<unsigned>("layerNumber");
                }
               
                auto dmaTasklayernumber = dmaOp->get<unsigned>("layerNumber");

                if(dpuTaskschedulingNumber > maxdpuTaskschedulingNumber)
                    maxdpuTaskschedulingNumber = dpuTaskschedulingNumber;


                if(task->getOpType() == "Output")
                    dpuTaskschedulingNumber= maxdpuTaskschedulingNumber+1;
               
                dmaOp->set<unsigned>("DPULevel",dpulevel);
                dmaOp->set<unsigned>("DMALevel",dmaTasklayernumber);
                //std::cout << "DMA name is " << dmaOp->getName() << std::endl;
                for(unsigned i = 0; i < dpuTaskschedulingNumbers.size(); i++)
                    std::cout << dpuTaskschedulingNumbers[i] << std::endl;
                //std::cout << "DMA name is " << dmaOp->getName() << " min scheduing number is " << *std::min_element(std::begin(dpuTaskschedulingNumbers), std::end(dpuTaskschedulingNumbers)) << std::endl;
                dmaOp->set<unsigned>("DPU-schedule-number",*std::min_element(std::begin(dpuTaskschedulingNumbers), std::end(dpuTaskschedulingNumbers)));
                dmaOp->set<std::array<unsigned short, 2>>("DMALevel-DPU-schedule-number", {dmaTasklayernumber, dpuTaskschedulingNumber});
            }
    }
}
