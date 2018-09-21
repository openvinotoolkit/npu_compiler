#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bPRelu.hpp"

namespace mv
{
    void bPRelu::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        //printf("Serialization Warning: Manual Override of PReLU Software layer OrderType\n");
        //this->output->setOrder(OrderType::RowMajor);
        //this->input->setOrder(OrderType::RowMajor);
        //this->neg_slope->setOrder(OrderType::RowMajor);

        Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
        Blob_Tensor nSlopeTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->neg_slope);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);
        nSlopeTensor.write(b);

    }

    bPRelu::bPRelu(mv::ComputationOp* it) :
    Blob_Op_Definition(),
    input((it->getInputTensor(0))),
    neg_slope((it->getInputTensor(1))),
    output((it->getOutputTensor(0)))
    {
        
    }

    int bPRelu::getSerializedSize(){
        int fields = 0;
        fields += 3*10 ; // Input, Output
        return fields*4 ;
    }

}
