///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      PlgStreamResult.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgStreamResult Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_INFERENCE_OUTPUT_H__
#define __PLG_INFERENCE_OUTPUT_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class PlgInferenceOutput : public PluginStub
{
  private:
    uint16_t channelID = -1; // TODO[OB] - maybe use the XLink type.. it is uint16_t currently...

  public:
    SReceiver<InferenceMsgPtr> inferenceIn;

    /** Constructor. */
    PlgInferenceOutput(uint32_t device_id = 0) : PluginStub("PlgInfOutput", device_id){}

    /** Destructor. */
    ~PlgInferenceOutput();

    /**
     * Plugin Create method.
     *
     */
    void Create(uint32_t maxSz, uint16_t chanId_unused);

    /**
     * Plugin Delete method.
     *
     * Close the XLink stream.
     */
    virtual void Delete();

    /**
     * Pull a frame from the plugin.
     * This is not a remote method, it simply performs a blocking read on the
     * plugin's XLink stream.
     */
    int PullInferenceID(uint32_t *pAddr, uint32_t *length);
};

#endif // __PLG_INFERENCE_OUTPUT_H__
