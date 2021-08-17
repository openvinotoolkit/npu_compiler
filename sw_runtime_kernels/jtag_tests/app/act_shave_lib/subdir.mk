include-dirs-shave_nn-y += $(OBJDIR)
$(warning "objdir=$(OBJDIR)")

CURRENT_DIR := $(abspath ./)
VPUIP_2_ABS_DIR := $(abspath ${VPUIP_2_Directory})

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
REL_TO_ROOT := $(subst /, ,${CURRENT_DIR})
REL_TO_ROOT := $(patsubst %,../,${REL_TO_ROOT})
REL_TO_ROOT := $(subst $(SPACE),,$(REL_TO_ROOT))
VPUIP_2_REL_THROUGH_ROOT := $(REL_TO_ROOT)$(VPUIP_2_ABS_DIR)

include-dirs-shave_nn-y += leon/inc

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  shave_nn
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  leon

#presilicon-dir := ../../../../../vpuip_2/presilicon
presilicon-dir := $(VPUIP_2_REL_THROUGH_ROOT)/presilicon

include-dirs-shave_nn-y += $(presilicon-dir)/swCommon/shave_code/include
include-dirs-shave_nn-y += $(presilicon-dir)/drivers/shave/include

include-dirs-lrt-y += $(presilicon-dir)/drivers/leon/drv/include
include-dirs-lrt-y += $(presilicon-dir)/swCommon/leon/include
include-dirs-lrt-y += $(presilicon-dir)/swCommon/shared/include

subdirs-shave_nn-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/swCommon

subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  $(presilicon-dir)/drivers

