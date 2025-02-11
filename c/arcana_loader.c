#include "vf_shadertoy_arcana.h"

void arcana_register(char * conf_string)
{
    arcana_register_filter((void*)&ff_vsrc_shadertoy);
}