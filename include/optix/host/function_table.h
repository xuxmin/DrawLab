#pragma once

#include "optix/material/material.h"
#include "optix/integrator/integrator.h"
#include <map>

namespace optix {

static std::map<int, std::string> MaterialCUFiles = {
    {Material::Type::DIFFUSE, "optix/cuda/material/diffuse.cu"},
    {Material::Type::MIRROR, "optix/cuda/material/mirror.cu"},
    {Material::Type::DIELECTRIC, "optix/cuda/material/dielectric.cu"},
    {Material::Type::MICROFACET, "optix/cuda/material/microfacet.cu"},
    {Material::Type::ANISOGGX, "optix/cuda/material/aniso_ggx.cu"}
};

static std::map<int, std::vector<std::string>> MaterialCallableFuncs = {
    {
        Material::Type::DIFFUSE, 
        {"__direct_callable__diffuse_eval", "__direct_callable__diffuse_pdf", "__direct_callable__diffuse_sample"}
    },
    {
        Material::Type::MIRROR, 
        {"__direct_callable__mirror_eval", "__direct_callable__mirror_pdf", "__direct_callable__mirror_sample"}
    },
    {
        Material::Type::DIELECTRIC, 
        {"__direct_callable__dielectric_eval", "__direct_callable__dielectric_pdf", "__direct_callable__dielectric_sample"}
    },
    {
        Material::Type::MICROFACET,
        {"__direct_callable__microfacet_eval", "__direct_callable__microfacet_pdf", "__direct_callable__microfacet_sample"}
    },
    {
        Material::Type::ANISOGGX,
        {"__direct_callable__anisoggx_eval", "__direct_callable__anisoggx_pdf", "__direct_callable__anisoggx_sample"}
    }
};

static std::map<int, std::vector<const char*>> IntegratorTables = {
    {
        Integrator::Type::PATH, 
        {"optix/cuda/integrator/path.cu", "__raygen__path"}
    },
    {
        Integrator::Type::NORMAL,
        {"optix/cuda/integrator/normal.cu", "__raygen__normal"}
    }
};

}