# Drawlab

Drawlab is a physically based path tracer that runs on a NVIDIA graphics card with OptiX 7.

## Features

- Unidirectional path tracer
    - Support two backends: CPU, OptiX
    - Two rendering modes: online GUI interaction, offline rendering
    - XML scene description file like mitsuba
    - Multi-importance sampling
- Material
    - Diffuse
    - Mirror
    - Dielectric
    - Microfacet
    - Anisotropic GGX
- Camera
    - Perspective camera
    - Pinhole camera(opencv camera)
- Light
    - Area light(bind to a triangle mesh)
    - Point light
    - Environment light(HDR/EXR image)
- Geometry
    - Mesh(.obj)
    - Rectangle


## Gallery

<img src="http://124.223.26.211:8080/images/2023/03/03/5315f8f873ee.png" width=1200>

<img src="http://124.223.26.211:8080/images/2023/03/03/7f0c67bbbdb3.png" width=1200>

<img src="http://124.223.26.211:8080/images/2023/03/03/ca8658ffb3ca.png" width=1200>

<img src="http://124.223.26.211:8080/images/2023/03/03/f8ca3e151df3.png" width=1200>


## Install

Please refer to the [INSTALL](INSTALL.md) for the build instructions for Windows and Linux.

## Documentation

For documentation, please refer to [Documentation](DOCUMENTATION.md)
  

