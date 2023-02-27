
DrawLab supports two file formats(XML and JSON) to represent scene. The design is borrowed from [**Mitsuba**](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html).

A basic scene description file is as follows:

```xml
<scene>
	<bsdf type="" id=""/>
    <texture type="" id=""/>
	<integrator type=""/>
	<sampler type=""/>
	<camera type=""/>
	<mesh type=""/>
	<emitter type=""/>
</scene>
```

A valid scene description file should has an integrator, a sampler, a camera, some meshes and some emitters.

## Properties

Some simple properties:

```xml
<integer name="int_property" value="1234"/>
<float name="float_property" value="-1.5e3"/>
<color name="pd" value="0., 0., 0."/>
<string name="filename" value="uffizi.hdr"/>
<vector name="normal" value="-1, 0, 0"/>
<boolean name="bool_property" value="true"/>
<point name="position" value="0, 0, 0"/>
```

### Transform

Transform property requires more than a single tag, from identity matrix to build a transform matrix base on the commands. For examples:

```xml
<transform name="toWorld">
	<scale value="0.06,0.06,-1"/>
	<translate value="10,0,25"/>
</transform>
```

The supported transformation:

- Translations:

  ```xml
  <translate value="-1, 3, 4"/>
  ```

- Scale.  The coefficients may also be negative to obtain a flip:

  ```xml
  <scale value="5"/>        <!-- uniform scale -->
  <scale value="2, 1, -1"/> <!-- non-uniform scale -->
  ```

- Rotate along an axis:

  ```xml
  <rotate axis="0.701, 0.701, 0" angle="180"/>
  ```

- Directly provide a transform matrix:

  ```xml
  <matrix value="-9.98730179e-01, -3.67203129e-02,  3.44912613e-02, -1.84824074e+01,
  	1.52010147e-03,  6.62361053e-01,  7.49183238e-01, -6.65465801e+01,
  	-5.03559125e-02,  7.48284340e-01, -6.61464155e-01, 2.47364011e+02,
  	0.0, 0.0, 0.0, 1.0"/>
  ```

## Integrators

### Path Tracer(`path`)

A basic path tracer with multiple importance sampling.

```xml
<integrator type="path"/>
```

### Normal(`normals`)

Rendering the normal of the hitpoint.

```xml
<integrator type="normals"/>
```

## Samplers

### Independent sampler(`independent`)

| Parameter     | Type    | Description                              |
| ------------- | ------- | ---------------------------------------- |
| `sampleCount` | integer | Number of samples per pixel (Default: 1) |

```xml
<sampler type="independent">
	<integer name="sampleCount" value="1"/>
</sampler>
```

## Cameras

### Perspective camera(`virtual`)

A simple idealized perspective camera model.

| Parameter | Type    | Description                                                 |
| --------- | ------- | ----------------------------------------------------------- |
| `eye`     | vector  | Camera position                                             |
| `lookat`  | vector  | The point that camera will look at.                         |
| `up`      | vector  | The up direction in the final rendered image.               |
| `fov`     | float   | Camera’s field of view  in degrees along the vertical axis. |
| `width`   | integer | The image width of the final image.                         |
| `height`  | integer | The image height of the final image.                        |

```xml
<camera type="virtual">
	<vector name="eye" value="0, 0.919769, 5.41159"/>
	<vector name="lookat" value="0, 0.893051, 4.41198"/>
	<vector name="up" value="0, 1, 0"/>
	<float name="fov" value="27.7856"/>
	<integer name="width" value="600"/>
	<integer name="height" value="600"/>
</camera>
```

### OpenCV camera(`opencv`)

An opencv camera model with extrinsic and intrinsic.

| Parameter           | Type      | Description                          |
| ------------------- | --------- | ------------------------------------ |
| `cx`,`cy`,`fx`,`fy` | float     | intrinsic parameters                 |
| `extrinsic`         | transform | extrinsic parameters                 |
| `width`             | float     | The image width of the final image.  |
| `height`            | float     | The image height of the final image. |

```xml
<camera type="opencv">
	<float name="cx" value="2707.4819"/>
	<float name="cy" value="3039.0408"/>
	<float name="fx" value="8573.8516"/>
	<float name="fy" value="8597.4912"/>
	<integer name="width" value="8163"/>
	<integer name="height" value="5383"/>
	<transform name="extrinsic">
		<matrix value="-9.98730179e-01, -3.67203129e-02,  3.44912613e-02, -1.84824074e+01, 1.52010147e-03,  6.62361053e-01,  7.49183238e-01, -6.65465801e+01, -5.03559125e-02,  7.48284340e-01, -6.61464155e-01, 2.47364011e+02, 0.0, 0.0, 0.0, 1.0"/>
	</transform>
</camera>
```

## Meshes

Each mesh can bind an transform, or a material.

Optional parameters:

| Parameter | Type      | Description                        |
| --------- | --------- | ---------------------------------- |
| `toWorld` | transform | Transform the mesh to world space. |
|           | bsdf      | The material(bsdf) bound to mesh.  |

### Wavefront OBJ mesh(`obj`)

Load an triangle mesh with specific material and do transformation optional.

| Parameter  | Type   | Description                                    |
| ---------- | ------ | ---------------------------------------------- |
| `filename` | string | Filename of the OBJ file that should be loaded |

Example:

```xml
<mesh type="obj">
	<string name="filename" value="ext_ball.obj"/>
	<bsdf type="diffuse">
		<color name="albedo" value="0.5, 0.5, 0.5"/>
	</bsdf>
	<transform name="toWorld">
		<scale value="0.06,0.06,-1"/>
		<translate value="10,0,25"/>
	</transform>
</mesh>
```

### Rectangle(`rectangle`)

A simple rectangular shape primitive.

| Parameter | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `pos`     | vector | The corner point of the rectangle     |
| `v0`      | vector | An edge vector of the rectangle.      |
| `v1`      | vector | Another edge vector of the rectangle. |
| `normal`  | vector | The normal of the rectangle.          |

```xml
<mesh type="rectangle">
	<vector name="pos" value="-1, -100, 99"/>
	<vector name="v0" value="2, 0, 0"/>
	<vector name="v1" value="0, 0, 2"/>
	<vector name="normal" value="0, 1, 0"/>
</mesh>
```

## Emitters

### Area Light(`area`)

Turn a geometric object into a area light source.

| Parameter | Type  | Description                                                  |
| --------- | ----- | ------------------------------------------------------------ |
| radiance  | color | Specifies the emitted radiance in units of power per unit area per unit steradian. |

```xml
<shape type="obj">
    <emitter type="area">
        <color name="radiance" value="1.0 1.0 1.0"/>
    </emitter>
</shape>
```

### Point Light (`point`)

A simple point light source, which uniformly radiates illumination into all directions.

| Parameter | Type  | Description                                                  |
| --------- | ----- | ------------------------------------------------------------ |
| intensity | color | Specifies the emitted radiance in units of power per unit area per unit steradian. |
| position  | point | light source position                                        |

```xml
<emitter type="area">
	<color name="intensity" value="3,3,2.5"/>
    <point name="position" value="0, 0, 0"/>
</emitter>
```

### Environment Light(`envmap`)

An HDRI (high dynamic range imaging) environment map.

| Parameter  | Type   | Description                                                  |
| ---------- | ------ | ------------------------------------------------------------ |
| `filename` | string | Filename of the radiance-valued input image to be loaded; must be in latitude-longitude format. |

```xml
<emitter type="envmap">
	<string name="filename" value="uffizi.hdr"/>
</emitter>
```

## Textures

Texture objects can be attached to certain material to introduce spatial variation.

### Bitmap texture(`bitmap`)

| Parameter  | Type   | Description                         |
| ---------- | ------ | ----------------------------------- |
| `filename` | string | Filename of the bitmap to be loaded |

Examples:

```xml
<texture type="bitmap">
	<string name="filename" value="vrata_kr.JPG"/>
</texture>
```

## BSDFs

### Diffuse(`diffuse`)

An ideally diffuse material, also referred to as *Lambertian*.

<img src="http://124.223.26.211:8080/images/2023/02/27/90bb5a767f76.png" style="zoom:33%;" align="left"/>


| Parameter | Type             | Description                                                 |
| --------- | ---------------- | ----------------------------------------------------------- |
| `albedo`  | color or texture | Specifies the diffuse albedo of the material (Default: 0.5) |

Example:

```xml
<bsdf type="diffuse">
	<color name="albedo" value="0.5,0.5,0.5"/>
</bsdf>
```

or textured albedo:

```xml
<bsdf type="diffuse">
	<texture type="bitmap">
		<string name="filename" value="wood.jpg"/>
	</texture>
</bsdf>
```

### Dielectric(`dielectric`)

The material models the reflection and refraction between two dielectric materials having mismatched indices of refraction (for instance, water ↔ air)

<img src="http://124.223.26.211:8080/images/2023/02/27/d0d2b96bf7f6.png" style="zoom:33%;" align="left"/>

| Parameter | Type  | Description                                                  |
| --------- | ----- | ------------------------------------------------------------ |
| `intIOR`  | float | Interior index of refraction specified numerically(Default: 1.5046) |
| `extIOR`  | float | Exterior index of refraction specified numerically("exterior" refers to the side that contains the surface normal)(Default: 1.000277) |

```xml
<bsdf type="dielectric"/>
```

### Mirror(`mirror`)

An ideal mirror.

<img src="http://124.223.26.211:8080/images/2023/02/27/211681eb196c.png" style="zoom:33%;" align="left"/>

```xml
<bsdf type="mirror"/>
```

### Microfacet(`microfacet`)

Diffuse brdf + rough conductor brdf.

The microfacet normal distribution is **beckmann** distribution.

<img src="http://124.223.26.211:8080/images/2023/02/27/ebba7b1c17a5.png" style="zoom: 50%;" align="left"/>

| Parameter | Type  | Description                                                  |
| --------- | ----- | ------------------------------------------------------------ |
| `intIOR`  | float | Interior index of refraction specified numerically(Default: 1.5046) |
| `extIOR`  | float | Exterior index of refraction specified numerically("exterior" refers to the side that contains the surface normal)(Default: 1.000277) |
| `alpha`   | float | Surface roughness                                            |
| `kd`      | color | Albedo of the diffuse base material                          |

Example:

```xml
<bsdf type="microfacet">
	<float name="intIOR" value="1.7"/>
	<color name="kd" value="0.2 0.2 0.4"/>
	<float name="alpha" value="0.28"/>
</bsdf>
```

### Anisotropic GGX(`aniso_ggx`)

Diffuse brdf + anisotropic ggx brdf

<img src="http://124.223.26.211:8080/images/2023/02/27/ebba7b1c17a5.png" style="zoom: 50%;" align="left"/>

The microfacet normal distribution is GGX distribution.

| Parameter | Type             | Description       |
| --------- | ---------------- | ----------------- |
| `pd`      | color or texture | diffuse albedo    |
| `ps`      | color or texture | specular albedo   |
| `axay`    | color or texture | surface roughness |

Optional parameters:

| Parameter          | Type    | Description                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `normal`           | texture | normal texture                                               |
| `tangent`          | texture | tangent texture                                              |
| `is_tangent_space` | boolean | Whether the normal/tangent texture is in tangent space or not(default true). |

Examples:

```xml
<bsdf type="aniso_ggx">
	<boolean name="is_tangent_space" value="true"/>
	<color name="pd" value="0., 0., 0."/>
	<color name="ps" value="0.5, 0.5, 0.5"/>
	<color name="axay" value="0.2, 0.2, 1."/>
</bsdf>
```

```xml
<bsdf type="aniso_ggx">
	<boolean name="is_tangent_space" value="false"/>
	<texture type="bitmap" name="pd">
		<string name="filename" value="pd_texture.exr"/>
	</texture>
	<texture type="bitmap" name="ps">
		<string name="filename" value="ps_texture.exr"/>
	</texture>
	<texture type="bitmap" name="axay">
		<string name="filename" value="ax_ay_texture.exr"/>
	</texture>
	<texture type="bitmap" name="normal">
		<string name="filename" value="normal_texture.exr"/>
	</texture>
	<texture type="bitmap" name="tangent">
		<string name="filename" value="tangent_texture.exr"/>
	</texture>
</bsdf>
```

## Declare and References

If one material or texture will be used many times, you can declare the material/texture forward, and reference it in many places.

For examples:

```xml
<scene>
	<bsdf type="diffuse" id="test_mat">
		<color name="albedo" value="0.5, 0.5, 0.5"/>
	</bsdf>
    <mesh type="obj">
        <ref id="test_mat">
    </mesh>
</scene>
```



