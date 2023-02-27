
DrawLab supports two file formats(XML and JSON) to represent scene. The design is borrowed from **Mitsuba**.

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

This describes a scene with different objects