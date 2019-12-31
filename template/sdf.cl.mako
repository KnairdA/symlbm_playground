typedef float3 vec3;
typedef float2 vec2;

float3 v3(float x, float y, float z) {
	return (float3)(x,y,z);
}

float2 v2(float x, float y) {
	return (float2)(x,y);
}

__constant float3 center = (float3)(${geometry.size_x/2.5}, ${geometry.size_y/2}, ${geometry.size_z/2});

<%include file="sdf.lib.glsl.mako"/>

${sdf_src}

__kernel void setup_channel_with_sdf_obstacle(__global int* material) {
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    const unsigned z = get_global_id(2);

    const unsigned gid = z*${geometry.size_x*geometry.size_y} + y*${geometry.size_x} + x;

    if (x == 0 || x == ${geometry.size_x-1} ||
        y == 0 || y == ${geometry.size_y-1} ||
        z == 0 || z == ${geometry.size_z-1}) {
        material[gid] = 0;
        return;
    }

    if (x == 1) {
        material[gid] = 3;
        return;
    }

    if (x == ${geometry.size_x-2}) {
        material[gid] = 4;
        return;
    }

    if (y == 1 || y == ${geometry.size_y-2} ||
        z == 1 || z == ${geometry.size_z-2}) {
        material[gid] = 2;
        return;
    }

    if (sdf((float3)(x,y,z)) < 0.0) {
        material[gid] = 2;
        return;
    }

    material[gid] = 1;
}
