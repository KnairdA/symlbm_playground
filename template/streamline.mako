<%
def gid():
    return {
        2: 'get_global_id(1)*%d + get_global_id(0)' % memory.size_x,
        3: 'get_global_id(2)*%d + get_global_id(1)*%d + get_global_id(0)' % (memory.size_x*memory.size_y, memory.size_x)
    }.get(descriptor.d)
%>

__kernel void dillute(__global int* material,
                      __read_write image2d_t streamlines)
{
    const unsigned int gid = ${gid()};
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 color = read_imagef(streamlines, pos);

    if (material[gid] == 1) {
        color.xyz *= 0.975;
    } else {
        color.xyz = 0.2;
    }

    write_imagef(streamlines, pos, color);
}

float3 blueRedPalette(float x) {
    return mix(
        (float3)(0.0, 0.0, 1.0),
        (float3)(1.0, 0.0, 0.0),
        x
    );
}

__kernel void draw_streamline(__global float4* moments,
                              __global int*    material,
                              __global float2* origins,
                              __read_write image2d_t streamlines)
{
    float2 particle = origins[get_global_id(0)];

    for (int i = 0; i < ${2*memory.size_x}; ++i) {
    const unsigned int gid = round(particle.y)*${memory.size_x} + round(particle.x);
        const float4 moment = moments[gid];

        if (material[gid] != 1) {
            break;
        }

        particle.x += 0.5 * moment.y / 0.01;
        particle.y += 0.5 * moment.z / 0.01;

        const int2 pos = (int2)(round(particle.x), round(particle.y));

        float4 color = read_imagef(streamlines, pos);
        color.xyz += 0.05;

        write_imagef(streamlines, pos, color);
    }
}
