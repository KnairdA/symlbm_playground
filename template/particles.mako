__kernel void update_particles(__global __read_only  float4* moments,
                               __global __read_only  int*    material,
                               __global __write_only float4* particles,
                               __global __read_only  float4* init_particles,
                               float aging)
{
  const unsigned int pid = get_global_id(0);

  float4 particle = particles[pid];

% if descriptor.d == 2:
  const unsigned int gid = floor(particle.y)*${memory.size_x} + floor(particle.x);
% elif descriptor.d == 3:
  const unsigned int gid = floor(particle.z)*${memory.size_x*memory.size_y} + floor(particle.y)*${memory.size_x} + floor(particle.x);
% endif

  const float4 moment = moments[gid];

  if (material[gid] == 1 && particle.w < 1.0) {
    particle.x += moment.y;
    particle.y += moment.z;
% if descriptor.d == 3:
    particle.z += moment.w;
% endif
    particle.w += min(particle.x, particle.y) * aging;
  } else {
    particle.xyz = init_particles[pid].xyz;
    particle.w   = particle.w-1.0;
  }

  particles[pid] = particle;
}
