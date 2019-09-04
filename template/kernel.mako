<%
def gid():
    return {
        2: 'get_global_id(1)*%d + get_global_id(0)' % memory.size_x,
        3: 'get_global_id(2)*%d + get_global_id(1)*%d + get_global_id(0)' % (memory.size_x*memory.size_y, memory.size_x)
    }.get(descriptor.d)

def pop_offset(i):
    return i * memory.volume
%>

__kernel void equilibrilize(__global __write_only ${float_type}* f_next,
                            __global __write_only ${float_type}* f_prev)
{
    const unsigned int gid = ${gid()};

    __global __write_only ${float_type}* preshifted_f_next = f_next + gid;
    __global __write_only ${float_type}* preshifted_f_prev = f_prev + gid;

% if pop_eq_src == '':
%     for i, w_i in enumerate(descriptor.w):
    preshifted_f_next[${pop_offset(i)}] = ${w_i}.f;
    preshifted_f_prev[${pop_offset(i)}] = ${w_i}.f;
%     endfor
% else:
    ${pop_eq_src}
% endif
}

<%
def neighbor_offset(c_i):
    return {
        2: lambda:                                      c_i[1]*memory.size_x + c_i[0],
        3: lambda: c_i[2]*memory.size_x*memory.size_y + c_i[1]*memory.size_x + c_i[0]
    }.get(descriptor.d)()

%>

__kernel void collide_and_stream(__global __write_only ${float_type}* f_next,
                                 __global __read_only  ${float_type}* f_prev,
                                 __global __read_only  int* material,
                                 unsigned int time)
{
    const unsigned int gid = ${gid()};

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    __global __write_only ${float_type}* preshifted_f_next = f_next + gid;
    __global __read_only  ${float_type}* preshifted_f_prev = f_prev + gid;

% for i, c_i in enumerate(descriptor.c):
    const ${float_type} f_curr_${i} = preshifted_f_prev[${pop_offset(i) + neighbor_offset(-c_i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    ${float_type} ${ccode(expr)}
% endfor

  ${boundary_src}

% for i, expr in enumerate(collide_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(collide_assignment):
    const ${float_type} ${ccode(expr)}
% endfor

% for i in range(0,descriptor.q):
    preshifted_f_next[${pop_offset(i)}] = f_next_${i};
% endfor
}

__kernel void collect_moments(__global __read_only  ${float_type}* f,
                              __global __write_only ${float_type}* moments)
{
    const unsigned int gid = ${gid()};

    __global __read_only ${float_type}* preshifted_f = f + gid;

% for i in range(0,descriptor.q):
    const ${float_type} f_curr_${i} = preshifted_f[${pop_offset(i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    moments[${pop_offset(i)} + gid] = ${ccode(expr.rhs)};
% endfor
}

__kernel void collect_gl_moments(__global __read_only  ${float_type}* f,
                                 __global __read_only  int* material,
                                 __global __write_only float4* moments)
{
    const unsigned int gid = ${gid()};

    __global __read_only ${float_type}* preshifted_f = f + gid;

% for i in range(0,descriptor.q):
    const ${float_type} f_curr_${i} = preshifted_f[${pop_offset(i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

    float4 data;

    if (material[gid] == 1) {
% if descriptor.d == 2:
      data.x = ${ccode(moments_assignment[0].rhs)};
      data.y = ${ccode(moments_assignment[1].rhs)};
      data.z = ${ccode(moments_assignment[2].rhs)};
      data.w = sqrt(data.y*data.y + data.z*data.z);
% elif descriptor.d == 3:
      data.x = ${ccode(moments_assignment[0].rhs)};
      data.y = ${ccode(moments_assignment[1].rhs)};
      data.z = ${ccode(moments_assignment[2].rhs)};
      data.w = ${ccode(moments_assignment[3].rhs)};
% endif
    } else {
      data.x = 0.0;
      data.y = 0.0;
      data.z = 0.0;
      data.w = -material[gid];
    }

    moments[gid] = data;
}

__kernel void update_particles(__global __read_only  float4* moments,
                               __global __read_only  int*    material,
                               __global __write_only float4* particles)
{
  const unsigned int pid = get_global_id(0);

  float4 particle = particles[pid];

  const unsigned int gid = floor(particle.y)*${memory.size_x} + floor(particle.x);

  float4 moment = moments[gid];

  if (material[gid] == 1) {
    particle.x += moment.y;
    particle.y += moment.z;
  } else {
    particle.x = particle.z;
    particle.y = particle.w;
  }

  particles[pid] = particle;
}
