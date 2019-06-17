<%
def gid():
    return {
        2: 'get_global_id(1)*%d + get_global_id(0)' % geometry.size_x,
        3: 'get_global_id(2)*%d + get_global_id(1)*%d + get_global_id(0)' % (geometry.size_x*geometry.size_y, geometry.size_x)
    }.get(descriptor.d)

def pop_offset(i):
    return i * geometry.volume
%>

__kernel void equilibrilize(__global __write_only float* f_next,
                            __global __write_only float* f_prev)
{
    const unsigned int gid = ${gid()};

    __global __write_only float* preshifted_f_next = f_next + gid;
    __global __write_only float* preshifted_f_prev = f_prev + gid;

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
        2: lambda:                                          c_i[1]*geometry.size_x + c_i[0],
        3: lambda: c_i[2]*geometry.size_x*geometry.size_y + c_i[1]*geometry.size_x + c_i[0]
    }.get(descriptor.d)()

%>

__kernel void collide_and_stream(__global __write_only float* f_next,
                                 __global __read_only  float* f_prev,
                                 __global __read_only  int* material)
{
    const unsigned int gid = ${gid()};

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    __global __write_only float* preshifted_f_next = f_next + gid;
    __global __read_only  float* preshifted_f_prev = f_prev + gid;

% for i, c_i in enumerate(descriptor.c):
    const float f_curr_${i} = preshifted_f_prev[${pop_offset(i) + neighbor_offset(-c_i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    float ${ccode(expr)}
% endfor

  ${boundary_src}

% for i, expr in enumerate(collide_subexpr):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(collide_assignment):
    const float ${ccode(expr)}
% endfor

% for i in range(0,descriptor.q):
    preshifted_f_next[${pop_offset(i)}] = f_next_${i};
% endfor
}

__kernel void collect_moments(__global __read_only  float* f,
                              __global __write_only float* moments)
{
    const unsigned int gid = ${gid()};

    __global __read_only float* preshifted_f = f + gid;

% for i in range(0,descriptor.q):
    const float f_curr_${i} = preshifted_f[${pop_offset(i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    moments[${pop_offset(i)} + gid] = ${ccode(expr.rhs)};
% endfor
}
