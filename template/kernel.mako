__kernel void equilibrilize(__global __write_only float* f_next,
                            __global __write_only float* f_prev)
{
    const unsigned int gid = get_global_id(2)*(${geometry.size_x*geometry.size_y}) + get_global_id(1)*${geometry.size_x} + get_global_id(0);

    __global __write_only float* preshifted_f_next = f_next + gid;
    __global __write_only float* preshifted_f_prev = f_prev + gid;

% if pop_eq_src == '':
%     for i, w_i in enumerate(descriptor.w):
    preshifted_f_next[${i*geometry.volume}] = ${w_i}.f;
    preshifted_f_prev[${i*geometry.volume}] = ${w_i}.f;
%     endfor
% else:
    ${pop_eq_src}
% endif
}

<%
def neighbor_offset(c_i):
    if descriptor.d == 2:
        if c_i[1] == 0:
            return c_i[0]
        else:
            return c_i[1]*geometry.size_x + c_i[0]
    elif descriptor.d == 3:
        if c_i[1] == 0:
            return c_i[2]*geometry.size_x*geometry.size_y + c_i[0]
        else:
            return c_i[2]*geometry.size_x*geometry.size_y + c_i[1]*geometry.size_x + c_i[0]

%>

__kernel void collide_and_stream(__global __write_only float* f_next,
                                 __global __read_only  float* f_prev,
                                 __global __read_only  int* material)
{
    const unsigned int gid = get_global_id(2)*(${geometry.size_x*geometry.size_y}) + get_global_id(1)*${geometry.size_x} + get_global_id(0);

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    __global __write_only float* preshifted_f_next = f_next + gid;
    __global __read_only  float* preshifted_f_prev = f_prev + gid;

% for i, c_i in enumerate(descriptor.c):
    const float f_curr_${i} = preshifted_f_prev[${i*geometry.volume + neighbor_offset(-c_i)}];
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
    preshifted_f_next[${i*geometry.volume}] = f_next_${i};
% endfor
}

__kernel void collect_moments(__global __read_only  float* f,
                              __global __write_only float* moments)
{
    const unsigned int gid = get_global_id(2)*(${geometry.size_x*geometry.size_y}) + get_global_id(1)*${geometry.size_x} + get_global_id(0);

    __global __read_only float* preshifted_f = f + gid;

% for i in range(0,descriptor.q):
    const float f_curr_${i} = preshifted_f[${i*geometry.volume}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    moments[${i*geometry.volume} + gid] = ${ccode(expr.rhs)};
% endfor
}
