__constant float tau = ${tau};

bool is_in_circle(float x, float y, float a, float b, float r) {
    return sqrt(pow(x-a,2)+pow(y-b,2)) < r;
}

__kernel void equilibrilize(__global __write_only float* f_a,
                            __global __write_only float* f_b)
{
    const unsigned int gid = get_global_id(1)*${nX} + get_global_id(0);

    __global __write_only float* preshifted_f_a = f_a + gid;
    __global __write_only float* preshifted_f_b = f_b + gid;

    if (  is_in_circle(get_global_id(0), get_global_id(1), ${nX//4},    ${nY//4},    ${nX//10})
       || is_in_circle(get_global_id(0), get_global_id(1), ${nX//4},    ${nY-nY//4}, ${nX//10})
       || is_in_circle(get_global_id(0), get_global_id(1), ${nX-nX//4}, ${nY//4},    ${nX//10})
       || is_in_circle(get_global_id(0), get_global_id(1), ${nX-nX//4}, ${nY-nY//4}, ${nX//10}) ) {
% for i, w_i in enumerate(w):
        preshifted_f_a[${i*nCells}] = 1./24.f;
        preshifted_f_b[${i*nCells}] = 1./24.f;
% endfor
    } else {
% for i, w_i in enumerate(w):
        preshifted_f_a[${i*nCells}] = ${w_i}.f;
        preshifted_f_b[${i*nCells}] = ${w_i}.f;
% endfor
    }
}

<%
def direction_index(c_i):
    return (c_i[0]+1) + 3*(1-c_i[1])

def neighbor_offset(c_i):
    if c_i[1] == 0:
        return c_i[0]
    else:
        return c_i[1]*nX + c_i[0]
%>

__kernel void collide_and_stream(__global __write_only float* f_a,
                                 __global __read_only  float* f_b,
                                 __global __read_only  int* material)
{
    const unsigned int gid = get_global_id(1)*${nX} + get_global_id(0);

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    __global __write_only float* preshifted_f_a = f_a + gid;
    __global __read_only  float* preshifted_f_b = f_b + gid;

% for i, c_i in enumerate(c):
    const float f_curr_${i} = preshifted_f_b[${direction_index(c_i)*nCells + neighbor_offset(-c_i)}];
% endfor

% for i, expr in enumerate(moments_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    float ${ccode(expr)}
% endfor

    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
    }

% for i, expr in enumerate(collide_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(collide_assignment):
    const float ${ccode(expr)}
% endfor

% for i in range(0,len(c)):
    preshifted_f_a[${i*nCells}] = f_next_${i};
% endfor
}

__kernel void collect_moments(__global __read_only  float* f,
                              __global __write_only float* moments)
{
    const unsigned int gid = get_global_id(1)*${nX} + get_global_id(0);

    __global __read_only float* preshifted_f = f + gid;

% for i in range(0,len(c)):
    const float f_curr_${i} = preshifted_f[${i*nCells}];
% endfor

% for i, expr in enumerate(moments_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    moments[${i*nCells} + gid] = ${ccode(expr.rhs)};
% endfor
}
