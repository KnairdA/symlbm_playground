#include <array>
#include <cstdint>
#include <memory>
#include <chrono>
#include <iostream>

<%
def pop_offset(i):
    return i * geometry.volume
%>

void equilibrilize(${float_type}*    f_next,
                   ${float_type}*    f_prev,
                   const std::size_t gid)
{
    ${float_type}* preshifted_f_next = f_next + gid;
    ${float_type}* preshifted_f_prev = f_prev + gid;

% for i, w_i in enumerate(descriptor.w):
    preshifted_f_next[${pop_offset(i)}] = ${w_i.evalf()};
    preshifted_f_prev[${pop_offset(i)}] = ${w_i.evalf()};
% endfor
}

<%
def neighbor_offset(c_i):
    return {
        2: lambda:                                          c_i[1]*geometry.size_x + c_i[0],
        3: lambda: c_i[2]*geometry.size_x*geometry.size_y + c_i[1]*geometry.size_x + c_i[0]
    }.get(descriptor.d)()

def padding():
    return {
        2: lambda:                                     1*geometry.size_x + 1,
        3: lambda: 1*geometry.size_x*geometry.size_y + 1*geometry.size_x + 1
    }.get(descriptor.d)()
%>

void collide_and_stream(      ${float_type}* f_next,
                        const ${float_type}* f_prev,
                        const int*           material,
                        const std::size_t    gid)
{
    const int m = material[gid];

          ${float_type}* preshifted_f_next = f_next + gid;
    const ${float_type}* preshifted_f_prev = f_prev + gid;

% for i, c_i in enumerate(descriptor.c):
    const ${float_type} f_curr_${i} = preshifted_f_prev[${pop_offset(i) + neighbor_offset(-c_i)}];
% endfor

% for i, expr in enumerate(moments_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    ${float_type} ${ccode(expr)}
% endfor

% for i, expr in enumerate(collide_subexpr):
    const ${float_type} ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(collide_assignment):
    const ${float_type} ${ccode(expr)}
% endfor

% for i, expr in enumerate(collide_assignment):
    preshifted_f_next[${pop_offset(i)}] = m*f_next_${i} + (1.0-m)*${descriptor.w[i].evalf()};
% endfor
}

int main()
{
    auto f_a = std::make_unique<${float_type}[]>(${geometry.volume*descriptor.q + 2*padding()});
    auto f_b = std::make_unique<${float_type}[]>(${geometry.volume*descriptor.q + 2*padding()});
    auto material = std::make_unique<int[]>(${geometry.volume});

    // buffers are padded by maximum neighbor overreach to prevent invalid memory access
    ${float_type}* f_prev = f_a.get() + ${padding()};
    ${float_type}* f_next = f_b.get() + ${padding()};

    for (int iX = 0; iX < ${geometry.size_x}; ++iX) {
        for (int iY = 0; iY < ${geometry.size_y}; ++iY) {
            for (int iZ = 0; iZ < ${geometry.size_z}; ++iZ) {
                if (iX == 0 || iY == 0 || iZ == 0 || iX == ${geometry.size_x-1} || iY == ${geometry.size_y-1} || iZ == ${geometry.size_z-1}) {
                    material[iZ*${geometry.size_x*geometry.size_y} + iY*${geometry.size_x} + iX] = 0;
                } else {
                    material[iZ*${geometry.size_x*geometry.size_y} + iY*${geometry.size_x} + iX] = 1;
                }
            }
        }
    }

    for (std::size_t iCell = 0; iCell < ${geometry.volume}; ++iCell) {
        equilibrilize(f_prev, f_next, iCell);
    }

    const auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t iStep = 0; iStep < ${steps}; ++iStep) {
        if (iStep % 2 == 0) {
            f_next = f_a.get();
            f_prev = f_b.get();
        } else {
            f_next = f_b.get();
            f_prev = f_a.get();
        }

% if enable_omp_simd:
#pragma omp simd
% endif
        for (std::size_t iCell = 0; iCell < ${geometry.volume}; ++iCell) {
            collide_and_stream(f_next, f_prev, material.get(), iCell);
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "MLUPS: " << ${steps*geometry.volume}/(1e6*duration.count()) << std::endl;

    // calculate average rho as a basic quality check
    double rho_sum = 0.0;

    for (std::size_t i = 0; i < ${geometry.volume*descriptor.q}; ++i) {
        rho_sum += f_next[i];
    }

    std::cout << "avg rho: " << rho_sum/${geometry.volume} << std::endl;

    return 0;
}
