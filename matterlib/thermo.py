from matterlib.symbolic import sympy_phys

T, P, v, beta, c_P, s, q, w, u, h, g, f, kappa = sympy_phys.symbols(
    "T P v beta c_P s q w u h g f kappa"
)
# (dv/dT)_P = beta * v
# (dv/dP)_T = -kappa * v
const_var_map = {}
const_var_map[P] = {}
const_var_map[P][T] = 1
const_var_map[P][v] = beta * v
const_var_map[P][s] = c_P / T
const_var_map[P][q] = c_P
const_var_map[P][w] = P * beta * v
const_var_map[P][u] = c_P - P * beta * v
const_var_map[P][h] = c_P
const_var_map[P][g] = -s
const_var_map[P][f] = -s - P * beta * v
const_var_map[T] = {}
const_var_map[T][P] = -1
const_var_map[T][v] = v * kappa
const_var_map[T][s] = beta * v
const_var_map[T][q] = T * beta * v
const_var_map[T][w] = P * kappa * v
const_var_map[T][u] = T * beta * v - P * kappa * v
const_var_map[T][h] = -v + T * beta * v
const_var_map[T][g] = -v
const_var_map[T][f] = -P * kappa * v
const_var_map[h] = {}
const_var_map[h][P] = -c_P
const_var_map[h][T] = v - T * beta * v
const_var_map[h][v] = c_P * kappa * v - T * (beta * v) ** 2 + v**2 * beta
const_var_map[h][s] = v * c_P / T
const_var_map[h][q] = v * c_P
const_var_map[h][w] = -P * (-c_P * kappa * v + T * (beta * v) ** 2 - v**2 * beta)

const_var_map[s] = {}
const_var_map[s][P] = -c_P / T
const_var_map[s][T] = -beta * v
const_var_map[s][v] = -(1 / T) * (-c_P * kappa * v + T * (beta * v) ** 2)
const_var_map[s][q] = 0
const_var_map[s][w] = -(P / T) * (-c_P * kappa * v + T * (beta * v) ** 2)
const_var_map[s][u] = (P / T) * (-c_P * kappa * v + T * (beta * v) ** 2)
const_var_map[s][h] = -v * c_P / T
const_var_map[s][g] = -(1 / T) * (v * c_P - s * T * beta * v)
const_var_map[s][f] = (1 / T) * (
    -P * c_P * kappa * v + P * T * (beta * v) ** 2 + s * T * beta * v
)
const_var_map[g] = {}
const_var_map[g][P] = s
const_var_map[g][T] = v
const_var_map[g][v] = v**2 * beta - s * kappa * v
const_var_map[g][s] = (1 / T) * (v * c_P - s * T * beta * v)
const_var_map[g][q] = -s * T * beta * v + v * c_P
const_var_map[g][w] = P * (v**2 * beta - s * kappa * v)

const_var_map[v] = {}
const_var_map[v][P] = -beta * v
const_var_map[v][T] = -kappa * v
const_var_map[v][s] = (1 / T) * (-c_P * kappa * v + T * (beta * v) ** 2)
const_var_map[v][q] = -c_P * (kappa * v) + T * (beta * v) ** 2
const_var_map[v][w] = 0
const_var_map[v][u] = -c_P * (kappa * v) + T * (beta * v) ** 2
const_var_map[v][h] = -c_P * (kappa * v) + T * (beta * v) ** 2 - v**2 * beta
const_var_map[v][g] = -(v**2) * beta + s * kappa * v
const_var_map[v][f] = s * kappa * v


def make_standard_partial(dependent, wrt, hold):
    if hold not in const_var_map:
        raise Exception(
            f"No standard derivative for constant {hold}. Options for constant are {list(const_var_map.keys())}"
        )
    if dependent not in const_var_map[hold]:
        raise Exception(f"No standard derivative for {dependent} at constant {hold}")
    if wrt not in const_var_map[hold]:
        raise Exception(f"No standard derivative for {wrt} at constant {hold}")
    lhs = sympy_phys.partial(dependent, wrt, hold)
    rhs = const_var_map[hold][dependent] / const_var_map[hold][wrt]
    return sympy_phys.Eq(lhs, rhs)
