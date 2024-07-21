import sys
from sage.all import *
import cProfile
import math


EPS = 10.0**(-9)
PI = float(pi)
C_FIELD = CDF
C_I = C_FIELD(I)
C_PI = C_FIELD(pi)
C_1 = C_FIELD(1)


def main():
    tmp = kronecker_example()


def kronecker_example(verbose=True):
    # ~1.5 hours
    D_min =  95000
    D_max = 105000
    num_eval_pts = 10000
    eval_range = (1.3,1e7)

    c = C_FIELD(3)/5 # 1/2 + 1/logX << c < 3/4
    T_max = 900
    num_int_pts = 180000
    
    D_list = [d for d in range(D_min,D_max) if ZZ(d).is_fundamental_discriminant()]
    eval_pts = get_eval_pts(eval_range, num_eval_pts)

    if verbose:
        print('D_min = %i,  D_max = %i,  num_eval_pts = %i,  eval_range = ' % (D_min, D_max, num_eval_pts) + str(eval_range))
        print('c = '+str(c))
        print('T_max = %i,  num_int_pts = %i' % (T_max, num_int_pts))
        print('len(D_list): '+str(len(D_list)))
    
    # 1/(sqrt(X) #F) \sum_{d \in F} \sum_{p^k < X} chi_d(p^k) log p
    kronecker_avg_vals = build_kronecker_avgs(D_list, eval_range[1], with_prime_powers=True, odd_only=True) # long time
    if verbose:
        print('kronecker_avg_vals: ' + str(sorted(list(kronecker_avg_vals.items()))[:10]) + str(sorted(list(kronecker_avg_vals.items()))[-10:]))
        print('len(kronecker_avg_vals): '+str(len(kronecker_avg_vals)))
    kronecker_partial_sums = eval_kronecker_partial_sums(kronecker_avg_vals, eval_pts=eval_pts, with_prime_powers=True, odd_only=True) # semi-long time
    if verbose:
        print('kronecker_partial_sums: ' + str(sorted(list(kronecker_partial_sums.items()))[:10]) + str(sorted(list(kronecker_partial_sums.items()))[-10:]))
        print('len(kronecker_partial_sums): '+str(len(kronecker_partial_sums)))
    
    # 1/(2pii) int_{c-iT}^{c+iT} pi^2/6 Gamma((1-s)/2)/Gamma(s/2) * zeta(2-2s)/zeta(3-2s) * 1/#F \sum_{d \in F} (pi X/d)^{s - 1/2} ds/s
    int_vals = get_int_vals(eval_pts, c, T_max, num_int_pts, D_list)
    if verbose:
        print('int_vals: ' + str(sorted(list(int_vals.items()))[:10]) + str(sorted(list(int_vals.items()))[-10:]))
        print('len(int_vals): '+str(len(int_vals)))
    int_vals_RR = {k:v.real_part() for k,v in int_vals.items()}

    pts_kronecker_partial_sums = points(kronecker_partial_sums.items(), dpi=400, size=2, color='blue')
    pts_int_vals = points(int_vals_RR.items(), dpi=400, size=2, color='darkgoldenrod')
    show(pts_kronecker_partial_sums + pts_int_vals, scale='semilogx', figsize=[18,8])

    to_ret = [kronecker_avg_vals, kronecker_partial_sums, int_vals]
    return to_ret


def get_kronecker_zeros(D_min, D_max, T_max=200, kronecker_zeros=None):
    if kronecker_zeros is None:
        # pass a dict to update in place
        # this does not validate that the same set is being averaged
        kronecker_zeros = {}
    zeta_zeros = get_zeta_zeros(T_max=T_max)
    for D in range(D_min,D_max):
        if ZZ(D).is_fundamental_discriminant():
            if D not in kronecker_zeros:
                K = NumberField(ZZ['x']([-D,0,1]),'z')
                L = K.zeta_function()
                zeros = L.zeros(T_max)
                zeros = [tmp for tmp in zeros if tmp not in zeta_zeros] # Dedekind zeta function is zeta function * quadratic Dirichlet L-function
                kronecker_zeros[D] = zeros
    return kronecker_zeros


def get_zeta_zeros(T_max=200):
    chi = DirichletGroup(1)[0]
    L = chi.lfunction()
    zeta_zeros = set(L.zeros(200))
    return zeta_zeros


def get_driving_fn(zero_list, normalizing_factor=1):
    # normalizing_factor could be 1/number of elliptic curves for averaging
    def fn(x):
        sumval = 0
        logx = C_FIELD(log(x))
        for gamma in zero_list: # gamma is imaginary part of zero on the critical line
            term = (exp(C_I*gamma*logx) / (0.5 + C_I*gamma)).real_part() * 2
            sumval += term
        sumval *= normalizing_factor
        return sumval
    return fn


def eval_driving_fn(driving_fn, eval_pts=None, eval_range=(2,10000), num_eval_pts=1000, driving_fn_vals=None):
    if driving_fn_vals is None:
        # pass a dict to update in place
        driving_fn_vals = {}
    if eval_pts is None:
        eval_pts = get_eval_pts(eval_range, num_eval_pts)
    for x in eval_pts:
        if x not in driving_fn_vals:
            driving_fn_vals[x] = driving_fn(x)
    return driving_fn_vals


def get_eval_pts(eval_range, num_eval_pts):
    log_min = float(log(eval_range[0]))
    log_max = float(log(eval_range[1]))
    delta = (log_max - log_min)/num_eval_pts
    eval_pts = [exp(log_min + i*delta) for i in range(num_eval_pts)]
    return eval_pts


def build_kronecker_avgs(D_list, x_max, with_prime_powers=False, odd_only=False, kronecker_avg_vals=None):
    if kronecker_avg_vals is None:
        # pass a dict to update in place
        # this does not validate that the same set is being averaged
        kronecker_avg_vals = {}
    if with_prime_powers:
        p_gen = prime_powers(x_max)
        if odd_only:
            p_gen = [p for p in p_gen if not ZZ(p).is_square()]
    else:
        p_gen = primes(x_max)
    for p in p_gen:
        p_sum = 0
        for D in D_list:
            kronecker_val = kronecker(D,p)
            p_sum += kronecker_val
        p_sum *= log(float(ZZ(p).radical())) # von Mangoldt
        avg_val = p_sum / len(D_list)
        kronecker_avg_vals[p] = avg_val
    return kronecker_avg_vals


def eval_kronecker_partial_sums(kronecker_avg_vals, weight_at_x=0.5, eval_pts=None, eval_range=(2,10000), num_eval_pts=1000, with_prime_powers=False, odd_only=False, partial_sum_vals=None):
    # weight_at_x = 0.5 means if x is prime the term is counted half
    # weight_at_x = 0 means sum_{p < x}
    # weight_at_x = 1 means sum_{p <= x}
    if partial_sum_vals is None:
        # pass a dict to update in place
        partial_sum_vals = {}
    if eval_pts is None:
        eval_pts = get_eval_pts(eval_range, num_eval_pts)
    eval_pts_set = set(eval_pts)
    x_max = eval_pts[-1]
    if with_prime_powers:
        if not odd_only:
            interesting_points = eval_pts + list(prime_powers(int(x_max)+1))
        else:
            interesting_points = eval_pts + [p for p in prime_powers(int(x_max)+1) if not ZZ(p).is_square()]
    else:
        interesting_points = eval_pts + list(primes(int(x_max)+1))
    interesting_points = sorted(list(set(interesting_points)))
    current_partial_sum = 0
    new_sum_chunk = 0
    for x in interesting_points:
        is_prime = (x in ZZ) and (ZZ(x).is_prime() or (with_prime_powers and ZZ(x).is_prime_power() and ((not odd_only) or (not ZZ(x).is_square()))))
        is_eval_pt = x in eval_pts_set
        if is_prime and (not is_eval_pt):
            new_sum_chunk += kronecker_avg_vals[x]
        elif is_prime and is_eval_pt:
            new_sum_chunk += weight_at_x * kronecker_avg_vals[x]
            current_partial_sum += new_sum_chunk
            coeff = 1.0 / sqrt(float(x))
            partial_sum_vals[x] = coeff * current_partial_sum
            new_sum_chunk = (1 - weight_at_x) * kronecker_avg_vals[x]
        elif (not is_prime) and is_eval_pt:
            current_partial_sum += new_sum_chunk
            coeff = 1.0 / sqrt(float(x))
            partial_sum_vals[x] = coeff * current_partial_sum
            new_sum_chunk = 0
        else:
            # ??
            raise ValueError('x = '+str(x))
    return partial_sum_vals


def get_int_vals(X_vals, c, T_max, num_pts, D_list, int_vals=None):
    # 1/(2pii) int_{c-iT}^{c+iT} pi^2/6 Gamma((1-s)/2)/Gamma(s/2) * zeta(2-2s)/zeta(3-2s) * 1/#F \sum_{d \in F} (pi X/d)^{s - 1/2} ds/s
    if int_vals is None:
        int_vals = {}
    
    s_vals = []
    gamma_vals = {}
    zeta_vals = {}
    dpow_vals = {}
    delta_t = C_FIELD(2)*T_max / num_pts
    s = c - C_I*T_max
    for _ in range(num_pts + 1):
        gamma_vals[s] = gamma((1-s)/2) / gamma(s/2)
        zeta_vals[s] = zeta(2-2*s) / zeta(3-2*s)
        dpow_val = 0
        for d in D_list:
            dpow_val += (C_PI/d)**(s - C_1/2)
        dpow_val /= s * len(D_list)
        dpow_vals[s] = dpow_val
        s_vals.append(s)
        s += C_I*delta_t
    
    for X in X_vals:
        C_X = C_FIELD(X)
        int_val = 0
        for s in s_vals:
            term = 1
            term *= gamma_vals[s]
            term *= zeta_vals[s]
            term *= dpow_vals[s]
            term *= C_X**(s - C_1/2)
            term /= 2*C_PI*C_I
            term *= C_I*delta_t
            int_val += term
        int_val *= C_PI**2 / 6
        int_vals[X] = int_val
    
    return int_vals



#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
