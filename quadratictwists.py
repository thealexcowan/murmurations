
import sys
from sage.all import *
import cProfile
import math
import random
import numpy
import itertools


EPS = 10.0**(-9)
prec = 52 #300
C_FIELD = ComplexField(prec) #CDF
R_FIELD = RealField(prec)
C_I = C_FIELD(I)
C_PI = C_FIELD(pi)
C_1 = C_FIELD(1)
PI = R_FIELD(pi)


# (x, D, T, F, q)
#rv, betav = var('rv, betav')
rv, alphav, betav = var('rv, alphav, betav')
MUPVEC = vector([0,  rv/2 + 2*alphav,  rv/2 + 2*betav,  0,  1/2.0])
MUMVEC = vector([0,  rv/2 - 2*alphav,  rv/2 - 2*betav,  0,  1/2.0])

DEFAULT_XSCALE = 1
DEFAULT_XSCALE_Q = 1
#DEFAULT_XSCALE = 0
#DEFAULT_XSCALE_Q = 0

NUM_SIGMAS = 7


def main():
    fname_suffix = '20250323_1'
    # if len(sys.argv) > 1:
    #     num_instances = int(sys.argv[1])
    #     fname_suffix += '_%i' % num_instances
    # if len(sys.argv) > 2:
    #     instance_index = int(sys.argv[2])
    #     fname_suffix += '_%i' % instance_index
    
    for _ in range(1):
        num_param_pts = 10
        num_atup_pts = 30
        F_exponent = 0.8 + 0.2*random.random()

        fname_prefix = 'param_search_results_ppts_%i_apts_%i_' % (num_param_pts, num_atup_pts)
        fname_prefix += 'Fexpo_%.7f_' % F_exponent
        filename = fname_prefix + fname_suffix

        to_update = search_over_params(num_param_pts, num_atup_pts, r=1, theta=0, fixed_param_vals=None, F_exponent=F_exponent)
        write_param_search_results(to_update, filename, overwrite=False)
    
    return


def run_multiple_instances(num_instances):
    for instance_index in range(num_instances):
        nohupsuffix = '_'+str(num_instances)+'_'+str(instance_index)
        os.system('nohup sage quadratictwists.py ' + str(num_instances) + ' ' + str(instance_index)+' > nohup'+nohupsuffix+'.out &') # apparently it's better to use subprocess.run
    return


def write_param_search_results(param_search_results, filename, overwrite=False):
    with open(filename,'w' if overwrite else 'a') as f:
        for tup in param_search_results:
            tmp = f.write(str(tup) + '\n')
    return


###


def print_param_search_results(param_search_results, num=None, print_stats=False, sort_results=True):
    # Input: Output from e.g. time to_update = kr.search_over_params(num_param_pts, num_atup_pts, r=1, theta=0, eps=0, add_eps_to_cs=False, with_constraints=with_contraints, to_update=to_update)
    # i.e., list of stuff like
    #
    # ((0.028841543719182138, 0.4999363991359422, 0.3257479714590725, 0.5734355495513134, 0.01710139965442526, 0.5000169427636131, 0.8747763331475651, 0.6132149776381597, 0.2086276375346861, 4.293839402741953e-05),
    #  ((0.5560261239423765, 0.9445736933257028, 9.999999999999999e-05),
    #   -0.05541783698676703))
    #
    # Output:
    #
    # sigma_t1ns1: 0.029
    # sigma_t2ns1: 0.500
    # sigma_t1s1: 0.326
    # sigma_t1s2: 0.573
    # sigma_t1s3: 0.017
    # sigma_t2s1: 0.500
    # sigma_t2s2: 0.875
    # sigma_t2s3: 0.613
    # sigma_t2s4: 0.209
    # beta: 0.0000
    # T_scale: 0.5560
    # F_scale: 0.9446
    # K_scale: 0.0001
    # 
    # -0.05542
    
    if sort_results:
        param_search_results = sorted(param_search_results, key = lambda tup: -tup[1][1])
    
    sigma_list_names = ['sigma_t1ns', 'sigma_t2ns', 'sigma_t1s', 'sigma_t2s1', 'sigma_t2s2', 'sigma_t2s3', 'sigma_t2s4']
    if num is None:
        tupgen = param_search_results
    else:
        tupgen = param_search_results[-num:]
    data_by_param = {sigma_name:[] for sigma_name in sigma_list_names}
    data_by_param['beta'] = []
    data_by_param['T_scale'] = []
    data_by_param['F_scale'] = []
    data_by_param['???'] = []
    for tup in tupgen:
        if not print_stats:
            print('')
        for i,tmp1 in enumerate(tup[0][:-1]):
            if i < len(sigma_list_names):
                sigma_name = sigma_list_names[i]
            else:
                sigma_name = '???'
            if not print_stats:
                print(sigma_name + ': %.3f' % tmp1)
            data_by_param[sigma_name].append(tmp1)
        data_by_param['beta'].append(tup[0][-1])
        data_by_param['T_scale'].append(tup[1][0][0])
        if len(tup[1][0]) > 1:
            data_by_param['F_scale'].append(tup[1][0][1])
        if not print_stats:
            print('beta: %.4f' % tup[0][-1])
            print('T_scale: %.4f' % tup[1][0][0])
            if len(tup[1][0]) > 1:
                print('F_scale: %.4f' % tup[1][0][1])
            print('')
            print('%.5f' % tup[1][1])
            print('')

    if print_stats:
        maxklen = max([len(k) for k in data_by_param])
        for k,v in data_by_param.items():
            if v:
                muval = numpy.mean(v)
                stdval = numpy.std(v)
                minv = min(v)
                maxv = max(v)

                mustr = '%.4f' % muval
                if mustr[0] != '-':
                    mustr = ' ' + mustr
                stdstr = ' %.4f' % stdval
                minvstr = '%.3f' % minv
                if minvstr[0] != '-':
                    minvstr = ' ' + minvstr
                maxvstr = '%.3f' % maxv
                if maxvstr[0] != '-':
                    maxvstr = ' ' + maxvstr
                rangestr = '%s  -- %s' % (minvstr, maxvstr)
                print(' '*(maxklen - len(k)) + k + ':     mean: %s     std: %s     range: %s' % (mustr, stdstr, rangestr))
    return



################
### Searches ###
################


def search_over_params(num_param_pts, num_atup_pts, r=1, theta=0, fixed_param_vals=None, F_exponent=None, T_exponent=None, integrand_size=False, with_alpha=False, to_update=None):
    if to_update is None:
        to_update = []

    if not integrand_size:
        param_fn = build_param_fn(num_pts=num_atup_pts, r=r, theta=theta, fixed_param_vals=fixed_param_vals, F_exponent=F_exponent, T_exponent=T_exponent)
        param_constraints = build_param_constraints(theta=theta, fixed_param_vals=fixed_param_vals)
    else:
        param_fn = build_param_fn_integrand_size(num_pts=num_atup_pts, r=r, theta=theta, fixed_param_vals=fixed_param_vals, F_exponent=F_exponent, T_exponent=T_exponent)
        param_constraints = build_param_constraints_integrand_size(theta=theta, fixed_param_vals=fixed_param_vals)
    
    for _ in range(num_param_pts):
        ptup = tuple(get_random_params(fixed_param_vals, integrand_size=integrand_size, with_alpha=integrand_size or with_alpha))
        if param_constraints:
            mintup = minimize_constrained(param_fn, param_constraints, ptup)
        else:
            mintup = minimize(param_fn, ptup)
        to_append = (mintup, param_fn(mintup, with_atup=True))
        to_update.append(to_append)
    
    return to_update


def search_from_params(sigma_list, beta, alpha=0, num_pts=10**3, r=1, theta=0, include_horizontal=True, c=0.5, F_exponent=None, T_exponent=None, integrand_size=False, to_update=None):
    if to_update is None:
        to_update = []
    
    error_term_list = get_error_term_powers(sigma_list, beta, alpha=alpha, r=r, theta=theta, include_horizontal=include_horizontal, c=c, integrand_size=integrand_size)
    fn = build_fn(error_term_list, F_exponent=F_exponent, T_exponent=T_exponent, exclude_t0=integrand_size)
    constraints = build_constraints(F_exponent=F_exponent, T_exponent=T_exponent)
    
    for _ in range(num_pts):
        atup = []
        if T_exponent is None:
            a_T = random.gauss(0.5,1)
            atup.append(a_T)
        if F_exponent is None:
            a_F = random.gauss(0.5,1)
            atup.append(a_F)
        atup = tuple(atup)

        if constraints:
            mintup = minimize_constrained(fn, constraints, atup)
        else:
            mintup = minimize(fn, atup)
        fnval = fn(mintup)
        to_append = (mintup, fnval)
        to_update.append(to_append)
    
    return to_update




######################################
### Error term size conglomeration ###
######################################


def build_param_fn(num_pts=10**2, r=1, theta=0, fixed_param_vals=None, include_horizontal=True, c=0.5, F_exponent=None, T_exponent=None):
    
    def param_fn(ptup, with_atup=False, num_atup_pts=num_pts, r=r, theta=theta, fixed_param_vals=fixed_param_vals, include_horizontal=include_horizontal, c=c, F_exponent=F_exponent, T_exponent=T_exponent, with_alpha=False):
        if fixed_param_vals is None:
            fixed_param_vals = {}
        sigma_t1ns = fixed_param_vals.get('t1ns',None)
        sigma_t2ns = fixed_param_vals.get('t2ns',None)
        sigma_t1s = fixed_param_vals.get('t1s',None)
        sigma_t2s1 = fixed_param_vals.get('t2s1',None)
        sigma_t2s2 = fixed_param_vals.get('t2s2',None)
        sigma_t2s3 = fixed_param_vals.get('t2s3',None)
        sigma_t2s4 = fixed_param_vals.get('t2s4',None)
        beta = fixed_param_vals.get('beta',None)
        alpha = fixed_param_vals.get('alpha',None)

        # =(
        counter = 0
        if sigma_t1ns is None:
            sigma_t1ns = ptup[counter]
            counter += 1
        if sigma_t2ns is None:
            sigma_t2ns = ptup[counter]
            counter += 1
        if sigma_t1s is None:
            sigma_t1s = ptup[counter]
            counter += 1
        if sigma_t2s1 is None:
            sigma_t2s1 = ptup[counter]
            counter += 1
        if sigma_t2s2 is None:
            sigma_t2s2 = ptup[counter]
            counter += 1
        if sigma_t2s3 is None:
            sigma_t2s3 = ptup[counter]
            counter += 1
        if sigma_t2s4 is None:
            sigma_t2s4 = ptup[counter]
            counter += 1
        if beta is None:
            beta = ptup[counter]
            counter += 1
        if (alpha is None) and with_alpha:
            alpha = ptup[counter]
            counter += 1
        
        sigma_list = [sigma_t1ns, sigma_t2ns, sigma_t1s, sigma_t2s1, sigma_t2s2, sigma_t2s3, sigma_t2s4]
        if with_alpha:
            min_atups = search_from_params(sigma_list, beta, alpha=alpha, num_pts=num_atup_pts, r=r, theta=theta, include_horizontal=include_horizontal, c=c, F_exponent=F_exponent, T_exponent=T_exponent)
        else:
            min_atups = search_from_params(sigma_list, beta, num_pts=num_atup_pts, r=r, theta=theta, include_horizontal=include_horizontal, c=c, F_exponent=F_exponent, T_exponent=T_exponent)
        min_tup = sorted(min_atups, key = lambda tup: tup[1])[0]
        if with_atup:
            to_ret = min_tup
        else:
            to_ret = min_tup[1]
        return to_ret
    
    return param_fn


def build_param_fn_integrand_size(num_pts=10**2, r=1, theta=0, fixed_param_vals=None, F_exponent=None, T_exponent=None):
    
    def param_fn(ptup, with_atup=False, num_atup_pts=num_pts, r=r, theta=theta, fixed_param_vals=fixed_param_vals, F_exponent=F_exponent, T_exponent=T_exponent):
        if fixed_param_vals is None:
            fixed_param_vals = {}
        sigma = fixed_param_vals.get('sigma',None)
        beta = fixed_param_vals.get('beta',None)
        alpha = fixed_param_vals.get('alpha',None)
        
        counter = 0
        if sigma is None:
            sigma = ptup[counter]
            counter += 1
        if beta is None:
            beta = ptup[counter]
            counter += 1
        if alpha is None:
            alpha = ptup[counter]
            counter += 1
        
        sigma_list = [sigma] * NUM_SIGMAS
        min_atups = search_from_params(sigma_list, beta, alpha=alpha, num_pts=num_atup_pts, r=r, theta=theta, include_horizontal=False, c=sigma, F_exponent=F_exponent, T_exponent=T_exponent, integrand_size=True)
        min_tup = sorted(min_atups, key = lambda tup: tup[1])[0]
        if with_atup:
            to_ret = min_tup
        else:
            to_ret = min_tup[1]
        return to_ret
    
    return param_fn


def build_fn(error_term_list, F_exponent=None, T_exponent=None, exclude_tT=False, exclude_t0=False):
    def fn(atup, print_individual=False, q=False, rat=False, exclude_tT=exclude_tT, exclude_t0=exclude_t0):
        if q:
            a_T = 0
            a_F = 0
        else:
            if len(atup) == 0:
                a_T = T_exponent
                a_F = F_exponent
            elif len(atup) == 2:
                a_T, a_F = atup
            elif len(atup) == 1:
                if F_exponent is not None:
                    a_T = atup[0]
                    a_F = F_exponent
                elif T_exponent is not None:
                    a_T = T_exponent
                    a_F = atup[0]
                else:
                    raise NotImplementedError(str(atup))
            else:
                raise NotImplementedError(str(atup))

        terms = []
        for et_counter, et in enumerate(error_term_list):
            
            if (not et.horizontal) and (not exclude_t0):
                # contribution from t ~= 0
                if q:
                    t0 = et.term_size_q({'T':0, 'F':a_F})
                else:
                    t0 = et.term_size({'T':0, 'F':a_F})
                if rat:
                    t0 = identify_rational(t0, best_only=True)
                terms.append(t0)
                if print_individual:
                    print(et_counter, '\t', et.name, '\t', t0)

            # contribution from t ~= T
            if q:
                t1 = et.term_size_q({'T':a_T, 'F':a_F})
            else:
                t1 = et.term_size({'T':a_T, 'F':a_F})
            if q or (not exclude_tT):
                if rat:
                    t1 = identify_rational(t1, best_only=True)
                terms.append(t1)
                if print_individual:
                    print(et_counter, '\t', et.name, '\t', t1)
        
        to_ret = max(terms)
        return to_ret
    return fn


def get_error_term_powers(sigma_list, beta, alpha=0, r=1, theta=0, include_horizontal=True, c=0.5, integrand_size=False):
    sigma_t1ns, sigma_t2ns, sigma_t1s, sigma_t2s1, sigma_t2s2, sigma_t2s3, sigma_t2s4 = sigma_list
    powers = []
    powers += t1ns_1(sigma_t1ns, r, theta, beta, alpha=alpha)
    powers += t2ns_1(sigma_t2ns, r, theta, beta, alpha=alpha)
    if not integrand_size:
        powers += t1s_full(sigma_t1s, r, theta, beta, alpha=alpha)
        powers += t2s_1(sigma_t2s1, r, theta, beta, alpha=alpha)
        powers += t2s_2(sigma_t2s2, r, theta, beta, alpha=alpha)
        powers += t2s_3(sigma_t2s3, r, theta, beta, alpha=alpha)
        powers += t2s_4(sigma_t2s4, r, theta, beta, alpha=alpha)
    else:
        powers += t1s_integrand_size(sigma_t1s, r, theta, beta, alpha=alpha)
        powers += t2s_full(sigma_t2s1, r, theta, beta, alpha=alpha)
    if include_horizontal:
        powers_horizontal = []
        powers_horizontal += t1ns_1(c, r, theta, beta, alpha=alpha)
        powers_horizontal += t2ns_1(c, r, theta, beta, alpha=alpha)
        powers_horizontal += t1s_full(c, r, theta, beta, alpha=alpha)
        #powers_horizontal += t2s_1(c, r, theta, beta, alpha=alpha)
        #powers_horizontal += t2s_2(sigma_t2s1, r, theta, beta, alpha=alpha)
        #powers_horizontal += t2s_3(sigma_t2s2, r, theta, beta, alpha=alpha)
        #powers_horizontal += t2s_4(sigma_t2s3, r, theta, beta, alpha=alpha)
        powers_horizontal += t2s_full(c, r, theta, beta, alpha=alpha)
        powers_horizontal += t2s_full(sigma_t2s1, r, theta, beta, alpha=alpha)
        powers_horizontal += t2s_full(sigma_t2s2, r, theta, beta, alpha=alpha)
        powers_horizontal += t2s_full(sigma_t2s3, r, theta, beta, alpha=alpha)
        powers_horizontal += t2s_full(sigma_t2s4, r, theta, beta, alpha=alpha)
        for et in powers_horizontal:
            et.Tpow -= 1
        powers_horizontal.append(perron_et(1+theta, r, theta, beta, alpha=alpha))
        for et in powers_horizontal:
            et.horizontal = True
        powers += powers_horizontal
    if integrand_size:
        for et in powers:
            et.xscale = 0
            et.xscale_q = 0
    return powers



###################
### Constraints ###
###################


def build_param_constraints(theta=0, fixed_param_vals=None):
    if fixed_param_vals is None:
        fixed_param_vals = {}
    sigma_t1ns = fixed_param_vals.get('t1ns',None)
    sigma_t2ns = fixed_param_vals.get('t2ns',None)
    sigma_t1s = fixed_param_vals.get('t1s',None)
    sigma_t2s1 = fixed_param_vals.get('t2s1',None)
    sigma_t2s2 = fixed_param_vals.get('t2s2',None)
    sigma_t2s3 = fixed_param_vals.get('t2s3',None)
    sigma_t2s4 = fixed_param_vals.get('t2s4',None)
    beta = fixed_param_vals.get('beta',None)
    #alpha = fixed_param_vals.get('alpha',None)

    clist = []

    # =( !
    counter = 0
    if sigma_t1ns is None:
        clist.append(lambda atup: atup[0]) # sigma > 0
        clist.append(lambda atup: 1-atup[0]) # sigma < 1
        counter += 1
    if sigma_t2ns is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1-atup[0]) # sigma < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1-atup[1]) # sigma < 1
        counter += 1
    if sigma_t1s is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1-atup[0]) # sigma < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1-atup[1]) # sigma < 1
        if counter == 2:
            clist.append(lambda atup: atup[2]) # sigma > 0
            clist.append(lambda atup: 1-atup[2]) # sigma < 1
        counter += 1
    if sigma_t2s1 is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1-atup[0]) # sigma < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1-atup[1]) # sigma < 1
        if counter == 2:
            clist.append(lambda atup: atup[2]) # sigma > 0
            clist.append(lambda atup: 1-atup[2]) # sigma < 1
        if counter == 3:
            clist.append(lambda atup: atup[3]) # sigma > 0
            clist.append(lambda atup: 1-atup[3]) # sigma < 1
        counter += 1
    if sigma_t2s2 is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1-atup[0]) # sigma < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1-atup[1]) # sigma < 1
        if counter == 2:
            clist.append(lambda atup: atup[2]) # sigma > 0
            clist.append(lambda atup: 1-atup[2]) # sigma < 1
        if counter == 3:
            clist.append(lambda atup: atup[3]) # sigma > 0
            clist.append(lambda atup: 1-atup[3]) # sigma < 1
        if counter == 4:
            clist.append(lambda atup: atup[4]) # sigma > 0
            clist.append(lambda atup: 1-atup[4]) # sigma < 1
        counter += 1
    if sigma_t2s3 is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1-atup[0]) # sigma < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1-atup[1]) # sigma < 1
        if counter == 2:
            clist.append(lambda atup: atup[2]) # sigma > 0
            clist.append(lambda atup: 1-atup[2]) # sigma < 1
        if counter == 3:
            clist.append(lambda atup: atup[3]) # sigma > 0
            clist.append(lambda atup: 1-atup[3]) # sigma < 1
        if counter == 4:
            clist.append(lambda atup: atup[4]) # sigma > 0
            clist.append(lambda atup: 1-atup[4]) # sigma < 1
        if counter == 5:
            clist.append(lambda atup: atup[5]) # sigma > 0
            clist.append(lambda atup: 1-atup[5]) # sigma < 1
        counter += 1
    if sigma_t2s4 is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[0]) # t2s4: sigma < 1/2 - theta
        if counter == 1:
            clist.append(lambda atup: atup[1]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[1]) # t2s4: sigma < 1/2 - theta
        if counter == 2:
            clist.append(lambda atup: atup[2]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[2]) # t2s4: sigma < 1/2 - theta
        if counter == 3:
            clist.append(lambda atup: atup[3]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[3]) # t2s4: sigma < 1/2 - theta
        if counter == 4:
            clist.append(lambda atup: atup[4]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[4]) # t2s4: sigma < 1/2 - theta
        if counter == 5:
            clist.append(lambda atup: atup[5]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[5]) # t2s4: sigma < 1/2 - theta
        if counter == 6:
            clist.append(lambda atup: atup[6]) # sigma > 0
            clist.append(lambda atup: 1/2.0 - theta - atup[6]) # t2s4: sigma < 1/2 - theta
        counter += 1
    
    return clist


def build_param_constraints_integrand_size(theta=0, fixed_param_vals=None):
    if fixed_param_vals is None:
        fixed_param_vals = {}
    sigma = fixed_param_vals.get('sigma',None)
    beta = fixed_param_vals.get('beta',None)
    alpha = fixed_param_vals.get('alpha',None)

    clist = []
    counter = 0
    if sigma is None:
        clist.append(lambda atup: atup[0] - 0.5 - theta) # sigma > 1/2 + theta
        clist.append(lambda atup: 1-atup[0]) # sigma < 1
        counter += 1
    if beta is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]+1) # beta > -1
            clist.append(lambda atup: 1-atup[0]) # beta < 1
        if counter == 1:
            clist.append(lambda atup: atup[1]+1) # beta > -1
            clist.append(lambda atup: 1-atup[1]) # beta < 1
        counter += 1
    
    return clist


def build_constraints(F_exponent=None, T_exponent=None):
    # atup = (a_T, a_F)
    clist = []
    counter = 0
    if T_exponent is None:
        clist.append(lambda atup: atup[0]) # T >> 1 as D -> infty
        clist.append(lambda atup: 1 - atup[0]) # T << D
        counter += 1
    if F_exponent is None:
        if counter == 0:
            clist.append(lambda atup: atup[0]) # #F >> 1 as D -> infty
            clist.append(lambda atup: 1 - atup[0]) # #F << D
        if counter == 1:
            clist.append(lambda atup: atup[1]) # #F >> 1 as D -> infty
            clist.append(lambda atup: 1 - atup[1]) # #F << D
    return clist



###############
### Helpers ###
###############


def get_random_params(fixed_param_vals, integrand_size=False, with_alpha=False):
    if fixed_param_vals is None:
        fixed_param_vals = {}
    sigma = fixed_param_vals.get('sigma',None)
    sigma_t1ns = fixed_param_vals.get('t1ns',None)
    sigma_t2ns = fixed_param_vals.get('t2ns',None)
    sigma_t1s = fixed_param_vals.get('t1s',None)
    sigma_t2s1 = fixed_param_vals.get('t2s1',None)
    sigma_t2s2 = fixed_param_vals.get('t2s2',None)
    sigma_t2s3 = fixed_param_vals.get('t2s3',None)
    sigma_t2s4 = fixed_param_vals.get('t2s4',None)
    beta = fixed_param_vals.get('beta',None)
    alpha = fixed_param_vals.get('alpha',None)
    
    param_list = []
    if not integrand_size:
        if sigma_t1ns is None:
            param_list.append(random.random())
        if sigma_t2ns is None:
            param_list.append(random.random())
        if sigma_t1s is None:
            param_list.append(random.random())
        if sigma_t2s1 is None:
            param_list.append(random.random())
        if sigma_t2s2 is None:
            param_list.append(random.random())
        if sigma_t2s3 is None:
            param_list.append(random.random())
        if sigma_t2s4 is None:
            param_list.append(random.random())
    else:
        if sigma is None:
            sigma = random.random()
            param_list.append(sigma)
    
    if beta is None:
        param_list.append(random.gauss(0,1))
    if (alpha is None) and with_alpha:
        param_list.append(random.gauss(0,1))
    
    return param_list


def identify_rational(val, thresh=100, num_convergents=50, nonzero_only=False, best_only=False, verbose=False):
    vals_QQ = {}
    if val in ZZ:
        vals_QQ = {QQ(val):(0, ZZ(val))}
    else:
        trunc_indices = []
        cf = continued_fraction(val)
        cf_list = list(cf)
        cf_list = cf_list[:num_convergents]
        if len(cf_list) < num_convergents:
            cf_list.append(None)
        for i,cfval in enumerate(cf_list[::-1]):
            #if (cfval is not None) and (cfval > thresh):
            if (cfval is None) or (cfval > thresh):
                trunc_indices.append(i)
        if verbose:
            info_message = ''
            info_message += 'val: ' + str(val) + '\n'
            info_message += 'cf: ' + str(cf) + '\n'
            info_message += 'cf_list: ' + str(cf_list) + '\n'
            info_message += 'convergents: ' + str(cf.convergents()) + '\n'
            info_message += 'trunc_indices: ' + str(trunc_indices) + '\n'
            print(info_message)
        for trunc_index in trunc_indices:
            quotient_index = len(cf_list) - trunc_index - 1
            shift = 1
            convergent_index = quotient_index - shift
            if verbose:
                info_message = ''
                info_message += 'trunc_index: ' + str(trunc_index) + '\n'
                info_message += 'quotient_index: ' + str(quotient_index) + '\n'
                info_message += 'convergent_index: ' + str(convergent_index) #+ '\n'
                print(info_message)
            if convergent_index >= 0:
                #quotient = cf[quotient_index]
                quotient = cf_list[quotient_index]
                if quotient is None:
                    quotient = cf_list[quotient_index-1]+1 # this probably makes no sense
                convergent = cf.convergents()[quotient_index - shift]
                if (not nonzero_only) or (convergent != 0):
                    vals_QQ[convergent] = (quotient_index, quotient)
                if verbose:
                    info_message = ''
                    info_message += 'quotient: ' + str(quotient) + '\n'
                    info_message += 'convergent: ' + str(convergent) + '\n'
                    print(info_message)
            #if verbose:
            #    print(info_message)
    if not best_only:
        to_ret = vals_QQ
    else:
        if vals_QQ:
            #to_ret = sorted(list(vals_QQ.items()), key = lambda tup: -tup[1][1])[0][0] # convergent with largest quotient
            to_ret = sorted(list(vals_QQ.items()), key = lambda tup: -tup[1][1] + tup[0].denominator())[0][0] # hacky
        else:
            to_ret = None
    return to_ret



##############################
### Individual error terms ###
##############################


class ErrorTerm:
    
    def __init__(self, dat=None):
        if dat is None:
            dat = {}
        xpow = dat.get('xpow',0)
        Dpow = dat.get('Dpow',0)
        Tpow = dat.get('Tpow',0)
        Fpow = dat.get('Fpow',0)
        qpow = dat.get('qpow',0)
        muppow = dat.get('muppow',0)
        mumpow = dat.get('mumpow',0)
        mup_vec = dat.get('mupvec',MUPVEC)
        mum_vec = dat.get('mumvec',MUMVEC)
        r = dat.get('r',1)
        alpha = dat.get('alpha',0)
        beta = dat.get('beta',0)
        mup_vec = mup_vec.subs(rv=r, alphav=alpha, betav=beta)
        mum_vec = mum_vec.subs(rv=r, alphav=alpha, betav=beta)
        pow_vec = vector([xpow, Dpow, Tpow, Fpow, qpow])
        pow_vec += muppow * mup_vec
        pow_vec += mumpow * mum_vec
        self.xpow = pow_vec[0]
        self.Dpow = pow_vec[1]
        self.Tpow = pow_vec[2]
        self.Fpow = pow_vec[3]
        self.qpow = pow_vec[4]
        self.weird_deriv_thing = dat.get('weird_deriv_thing',False)
        self.horizontal = dat.get('horizontal',False)
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.xscale = dat.get('xscale', DEFAULT_XSCALE)
        self.xscale_q = dat.get('xscale_q', DEFAULT_XSCALE_Q)
        self.name = dat.get('name', '')

    
    def __repr__(self):
        to_ret = ''
        to_ret += self.pow_dict().__repr__() + '\n'
        to_ret += 'self.weird_deriv_thing: ' + str(self.weird_deriv_thing) + '\n'
        to_ret += 'self.horizontal: ' + str(self.horizontal) + '\n'
        to_ret += 'self.r: ' + str(self.r) + '\n'
        to_ret += 'self.alpha: ' + str(self.alpha) + '\n'
        to_ret += 'self.beta: ' + str(self.beta) + '\n'
        to_ret += 'self.xscale: ' + str(self.xscale) + '\n'
        to_ret += 'self.xscale_q: ' + str(self.r) + '\n'
        to_ret += 'self.name: ' + str(self.name) #+ '\n'
        return to_ret
    
    
    def pow_dict(self):
        pdict = {}
        pdict['x'] = self.xpow
        pdict['D'] = self.Dpow
        pdict['T'] = self.Tpow
        pdict['F'] = self.Fpow
        pdict['q'] = self.qpow
        return pdict
    
    
    def pow_list(self):
        plist = []
        plist.append(self.xpow)
        plist.append(self.Dpow)
        plist.append(self.Tpow)
        plist.append(self.Fpow)
        plist.append(self.qpow)
        return pow_list

    
    def pow_vec(self):
        pvec = vector(self.pow_list())
        return pvec


    def term_size(self, exponents):
        x_exponent = exponents.get('x',self.xscale) # This is where to tweak if you want to see how far you can extend murmurations
        D_exponent = exponents.get('D',1)
        T_exponent = exponents.get('T',0)
        F_exponent = exponents.get('F',0)
        size = 0
        size += x_exponent * self.pow_dict()['x']
        size += D_exponent * self.pow_dict()['D']
        size += T_exponent * self.pow_dict()['T']
        size += F_exponent * self.pow_dict()['F']
        if self.weird_deriv_thing:
            # |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| << D/(r|s|DeltaD)
            # Also, |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| = 1 + O(r|s|DeltaD/D)
            size_alt = 0
            size_alt += x_exponent * self.pow_dict()['x']
            size_alt += D_exponent * (self.pow_dict()['D'] - 1)
            size_alt += T_exponent * (self.pow_dict()['T'] + 1)
            size_alt += F_exponent * (self.pow_dict()['F'] + 1)
            size = min(size, size_alt)
        return size


    def term_size_q(self, exponents):
        x_exponent = exponents.get('x',self.xscale_q) # This is where to tweak if you want to see how far you can extend murmurations
        D_exponent = exponents.get('D',0)
        T_exponent = exponents.get('T',0)
        F_exponent = exponents.get('F',0)
        q_exponent = exponents.get('q',1)
        size = 0
        size += x_exponent * self.pow_dict()['x']
        size += D_exponent * self.pow_dict()['D']
        size += T_exponent * self.pow_dict()['T']
        size += F_exponent * self.pow_dict()['F']
        size += q_exponent * self.pow_dict()['q']
        if self.weird_deriv_thing:
            # |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| << D/(r|s|DeltaD)
            # Also, |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| = 1 + O(r|s|DeltaD/D)
            size_alt = 0
            size_alt += x_exponent * self.pow_dict()['x']
            size_alt += D_exponent * (self.pow_dict()['D'] - 1)
            size_alt += T_exponent * (self.pow_dict()['T'] + 1)
            size_alt += F_exponent * (self.pow_dict()['F'] + 1)
            size_alt += q_exponent * self.pow_dict()['q']
            size = min(size, size_alt)
        return size
        
    


### t1ns

def t1ns_1(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^{1/2} F^{-1} mu+^{1/2} (1 + mu-^{1/2 - sigma + theta}) q^{1/2}
    termval = []
    termval.append(t1ns_11(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1ns_12(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1ns_11(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^{1/2} F^{-1} mu+^{1/2} q^{1/2}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0
    Tpow = 0
    Fpow = -1
    qpow = 1/2.0
    muppow = 1/2.0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1ns_11'}
    term = ErrorTerm(dat=dat)
    return term


def t1ns_12(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^{1/2} F^{-1} mu+^{1/2} mu-^{1/2 - sigma + theta} q^{1/2}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0
    Tpow = 0
    Fpow = -1
    qpow = 1/2.0
    muppow = 1 - sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1ns_12'}
    term = ErrorTerm(dat=dat)
    return term


### t2ns


def t2ns_1(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} mu-^{1/2} (1 + mu-^{sigma - 1/2 + theta}) q^{1/2}
    termval = []
    termval.append(t2ns_11(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2ns_12(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2ns_11(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} mu-^{1/2} q^{1/2}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0 - r*(sigma - 1/2.0)
    Tpow = r*(1/2.0 - sigma)
    Fpow = -1
    qpow = 1/2.0 - (sigma - 1/2.0)
    mumpow = 1/2.0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2ns_11'}
    term = ErrorTerm(dat=dat)
    return term


def t2ns_12(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} mu-^{1/2} mu-^{sigma - 1/2 + theta} q^{1/2}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0 - r*(sigma - 1/2.0)
    Tpow = r*(1/2.0 - sigma)
    Fpow = -1
    qpow = 1/2.0 - (sigma - 1/2.0)
    mumpow = sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2ns_12'}
    term = ErrorTerm(dat=dat)
    return term


### t1s


def t1s_1(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (1 + mu+^{1/2 - sigma + theta}) D^{-1} F^{1}
    termval = []
    termval.append(t1s_11(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_12(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1s_2(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (1 + mu+^{1/2 - sigma + theta}) D^{1/2} D^{1/14} F^{-1} q^{1/6}
    termval = []
    termval.append(t1s_21(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_22(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1s_3(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (1 + mu+^{1/2 - sigma + theta})
    termval = []
    termval.append(t1s_31(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_32(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1s_full(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (1 + mu+^{1/2 - sigma + theta})
    termval = []
    #termval.append(t1s_full1(sigma, r, theta, beta, alpha=alpha))
    #termval.append(t1s_full2(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_31(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_32(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1s_integrand_size(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (mu+^{1/2 - sigma + theta} + D^1/2 q^3/2 / F)  not optimized in the q-aspect
    termval = []
    termval.append(t1s_is1(sigma, r, theta, beta, alpha=alpha))
    #termval.append(t1s_is2(sigma, r, theta, beta, alpha=alpha))
    termval.append(t1s_is3(sigma, r, theta, beta, alpha=alpha))
    return termval


def t1s_is1(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} mu+^{1/2 - sigma + theta}
    xpow = sigma - 1/2.0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    muppow = 1/2.0 - sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_is1'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_is2(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} F/D
    xpow = sigma - 1/2.0
    Dpow = -1
    Tpow = 0
    Fpow = 1
    qpow = 0
    muppow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_is2'}
    term = ErrorTerm(dat=dat)
    return term

def t1s_is3(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^1/2 q^3/2 / F
    xpow = sigma - 1/2.0
    Dpow = 1/2.0
    Tpow = 0
    Fpow = -1
    qpow = 3.0/2 # not optimized; I'm taking Stankus D^1/2 here to focus on D-aspect
    muppow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_is3'}
    term = ErrorTerm(dat=dat)
    return term



def t1s_11(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^{-1} F^{1}
    xpow = sigma - 1/2.0
    Dpow = -1
    Tpow = 0
    Fpow = 1
    qpow = 0
    muppow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_11'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_12(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} mu+^{1/2 - sigma + theta} D^{-1} F^{1}
    xpow = sigma - 1/2.0
    Dpow = -1
    Tpow = 0
    Fpow = 1
    qpow = 0
    muppow = 1/2.0 - sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_12'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_21(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} D^{1/2} D^{1/14} F^{-1} q^{1/6}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0 + 1/14.0
    Tpow = 0
    Fpow = -1
    qpow = 1/6.0
    muppow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_21'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_22(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} (1 + mu+^{1/2 - sigma + theta}) D^{1/2} D^{1/14} F^{-1} q^{1/6}
    xpow = sigma - 1/2.0
    Dpow = 1/2.0 + 1/14.0
    Tpow = 0
    Fpow = -1
    qpow = 1/6.0
    muppow = 1/2.0 - sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_22'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_31(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2}
    xpow = sigma - 1/2.0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    muppow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_31'}
    term = ErrorTerm(dat=dat)
    return term


def t1s_32(sigma, r, theta, beta, alpha=0):
    # x^{sigma - 1/2} mu+^{1/2 - sigma + theta}
    xpow = sigma - 1/2.0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    muppow = 1/2.0 - sigma + theta
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t1s_32'}
    term = ErrorTerm(dat=dat)
    return term


# def t1s_full1(sigma, r, theta, beta, alpha=0):
#     # x^{sigma - 1/2}
#     xpow = sigma - 1/2.0
#     Dpow = 0
#     Tpow = 0
#     Fpow = 0
#     qpow = 0
#     muppow = 0
#     dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha}
#     term = ErrorTerm(dat=dat)
#     return term


# def t1s_full2(sigma, r, theta, beta, alpha=0):
#     # x^{sigma - 1/2}  mu+^{1/2 - sigma + theta}
#     xpow = sigma - 1/2.0
#     Dpow = 0
#     Tpow = 0
#     Fpow = 0
#     qpow = 0
#     muppow = 1/2.0 - sigma + theta
#     dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'muppow':muppow, 'beta':beta, 'r':r, 'alpha':alpha}
#     term = ErrorTerm(dat=dat)
#     return term


### t2s


def t2s_1(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{-1} F^{1} (1 + mu-^{sigma - 1/2 + theta})
    termval = []
    termval.append(t2s_11(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_12(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2s_2(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} (D^{1/14} + |s|^{1/6}) (1 + mu-^{sigma - 1/2 + theta}) q^{1/6}
    termval = []
    termval.append(t2s_21(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_22(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_23(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_24(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2s_3(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{4/7} F^{-1} (1 + mu-^{sigma - 1/2 + theta}) q^{1/6} |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| 
    termval = []
    termval.append(t2s_31(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_32(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2s_4(sigma, r, theta, beta, alpha=0, with_assert=False):
    # sigma < 1/2 - theta
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} mu-^{sigma - 1/2 + theta} |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))|
    if with_assert:
        assert(sigma < 1/2.0 - theta + EPS)
    termval = []
    termval.append(t2s_41(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2s_full(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} (1 + mu-^{sigma - 1/2 + theta})
    termval = []
    termval.append(t2s_full1(sigma, r, theta, beta, alpha=alpha))
    termval.append(t2s_full2(sigma, r, theta, beta, alpha=alpha))
    return termval


def t2s_11(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{-1} F^{1}
    xpow = 0
    Dpow = -1
    Tpow = 0
    Fpow = 1
    qpow = 0
    mumpow = 0
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_11'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_12(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{-1} F^{1} mu-^{sigma - 1/2 + theta}
    xpow = 0
    Dpow = -1 
    Tpow = 0
    Fpow = 1
    qpow = 0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_12'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_21(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} D^{1/14} q^{1/6}
    xpow = 0
    Dpow = 1/2.0 + 1/14.0
    Tpow = 0
    Fpow = -1
    qpow = 1/6.0
    mumpow = 0
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_21'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_22(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} s^{1/6} q^{1/6}
    xpow = 0
    Dpow = 1/2.0
    Tpow = 1/6.0
    Fpow = -1
    qpow = 1/6.0
    mumpow = 0
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_22'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_23(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} D^{1/14} mu-^{sigma - 1/2 + theta} q^{1/6}
    xpow = 0
    Dpow = 1/2.0 + 1/14.0
    Tpow = 0
    Fpow = -1
    qpow = 1/6.0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_23'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_24(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{1/2} F^{-1} s^{1/6} mu-^{sigma - 1/2 + theta} q^{1/6}
    xpow = 0
    Dpow = 1/2.0
    Tpow = 1/6.0
    Fpow = -1
    qpow = 1/6.0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_24'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_31(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{4/7} F^{-1} q^{7/6} |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))|
    xpow = 0
    Dpow = 4/7.0
    Tpow = 0
    Fpow = -1
    qpow = 9.1/12 #1/6.0 #7/6.0
    mumpow = 0
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    # |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| << D/(r|s|DeltaD)
    # Also, |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| = 1 + O(r|s|DeltaD/D)
    Dpow += 1
    Fpow += -1
    Tpow += -1
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'weird_deriv_thing':True, 'name':'t2s_31'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_32(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} D^{4/7} F^{-1} q^{7/6} mu-^{sigma - 1/2 + theta} |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))|
    xpow = 0
    Dpow = 4/7.0
    Tpow = 0
    Fpow = -1
    qpow = 9.1/12 #1/6.0 #7/6.0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    # |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| << D/(r|s|DeltaD)
    # Also, |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| = 1 + O(r|s|DeltaD/D)
    Dpow += 1
    Fpow += -1
    Tpow += -1
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'weird_deriv_thing':True, 'name':'t2s_32'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_41(sigma, r, theta, beta, alpha=0):
    # sigma < 1/2 - theta
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2} mu-^{sigma - 1/2 + theta} |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))|
    xpow = 0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    # |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| << D/(r|s|DeltaD)
    # Also, |D/DeltaD / (1 - r(s-1/2)) * (1 - (1 - DeltaD/D)^{1 - r(s-1/2)))| = 1 + O(r|s|DeltaD/D)
    Dpow += 1
    Fpow += -1
    Tpow += -1
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'weird_deriv_thing':True, 'name':'t2s_41'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_full1(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2}
    xpow = 0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    mumpow = 0
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_full1'}
    term = ErrorTerm(dat=dat)
    return term


def t2s_full2(sigma, r, theta, beta, alpha=0):
    # G(1-s)/G(s) (x/qD^r)^{sigma - 1/2}  mu-^{sigma - 1/2 + theta}
    xpow = 0
    Dpow = 0
    Tpow = 0
    Fpow = 0
    qpow = 0
    mumpow = sigma - 1/2.0 + theta
    # (x/qD^r)^{sigma - 1/2}
    xpow += sigma - 1/2.0
    Dpow += -r*(sigma - 1/2.0)
    qpow += -(sigma - 1/2.0)
    # G(1-s)/G(s)
    Tpow += r*(1/2.0 - sigma)
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'mumpow':mumpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'t2s_full2'}
    term = ErrorTerm(dat=dat)
    return term


### Perron


def perron_et(sigma, r, theta, beta, alpha=0):
    xpow = sigma - 1/2.0
    Dpow = 0
    Tpow = -1
    Fpow = 0
    qpow = 0
    dat = {'xpow':xpow, 'Dpow':Dpow, 'Tpow':Tpow, 'Fpow':Fpow, 'qpow':qpow, 'beta':beta, 'r':r, 'alpha':alpha, 'name':'perron_et'}
    term = ErrorTerm(dat=dat)
    return term



####



def integrand_size_random_samples(sigma_list, Fe_list, Te_list, r, theta=0, beta=None, alpha=None, overwrite=True, to_update=None):
    if to_update is None:
        to_update = {}
    product_list = itertools.product(sigma_list, Fe_list, Te_list)
    product_list = list(product_list)
    random.shuffle(product_list)
    for sv, Fev, Tev in product_list:
        if overwrite or ((sv, Fev, Tev) not in to_update):
            fixed_param_vals = {'sigma':sv}
            if beta is not None:
                fixed_param_vals['beta'] = beta
            if alpha is not None:
                fixed_param_vals['alpha'] = alpha
            dat = search_over_params(1, 1, r=r, fixed_param_vals=fixed_param_vals, integrand_size=True, F_exponent=Fev, T_exponent=Tev)
            to_update[(sv, Fev, Tev)] = dat
    return to_update



#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()', sort='cumtime')
    else:
        main()
