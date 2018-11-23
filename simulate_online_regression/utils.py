from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
import math
import numpy as np

'''
power function y = ax^b, logy = loga + b*logx
'''
def power_function(x, a, b):
    return np.exp(np.log(a)+b*np.log(x))

'''
decide the actual win_size
'''
def actual_win_size(win_size):
    half_win = int(math.floor(win_size/2))
    total_win = half_win*2+1
    return total_win, half_win
'''
smoothing by linear filtering
'''
def smooth_by_linear_filter(data, win_size, win_type='rectangular'):
#    half_win = int(math.floor(win_size/2))
#    total_win = half_win*2+1
    total_win, half_win = actual_win_size(win_size)
    if win_type == 'gaussian':
        data_smoothed = gaussian_filter(data, total_win)
        return data_smoothed
    if win_type == 'wiener':
        data_smoothed = wiener(data, total_win)
        return data_smoothed
    if win_type == 'rectangular':
        weights = np.ones(total_win)/total_win
    elif win_type == 'hamming':
        weights = np.hamming(total_win)/np.sum(np.hamming(total_win))
    data_smoothed = []
    for i in range(len(data)):
        end = i+half_win
        if i+half_win >= len(data):
            break
        if i-half_win < 0:
            start = 0
            data_smoothed.append(np.mean(data[start:end+1]))
        else:
            start = i-half_win
            data_smoothed.append(weights.dot(data[start:end+1].T))
    return np.array(data_smoothed)

def early_stopping_step(data, min_delta, patience, win_size, epochs_between_eval):
    smoothed_data = smooth_by_linear_filter(data, win_size)
    patience_cnt = 0
    rst = 0
    for epoch in range(len(data)):
        if epoch > 0:
            if smoothed_data[epoch-1] - smoothed_data[epoch] > min_delta:
                rst = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                break

    return (rst+1)*epochs_between_eval


'''
power regresssion: y = ax^b
log y = b * log x + log a
'''
def power_regression(x,y,weights):
    # coeff, _ = np.polynomial.polynomial.polyfit(np.log(x), np.log(y), deg=1, w=weights)
    # print(coeff)
    # return math.exp(coeff[0]), coeff[1]
    A = np.vstack([np.log(x)*weights, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, np.log(y)*weights, rcond=-1)[0]
    return math.exp(a), b

'''
solve the inequality: ax^b - a(x+n)^b < delta
'''
def _power_ineq_sol(a, b, n, delta, search_start=None):
    # if left-hand side is smaller than right-hand side
    def _is_smaller(x):
        return abs(a*np.power(x,b) - a*np.power(x+n,b)) < delta

    if search_start == None:
        search_start = 1e4
    s = 0
    e = search_start
    while True:
        if not  _is_smaller(e):
            s = e
            e *= 2
        else:
            break
    while e != s:
        mid = s + (e - s) // 2
        if _is_smaller(mid):
            e = mid
        else:
            s = mid + 1
    return s

'''
predict the early stopping epoch
the early stopping criteria here is no improving for several epochs
x should be a float
'''
def early_stopping_prediction(x, y, delta, patience,weights,search_start=None):
    a,b  = power_regression(x, y, weights=weights)
    # num epochs between two validation
    num_epochs_between = round((x[-1]-x[0])/(len(x)-1))
    x_patience_0 = _power_ineq_sol(a,b,num_epochs_between,delta,search_start=search_start)
    return x_patience_0 + patience*num_epochs_between, a, b



'''
simulate early stopping prediction
'''
def sim_online_early_stopping_prediction(data, epochs_between_eval, min_delta, patience, win_size, left_tail_size, period, start_point=100, weights_type='linear'):
    ########################################
    global_step = 0
    ########################################
    # generate weights
    w1_size = win_size - left_tail_size # the number of elements we assign weight 1
    if weights_type == 'linear':
        basic_weights0 = np.linspace(0,1,left_tail_size)
    elif weights_type == 'equal':
        basic_weights0 = np.ones(left_tail_size)
    basic_weights1 = np.ones(w1_size)
    basic_weights = np.concatenate((basic_weights0, basic_weights1))
    def weights_generator(length):
        if length <= w1_size:
            rst = np.ones(length)
        elif length <= win_size:
            s = win_size - length
            rst = basic_weights[s:]
        else:
            z = np.zeros(length - win_size)
            rst = np.concatenate((z, basic_weights))

        return rst


    preds = []
    coeffs = []
    shifts = []
        ########################################
    for i in range(len(data)):
        global_step += epochs_between_eval
        ########################################
        num_evals = global_step//epochs_between_eval

        if num_evals >= start_point and (num_evals-start_point)%period == 0:
            if num_evals < win_size:
                s = 0
            else:
                s = num_evals - win_size
            e = num_evals
            # x = (np.arange(s,e)+1) * epochs_between_eval
            # shift the piece of curve 
            x = (np.arange(0,e-s)+1) * epochs_between_eval
            y = data[s:e]
            weights = weights_generator(len(y))
            if len(preds) == 0:
                search_start = None
            else:
                search_start = preds[-1]

            pred, a, b = early_stopping_prediction(x, y, min_delta, patience,weights=weights, search_start=search_start)
            # we should do this because we shift the curve before
            pred += s*epochs_between_eval
            preds.append(pred)
            coeffs.append((a,b))
            shifts.append(s*epochs_between_eval)

    return preds, coeffs, shifts
