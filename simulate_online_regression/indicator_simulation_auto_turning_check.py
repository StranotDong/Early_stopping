from utils import *

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
import math
import scipy

"""
Predict the early stopping epoch by "adding-noise" method, 
currently assume that noise is rayleigh distributed
input:
    a, b, d: y = ax^b + d (x - s)
    var: the variance of unsmoothed data
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    smooth_win_size: the window size for smoothing
    num_samples: how many samples to generate when using bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit
    lower_limit: only consider the predicted epoch larger than upper_limit
    is_other_cases: if there is other extra a,b,d that we want to use to do the bootstrapping
    **kwargs: 
        weights: a list weights for each cases. The last element is associated with the original a,b,d.
            num_samples * weights is the number of bootstrapping samples for each case.
        coeffs: a list of extra (a,b,d) without the original a,b,d. The order of the elements should be associated
            with the order of elements in weights. The form is [(a1,b1,d1), (a2, b2, d2) ,...]
        vars: a list noise variances for each cases. The last element is associated with the original a,b,d.
         
return:
    mean: mean of all predicted epochs
    stopping_epochs: all predicted epochs
    sample: one of the sythetic data
    smoothed_sample: smoothing the sythetic data
"""
def early_stopping_prediction_adding_noise(a,b,d,linear_bias,
                                           var,min_delta,patience,
                                           num_epochs_between_eval,
                                           smooth_win_size,
                                           num_samples=100,
                                           upper_limit=2e4,
                                           lower_limit=0,
                                           is_other_cases=False,
                                           **kwargs
                                           ):
    num_points = int((upper_limit-lower_limit) // num_epochs_between_eval) # the number of noisy points we want to generate

    x = np.linspace(lower_limit, upper_limit, num_points)
    
    ys = []
    
    num_samples_list = []
    if is_other_cases:
        weights = np.array(kwargs['weights'])/np.sum(kwargs['weights']) # normalize        
    else:
        weights = np.array([1])
    num_samples_list = list(weights*num_samples)
    coeffs = []
    vars_ = []
    for i, w in enumerate(list(weights[:-1])):
        a_, b_, d_ = kwargs['coeffs'][i][0], kwargs['coeffs'][i][1], kwargs['coeffs'][i][2]
        ys.append(a_*np.power(x,b_) + d_*(x-linear_bias))
        vars_.append(kwargs['vars'][i])
        if w != 0: # collecting useful coeffs and return them
            coeffs.append((a_, b_, d_))
            
    ys.append(a*np.power(x,b) + d*(x-linear_bias))
    vars_.append(var)
    if weights[-1] != 0:
        coeffs.append((a, b, d))
        
    stopping_epochs = []
    samples = []
    smoothed_samples = []
    print(var)
    for j in range(len(weights)):
        for i in range(int(num_samples_list[j])):
            # noise = np.random.normal(0, np.sqrt(var), num_points)
            noise = np.random.rayleigh(np.sqrt(2*vars_[j]/(4-np.pi)), num_points)
            z = ys[j] + noise
            
            if i == 0:
                samples.append(z)
                smoothed_samples.append(smooth_by_linear_filter(z, smooth_win_size))
        
            try:
                stopping_epochs.append(early_stopping_step(z, min_delta, patience, smooth_win_size, num_epochs_between_eval) + lower_limit)
            except:
                continue

    return np.mean(stopping_epochs), np.array(stopping_epochs), samples, smoothed_samples, coeffs

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def CIs2errs(CIs):
    errs = []
    for ci in CIs:
        errs.append((ci[1]-ci[0])/2)
    
    return np.array(errs)

# find the first element that is greater or equal to target of an ordered list
# at most one target in the list
def _search_greater_0(list_, target):
    s = 0
    e = len(list_) - 1

    while s < e:
        m = int(s + (e-s)//2)
        if list_[m] == target:
            return m
        if list_[m] < target:
            s = m + 1
        else:
            e = m

    if list_[s] < target:
        return -1
    return s

def _regression_method(steps, y, weights, reg_method, **kwargs):
    if reg_method == 'linear_logx':
        # regression by linear regression on log-log scale
        a, b = power_regression(steps, y, weights)
        d = 0
    elif reg_method == 'power':
        # regression by numerical optimization
        fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))))*weights, 2))
        res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
        a = res.x[0]
        b = res.x[1]
        d = 0 # the coefficient of linear part should be 0 if we don't use linear part to fit the curve
    elif reg_method == 'power_linear':
        fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))+x[2]*(steps-steps[0])+x[3]))*weights,2))
        res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
        a = res.x[0]
        b = res.x[1]
        d = res.x[2]
    elif reg_method == 'power_shifted_linear':
        # fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))))*weights, 2))
        # res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
        # a = res.x[0]
        # b = res.x[1]
        # residual = y - power_function(steps, a, b)        
        # if kwargs['shift'] == None:
        #     d = 0
        # else:
        #     index = _search_greater_0(steps, kwargs['shift'])
        #     if index == -1:
        #         d = 0
        #     else:
        #         d, _ = linear_regression(steps[index:], residual[index:])
        #         d = max(d,0)
        if kwargs['shift'] == None:
            fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))))*weights, 2))
            res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
            a = res.x[0]
            b = res.x[1]
            d = 0
        else:
            index = _search_greater_0(steps, kwargs['shift'])
            if index == -1:
                fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))))*weights, 2))
                res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
                a = res.x[0]
                b = res.x[1]
                d = 0
            else:
                fun = lambda x: np.sum(np.power((y[:index]-(np.exp(x[1]*np.log(steps[:index])+np.log(x[0]))))*weights[:index], 2)) \
                + np.sum(np.power((y[index:]-(np.exp(x[1]*np.log(steps[index:])+np.log(x[0]))+x[2]*(steps[index:]-steps[index])))*weights[index:], 2))
                res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
                a = res.x[0]
                b = res.x[1]
                d = res.x[2]
    return a, b, max(d,0)


'''
simulate early stopping prediction by regression
input:
    earlyStoppingStep: ground truth
    data: the data on validation curve that used to predict the epoch
    epochs: on what epochs that do evaluation on validation dataset
    train_data: the data on train curve that used to predict the epoch
    train_epochs: on what epochs that do evaluation on train dataset
    smooth_win_size: window size for smoothing in the early stopping criteria
    epochs_between_eval: final number of epochs between two evals
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    pred_win_size: the window size for epoch prediction (when using validation curve), wrt points not real epochs
    period: the period for prediction (when using validation curve), wrt points not real epochs
    regression_method: three options:
        'linear_logx': power regression by linear regression on y-logx
        'power': power regression by numerical optimization
        'power_linear': power + linear regression by numerical optimization
        'power_shifted_linear': power + shifted linear regression by numerical optimization
    on_train_pred_win_size: the window size for epoch prediction (when using train curve), wrt points not real epochs
    on_train_period: the period for prediction (when using train curve), wrt points not real epochs
    is_power_linear: if using power + linear to do the regression
    bounds, inits, method: used in numerical optimal
    start_point: at which point we start the first prediction, wrt points not real epochs
    num_samples: how many samples to generate when predicting the epoch by bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit when doing bootstarpping
    online_smooth_win_size: before doing regression, firstly smooth the curve. Note this is different
        from smooth_win_size, which is a component used in early stopping criteria
    noise_est_win_size: the window size used for noise estimation (on validation curve). If none, it's equal to the pred_win_size
    train_noise_est_win_size:the window size used for noise estimation (on train curve). If none, it's equal to the on_train_pred_win_size
    pure_regression_on_train_end: the end of doing the prediction only based on training curve 
    pure_regression_on_val_begin: the beginning of doing the prediction only based on validation curve 
    pure_regression_on_regression_end: the end of doing the prediction only based on "regression on regression" method
    pure_regression_on_original_begin: the beginning of doing the prediction only based on "regression on original" method
    num_val_outliers: ignore the first num_val_outliers points
    max_reg_weight: the maximum regression weights. Only assign weights in power+linear mode. Note the minimum is 1.
    pure_simple_end_delay: after turning end, we should wait for several epochs to get enough points to do the shifted linear regression
        this is the end that we don't do power+shifted linear regression method
    pure_complex_begin_delay: the start that we only do power + shifted linear regression method
    turning_check_start: after getting this number of val_err, starting to check the turning curve
    first_order_num_points: use this number of points of val_err to do the linear regression in order to check turning
    second_order_num_points: use this number of points of the val_err slope to do the linear regression in order to check turning
    turning_check_patience: the number of consecutive checkpoints that satisfies the condition, then we think we find the turning begin or end
    thre_begin_high: when searching turning begin, if p-value of "H0: slope of slope > 0" is greater than this threshold, then we accept the null hypothesis
    thre_begin_low: when searching turning begin, if p-value of "H0: slope of slope = 0" is smaller than this threshold, then we reject the null hypothesis
    thre_begin_double_check, double_check_delay_epoch: when we find the turning begin, if within double_check_delay_epoch, there
        is one p-value of "H0: slope of slope = 0" is smaller than thre_begin_double_check, then we should discard the found turning begin
        and search for it after current global step
    thre_end: when searching turning end, if p-value of "H0: slope of slope = 0" is greater than this threshold, then we accept the null hypothesis
    num_interpolation: number of linear interpolated points between two slopes
    noise_var: variance of noise we want to add on the curve of slopes
output:
preds, rst_coeffs, shifts, samples, smoothed_samples, CIs, vars_, predicted_epoch_intevals
    preds: predicted epochs
    rst_coeffs: combination of all regression coeffients that have been done on a certain epoch [(a,b,d),...]
    shifts: the global steps on which doing predictions
    samples: the sythetic noisy data generated by all regression methods 
    smoothed_samples: smoothing the sythetic data
    CIs: 95% confidence intervals
    vars_: the estimated variances of noise at each prediction epoch
    predicted_epoch_intevals: the epoch intervals within which the epochs as well as associated val_err or train_err we used
        to do the predictions
'''
def powerRegressionIndicator(
                            ######################## simulation ######################
                            earlyStoppingStep, 
                            #########################################################
                            data, epochs, train_data, train_epochs,
                            smooth_win_size,
                            epochs_between_eval,
                            min_delta, patience,
                            pred_win_size, 
                            period, 
                            regression_method,
                            on_train_pred_win_size=None,
                            on_train_period=None,
                            bounds=None, inits=None, method='TNC',
                            start_point=100,
                            num_samples=100,
                            upper_limit=2e4,
                            online_smooth_win_size=1,
                            noise_est_win_size=None, train_noise_est_win_size=None,
                            pure_regression_on_train_end=25,
                            pure_regression_on_val_begin=100,
                            pure_regression_on_regression_delay=1000,
                            regression_transition_epochs=2000,
                            num_val_outliers=10,
                            # below are some parameters for power_shifted_linear method
                            max_reg_weight=10,
                            pure_simple_end_delay=1500,
                            pure_complex_begin_delay=3000,
                            # below are some parameters for turning curve check
                            turning_check_start_point=10,
                            first_order_num_points=100, second_order_num_points=15, turning_check_patience=20,
                            thre_begin_high=9e-1, thre_begin_low=1e-2, 
                            thre_begin_double_check=1e-6, double_check_delay_epochs=1000,
                            thre_end=1e-1,
                            num_interpolation=0, noise_var=1e-10
                            ):
    # original_data = data
    if smooth_win_size % 2 == 0 or online_smooth_win_size %2 == 0:
        raise ValueError('Please set (online) smoothed_win_size to be odd')

    if noise_est_win_size == None:
        noise_est_win_size = pred_win_size
    if on_train_pred_win_size == None:
        on_train_pred_win_size = pred_win_size
    if on_train_period == None:
        on_train_period = period
    if train_noise_est_win_size == None:
        train_noise_est_win_size = on_train_pred_win_size

    preds = []
    rst_coeffs = []
    shifts = []
    samples = []
    smoothed_samples = []
    CIs = []
    vars_ = []
    predicted_epoch_intevals = []
    pred = 0
    smoothed_data = None # smoothed by online_smooth_win_size
    smoothed_data_2 = None # smoothed by smooth_win_size

    # a queue to store current indices of data and epochs used to do the regression
    # max size is pred_win_size
    # firstly put the data into the buffer, then do the smoothing and put the smoothed data
    # into val_reg_data
    # the max size of the buffer is online_smooth_win_size, but when we get online_smooth_win_size // 2
    # elements in the buffer, we get the first smoothed data
    # val_data_buffer_2 is used to do the smoothing by smooth_win_size. the max size of the buffer is smooth_win_size, 
    # but when we get smooth_win_size // 2
    # elements in the buffer, we get the first smoothed data
    val_reg_steps = []
    val_data_buffer = []
    val_data_buffer_2 = []
    val_reg_data = []
    # a queue to store current indices of data and epochs used to estimate variance
    # max size is noise_est_win_size
    val_noise_est_steps = []
    val_noise_est_data = []
    val_noise_est_smoothed_data = []
    # turning curve check
    turning_check_first_order_steps = []
    turning_check_first_order_data = []
    turning_check_second_order_steps = []
    turning_check_second_order_data = []
    should_search_turning_begin = True
    should_search_turning_end = True
    turning_begin_accept_count = 0
    turning_end_accept_count = 0
    turning_begin = None
    turning_end = None

    # a queue to store current indices of train data and train epochs used to do the regression
    # max size is pred_win_size
    train_reg_steps = []
    train_reg_data = []
    # a queue to store current indices of train data and train epochs used to estimate variance
    # max size is noise_est_win_size
    train_noise_est_steps = []
    train_noise_est_data = []

    prev_regression_on_regression_weight = 1

    ################### simulation #####################
    global_step = 0
    train_pointer = 0
    val_pointer = 0
    while True:
        if global_step > earlyStoppingStep or val_pointer >= len(data) or val_pointer >= pure_regression_on_train_end + num_val_outliers:
            # print(global_step > earlyStoppingStep, val_pointer >= len(data), global_step >= pure_regression_on_train_end)
            break
        global_step = train_epochs[train_pointer]       
        # print(global_step, train_pointer)
        # regression train data and train epochs
        if len(train_reg_steps) >= on_train_pred_win_size:
            train_reg_steps.pop(0)
            train_reg_data.pop(0)
        train_reg_steps.append(train_epochs[train_pointer])
        train_reg_data.append(train_data[train_pointer])
        # train data and train epoch for estimating noise
        if len(train_noise_est_steps) >= train_noise_est_win_size:
            train_noise_est_steps.pop(0)
            train_noise_est_data.pop(0)    
        train_noise_est_steps.append(train_epochs[train_pointer])
        train_noise_est_data.append(train_data[train_pointer])

        # at the beginning, do the regression on training data/err
        while True:
            if val_pointer >= len(data) or epochs[val_pointer] > train_epochs[train_pointer]:
                break
            # smoothing window here is customized
            if len(val_data_buffer) >= online_smooth_win_size:
                val_data_buffer.pop(0)
            if val_pointer >= num_val_outliers:
                val_data_buffer.append(data[val_pointer])
            if online_smooth_win_size == 1 and len(val_data_buffer) == 1:
                smoothed_data = val_data_buffer[0]
            elif len(val_data_buffer) > online_smooth_win_size // 2:
                smoothed_data = np.mean(val_data_buffer)

            # smooth window here is the same as the one in early stopping criteria
            if len(val_data_buffer_2) >= smooth_win_size:
                val_data_buffer_2.pop(0)
            if val_pointer >= num_val_outliers:
                val_data_buffer_2.append(data[val_pointer])
            if smooth_win_size == 1 and len(val_data_buffer_2) == 1:
                smoothed_data_2 = val_data_buffer_2[0]
            elif len(val_data_buffer_2) > smooth_win_size // 2:
                smoothed_data_2 = np.mean(val_data_buffer_2)

            # data and epoch used for power regression
            if len(val_reg_steps) >= pred_win_size:
                val_reg_steps.pop(0)
                val_reg_data.pop(0) # in this case, the smoothed data is impossible to be None
            if smoothed_data != None:
                val_reg_steps.append(epochs[val_pointer - online_smooth_win_size // 2])
                val_reg_data.append(smoothed_data)

            # data and epoch for estimating noise
            if len(val_noise_est_steps) >= noise_est_win_size:
                val_noise_est_steps.pop(0)
                val_noise_est_data.pop(0)
            if val_pointer >= num_val_outliers:           
                val_noise_est_steps.append(epochs[val_pointer])
                val_noise_est_data.append(data[val_pointer])
            if len(val_noise_est_smoothed_data) >= noise_est_win_size - smooth_win_size // 2:
                val_noise_est_smoothed_data.pop(0)
            if smoothed_data_2 != None:
                val_noise_est_smoothed_data.append(smoothed_data_2)

            # data and epoch for checking turning curve
            if len(turning_check_first_order_steps) >= first_order_num_points:
                turning_check_first_order_steps.pop(0)
                turning_check_first_order_data.pop(0)
            if val_pointer >= num_val_outliers:
                turning_check_first_order_steps.append(epochs[val_pointer])
                turning_check_first_order_data.append(data[val_pointer])
            if len(turning_check_first_order_steps) >= turning_check_start_point:
                turning_check_slope, _, _, _, _ = scipy.stats.linregress(turning_check_first_order_steps,turning_check_first_order_data)
            if len(turning_check_second_order_steps) >= second_order_num_points:
                turning_check_second_order_steps.pop(0)
                turning_check_second_order_data.pop(0)
            if len(turning_check_first_order_steps) >= turning_check_start_point: # can store related values in second order buffer now
                turning_check_second_order_steps.append(epochs[val_pointer])
                turning_check_second_order_data.append(turning_check_slope)
            if len(turning_check_second_order_steps) >= second_order_num_points:
                p_value, p_value2 = _p_values_cal(turning_check_second_order_steps, turning_check_second_order_data, 
                                                  num_interpolation=num_interpolation, noise_var=noise_var)
                # print("p_value:{}, p_value2:{}".format(p_value, p_value2))
                if should_search_turning_begin:
                    if p_value > thre_begin_high and p_value2 < thre_begin_low:
                        turning_begin_accept_count += 1
                    if turning_begin_accept_count >= turning_check_patience:
                        turning_begin_delay = epochs[val_pointer]
                        turning_begin = epochs[val_pointer-turning_check_patience-second_order_num_points+1]
                        print("turning_begin:{}, {}".format(turning_begin, turning_begin_delay))
                        should_search_turning_begin = False
                if not should_search_turning_begin \
                    and epochs[val_pointer-turning_check_patience-second_order_num_points+1] -  double_check_delay_epochs < turning_begin \
                    and p_value < thre_begin_double_check:
                    should_search_turning_begin = True
                    turning_begin_accept_count = 0
                if should_search_turning_end and not should_search_turning_begin:
                    if p_value2 > thre_end:
                        turning_end_accept_count += 1
                    if turning_end_accept_count >= turning_check_patience:
                        turning_end_delay = epochs[val_pointer]
                        turning_end = epochs[val_pointer-turning_check_patience-second_order_num_points+1]
                        print("turning_end:{}, {}".format(turning_end, turning_end_delay))
                        should_search_turning_end = False

            val_pointer += 1
        
        # start to predict periodically
        # at the beginning, do the regression on training data/err
        if train_pointer >= start_point and (train_pointer - start_point) % on_train_period == 0:
            print("Global Step: {}".format(global_step))

            # always use "power" method on train err
            a, b, d = _regression_method(train_reg_steps, train_reg_data, np.ones_like(train_reg_steps), 'power', 
                                        shift=0, inits=inits, method=method, bounds=bounds)
            # a, b = power_regression(train_reg_steps, train_reg_data, weights)
            # d = 0
            # use the original - power_regression_on_original to calculate variance
            # TODO: we can also use original - linear_regression_on_original to calculate variance
            # TODO: variance could also be estimated on the validation curve
            var = np.var(train_noise_est_data-(a*np.power(train_noise_est_steps,b) + d*(train_noise_est_steps-train_reg_steps[0])))
            # var *= 5
            # linear regression
            a1, b1 = linear_regression(train_reg_steps, train_reg_data)
            # var = np.var(np.array(train_noise_est_data)-a1*np.array(train_noise_est_steps)-b1)

            # calculate regression_on_regression_weight
            regression_on_regression_weight, _ = _regression_on_regression_original_weights(
                                                                should_search_turning_begin, 
                                                                prev_regression_on_regression_weight,
                                                                global_step,
                                                                pure_regression_on_regression_delay, 
                                                                regression_transition_epochs,
                                                                turning_begin=turning_begin)
            prev_regression_on_regression_weight = regression_on_regression_weight
                    
            if a1 >= 0: # use regression on original if a1 is greater than 0
                is_other_cases = False
                al, bl, dl = 0, 0, 0
            else:
                is_other_cases = True
                linear_end = math.floor(-b1/a1)
                if b1/a1 == 0:
                    linear_end -= 1
                steps_l = np.arange(train_reg_steps[0], linear_end+1)
                yl = a1*steps_l+b1
                al, bl = power_regression(steps_l, yl, np.ones(len(steps_l)))
                dl = 0
            pred, stopping_epochs, sample, smoothed_sample, coeffs = early_stopping_prediction_adding_noise(
                                                            a,b,d,train_reg_steps[0],var,
                                                            min_delta,patience,
                                                             epochs_between_eval,
                                                             smooth_win_size,
                                                             num_samples=num_samples,
                                                             upper_limit=upper_limit,
                                                             lower_limit=global_step,
                                                             is_other_cases=is_other_cases,
                                                             weights=[regression_on_regression_weight,1-regression_on_regression_weight],
                                                             coeffs=[(al,bl,dl)],
                                                             vars=[var])
            print("Predicted Stopping epoch is {}. a = {}, b = {}, d^2={} on training curve".format(pred, al, bl, dl))
            print(var)
            # coeff[1] = (al,bl,dl)
       
            _, CI_left, CI_right = mean_confidence_interval(stopping_epochs)
            
            preds.append(pred)
            CIs.append((CI_left, CI_right))
            rst_coeffs.append(coeffs)
            shifts.append(global_step)
            samples.append(sample)
            smoothed_samples.append(smoothed_sample)            
            vars_.append(var)
            predicted_epoch_intevals.append((train_reg_steps[0], train_reg_steps[-1]))

        train_pointer += 1

    # a transition combining train and val, followed by pure val
    # pure_regression_on_val_begin += val_pointer
    # pure_regression_on_train_end = val_pointer
    on_val_start_point = val_pointer
    low_flag = False
    while True:
        if global_step > earlyStoppingStep or val_pointer >= len(data):
            break

        global_step = epochs[val_pointer]

        # regression data and epochs
        if len(val_data_buffer) >= online_smooth_win_size:
            val_data_buffer.pop(0)
        if val_pointer >= num_val_outliers:
            val_data_buffer.append(data[val_pointer])
        if online_smooth_win_size == 1 and len(val_data_buffer) == 1:
            smoothed_data = val_data_buffer[0]
        elif len(val_data_buffer) > online_smooth_win_size // 2:
            smoothed_data = np.mean(val_data_buffer)

        if len(val_data_buffer_2) >= smooth_win_size:
            val_data_buffer_2.pop(0)
        if val_pointer >= num_val_outliers:
            val_data_buffer_2.append(data[val_pointer])
        if smooth_win_size == 1 and len(val_data_buffer_2) == 1:
            smoothed_data_2 = val_data_buffer_2[0]
        elif len(val_data_buffer_2) > smooth_win_size // 2:
            smoothed_data_2 = np.mean(val_data_buffer_2)

        if len(val_reg_steps) >= pred_win_size:
            val_reg_steps.pop(0)
            val_reg_data.pop(0) # in this case, the smoothed data is impossible to be None
        if smoothed_data != None:
            val_reg_steps.append(epochs[val_pointer - online_smooth_win_size // 2])
            val_reg_data.append(smoothed_data)

        # data and epoch for estimating noise
        if len(val_noise_est_steps) >= noise_est_win_size:
            val_noise_est_steps.pop(0)
            val_noise_est_data.pop(0)   
        if val_pointer >= num_val_outliers:         
            val_noise_est_steps.append(epochs[val_pointer])
            val_noise_est_data.append(data[val_pointer])
        if len(val_noise_est_smoothed_data) >= noise_est_win_size - smooth_win_size // 2:
            val_noise_est_smoothed_data.pop(0)
        if smoothed_data_2 != None:
            val_noise_est_smoothed_data.append(smoothed_data_2)

        # data and epoch for checking turning curve
        if len(turning_check_first_order_steps) >= first_order_num_points:
            turning_check_first_order_steps.pop(0)
            turning_check_first_order_data.pop(0)
        if val_pointer >= num_val_outliers:
            turning_check_first_order_steps.append(epochs[val_pointer])
            turning_check_first_order_data.append(data[val_pointer])
        if len(turning_check_first_order_steps) >= turning_check_start_point:
            turning_check_slope, _, _, _, _ = scipy.stats.linregress(turning_check_first_order_steps,turning_check_first_order_data)
        if len(turning_check_second_order_steps) >= second_order_num_points:
            turning_check_second_order_steps.pop(0)
            turning_check_second_order_data.pop(0)
        if len(turning_check_first_order_steps) >= turning_check_start_point: # can store related values in second order buffer now
            turning_check_second_order_steps.append(epochs[val_pointer])
            turning_check_second_order_data.append(turning_check_slope)
        if len(turning_check_second_order_steps) >= second_order_num_points:
            p_value, p_value2 = _p_values_cal(turning_check_second_order_steps, turning_check_second_order_data, 
                                              num_interpolation=num_interpolation, noise_var=noise_var)
            # print("p_value:{}, p_value2:{}".format(p_value, p_value2))
            if should_search_turning_begin:
                if p_value > thre_begin_high and p_value2 < thre_begin_low:
                    turning_begin_accept_count += 1
                if turning_begin_accept_count >= turning_check_patience:
                    turning_begin_delay = epochs[val_pointer]
                    turning_begin = epochs[val_pointer-turning_check_patience-second_order_num_points+1]
                    print("turning_begin:{}, {}".format(turning_begin, turning_begin_delay))
                    should_search_turning_begin = False
            if not should_search_turning_begin \
                and epochs[val_pointer-turning_check_patience-second_order_num_points+1] -  double_check_delay_epochs < turning_begin \
                and p_value < thre_begin_double_check:
                should_search_turning_begin = True
                turning_begin_accept_count = 0
            if should_search_turning_end and not should_search_turning_begin:
                if p_value2 > thre_end:
                    turning_end_accept_count += 1
                if turning_end_accept_count >= turning_check_patience:
                    turning_end_delay = epochs[val_pointer]
                    turning_end = epochs[val_pointer-turning_check_patience-second_order_num_points+1]
                    print("turning_end:{}, {}".format(turning_end, turning_end_delay))
                    should_search_turning_end = False

        while True:
            if train_pointer >= len(train_data) or epochs[val_pointer] < train_epochs[train_pointer]:
                break
            # regression train data and train epochs
            if len(train_reg_steps) >= on_train_pred_win_size:
                train_reg_steps.pop(0)
                train_reg_data.pop(0)
            train_reg_steps.append(train_epochs[train_pointer])
            train_reg_data.append(train_data[train_pointer])
            # train data and train epoch for estimating noise
            if len(train_noise_est_steps) >= train_noise_est_win_size:
                train_noise_est_steps.pop(0)
                train_noise_est_data.pop(0)    
            train_noise_est_steps.append(train_epochs[train_pointer])
            train_noise_est_data.append(train_data[train_pointer])

            train_pointer += 1

        if val_pointer >= on_val_start_point and (val_pointer - on_val_start_point) % period == 0:
            print("Global Step: {}".format(global_step))

            # calculate the val_reg_weights
            # the condition that we begin to do the "power_shifted_linear" method is that
            ## the regression method is power_shifted_linear
            ## after the turning point end is found
            ## after the beginning of doing regression only on val err
            ## after there are enough points after the turning point end
            if regression_method == 'power_shifted_linear' and not should_search_turning_end \
                    and val_pointer > pure_regression_on_val_begin + num_val_outliers \
                     and global_step >= pure_simple_end_delay+turning_end:
                cur_max_reg_weight = min(max_reg_weight, max_reg_weight * (global_step - turning_end - pure_simple_end_delay)/(pure_complex_begin_delay - pure_simple_end_delay))
                val_reg_weights = _weights_generator(val_reg_steps, turning_begin, turning_end, cur_max_reg_weight)
                linear_part_shift = turning_end
                val_regression_method = regression_method
                simple_weight = max((global_step - turning_end - pure_complex_begin_delay)/(pure_simple_end_delay - pure_complex_begin_delay),0)
            else:
                val_reg_weights = np.ones(len(val_reg_steps))
                linear_part_shift = None
                # before doing "power linear in order" method, we just do 'power_linear' method
                if regression_method == 'power_shifted_linear':
                    val_regression_method = 'power_linear'
                else:
                    val_regression_method = regression_method
                simple_weight = 1


            # val_reg_weights = np.ones_like(val_reg_steps)
            a, b, d = _regression_method(val_reg_steps, val_reg_data, val_reg_weights, val_regression_method, 
                                        inits=inits, method=method, bounds=bounds, shift=linear_part_shift)
            # use the original - power_regression_on_original to calculate variance
            # if the points is not enough, we use the regression line to estimate the noise
            if len(val_noise_est_smoothed_data) >= 10:
	            var = np.var(np.array(val_noise_est_data[:len(val_noise_est_data)-smooth_win_size//2]) - np.array(val_noise_est_smoothed_data))
            else:
	            var = np.var(val_noise_est_data-(a*np.power(val_noise_est_steps,b) + d*(val_noise_est_steps-val_reg_steps[0])))
            # var = np.var(np.array(val_noise_est_data[:len(val_noise_est_data)-smooth_win_size//2]) - np.array(val_noise_est_smoothed_data))
            if not should_search_turning_end:
                var /= max((4-(global_step - turning_end)*(4-1)/turning_end),1)
            else:
                var /= 4

            # calculate regression_on_regression_weight
            regression_on_regression_weight, regression_on_original_weight = _regression_on_regression_original_weights(
                                                should_search_turning_begin, 
                                                prev_regression_on_regression_weight,
                                                global_step,
                                                pure_regression_on_regression_delay, 
                                                regression_transition_epochs,
                                                turning_begin=turning_begin)
            prev_regression_on_regression_weight = regression_on_regression_weight

            if val_pointer > pure_regression_on_val_begin + num_val_outliers:
                a1, b1 = linear_regression(val_reg_steps, val_reg_data)
                if a1 >= 0:
                    is_other_cases = False
                    al, bl, dl = 0, 0, 0
                else:
                    is_other_cases = True                
                    linear_end = math.floor(-b1/a1)              
                    if b1/a1 == 0:
                        linear_end -= 1
                    steps_l = np.arange(val_reg_steps[0], linear_end+1)
                    yl = a1*steps_l+b1
                    al, bl = power_regression(steps_l, yl, np.ones(len(steps_l)))
                    dl = 0
                
                if simple_weight == 1:
                    pred, stopping_epochs, sample, smoothed_sample, returned_coeffs = early_stopping_prediction_adding_noise(
                                                                    a,b,d,val_reg_steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=global_step,
                                                                     is_other_cases=is_other_cases,
                                                                     weights=[regression_on_regression_weight,1-regression_on_regression_weight],
                                                                     coeffs=[(al,bl,dl)],
                                                                     vars=[var])
                else:
                    input_weight2 = (1-regression_on_regression_weight)*simple_weight
                    input_weight3 = (1-regression_on_regression_weight)*(1-simple_weight)
                    a_simple, b_simple, d_simple = _regression_method(val_reg_steps, val_reg_data, val_reg_weights, 'power_linear', 
                                        inits=inits, method=method, bounds=bounds, shift=None)
                    pred, stopping_epochs, sample, smoothed_sample, returned_coeffs = early_stopping_prediction_adding_noise(
                                                                    a,b,d,val_reg_steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=global_step,
                                                                     is_other_cases=is_other_cases,
                                                                     weights=[regression_on_regression_weight, input_weight2, input_weight3],
                                                                     coeffs=[(al,bl,dl), (a_simple, b_simple, d_simple)],
                                                                     vars=[var,var])
                
            else:                
	            # deal with weights
                train_weight = min(max((epochs[pure_regression_on_val_begin-1] - global_step)\
                    /(epochs[pure_regression_on_val_begin - 1] - epochs[pure_regression_on_train_end-1]),0),1)
                val_weight = 1 - train_weight
                weights = [train_weight * regression_on_regression_weight, train_weight * regression_on_original_weight,
                            val_weight * regression_on_regression_weight, val_weight * regression_on_original_weight]

                coeffs = []
                # train curve: regression on regression
                a1, b1 = linear_regression(train_reg_steps, train_reg_data)
                if a1 >= 0:
                    weights[1] += weights[0]
                    weights[0] = 0
                    al, bl, dl = 0, 0, 0
                else:
                    linear_end = math.floor(-b1/a1)
                    if b1/a1 == 0:
                        linear_end -= 1
                    steps_l = np.arange(train_reg_steps[0], linear_end+1)
                    yl = a1*steps_l+b1
                    al, bl = power_regression(steps_l, yl, np.ones(len(steps_l)))
                coeffs.append((al, bl,dl))
                
                # train curve: regression on original
                # always use "power" method on train err
                al, bl, dl = _regression_method(train_reg_steps, train_reg_data, np.ones_like(train_reg_steps), 'power', 
                                                shift=0, inits=inits, method=method, bounds=bounds)
                coeffs.append((al, bl, dl))
                # get train variance here
                train_var = np.var(train_noise_est_data-(al*np.power(train_noise_est_steps,bl) + dl*(train_noise_est_steps-train_reg_steps[0])))

                # val curve: regression on regression
                a1, b1 = linear_regression(val_reg_steps, val_reg_data)
                if a1 >= 0:
                    weights[3] += weights[2]
                    weights[2] = 0
                    al, bl, dl = 0, 0, 0
                else:
                    linear_end = math.floor(-b1/a1)
                    if b1/a1 == 0:
                        linear_end -= 1
                    steps_l = np.arange(val_reg_steps[0], linear_end+1)
                    yl = a1*steps_l+b1
                    al, bl = power_regression(steps_l, yl, np.ones(len(steps_l)))
                coeffs.append((al, bl,dl))

                pred, stopping_epochs, sample, smoothed_sample , returned_coeffs = early_stopping_prediction_adding_noise(
                                                                a,b,d,val_reg_steps[0],var,
                                                                min_delta,patience,
                                                                 epochs_between_eval,
                                                                 smooth_win_size,
                                                                 num_samples=num_samples,
                                                                 upper_limit=upper_limit,
                                                                 lower_limit=global_step,
                                                                 is_other_cases=True,
                                                                 weights=weights,
                                                                 coeffs=coeffs,
                                                                 vars=[train_var, train_var, var])

            print("Predicted Stopping epoch is {}. a = {}, b = {}, d^2={} on validation curve".format(pred, a, b, d))
            _, CI_left, CI_right = mean_confidence_interval(stopping_epochs)
            
            preds.append(pred)
            CIs.append((CI_left, CI_right))
            rst_coeffs.append(returned_coeffs)
            shifts.append(global_step)
            samples.append(sample)
            smoothed_samples.append(smoothed_sample)            
            vars_.append(var)
            predicted_epoch_intevals.append((val_reg_steps[0], val_reg_steps[-1]))

        val_pointer += 1
     
    return preds, rst_coeffs, shifts, samples, smoothed_samples, CIs, vars_, predicted_epoch_intevals


# The max_weight will assigned to (turning+begin+turning_end) // 2
# linear decay to 1 with epoch increasing to turning_end
# or decrease to 1 
def _weights_generator(epochs, turning_begin, turning_end, max_weight):
    weights = []
    max_weight_epoch = (turning_begin+turning_end) // 2
    for e in epochs:
        if e < turning_end:
            weights.append(max_weight/max_weight_epoch*e)
        else:
            weights.append(max_weight/max_weight_epoch*(turning_end-e))
    return np.array(weights)

# interpolation
def _interpolation(steps, values, num_inters):
    rst_steps = []
    rst_values = []
    for i in range(len(steps)-1):
        rst_steps.append(steps[i])
        rst_values.append(values[i])
        for j in range(num_inters):
            rst_steps.append((steps[i+1]*(j+1)+steps[i]*(num_inters-j))/(num_inters+1))
            rst_values.append((values[i+1]*(j+1)+values[i]*(num_inters-j))/(num_inters+1))
    
    rst_steps.append(steps[-1])
    rst_values.append(values[-1])
    return rst_steps, rst_values


# given steps and slopes, calculate the p-values of the slope of slope
# p_value is based on H0: slope_slope > 0
# p_value2 is based on H0: slope_slope = 0
def _p_values_cal(steps, slopes, num_interpolation=0, noise_var=1e-10):
    noise_steps, interpolation_slopes = _interpolation(steps, slopes, num_interpolation)
    noise = np.random.normal(0, np.sqrt(noise_var), len(interpolation_slopes))
    noise_slopes = interpolation_slopes + noise

    slope_slope, _, _, p_value2, std_err = scipy.stats.linregress(noise_steps,noise_slopes)
    t_value = slope_slope / std_err
    p_value = scipy.stats.t.cdf(t_value, df=len(noise_steps)-2)

    return p_value, p_value2

# the function to define the weights between doing regression on regression and regression on original
def _regression_on_regression_original_weights(should_search_turning_begin, 
                                               prev_weight,
                                               global_step,
                                               pure_regression_on_regression_delay, 
                                               regression_transition_epochs,
                                               turning_begin=None):
    if should_search_turning_begin:
        regression_on_regression_weight = prev_weight
    else:
        pure_regression_on_regression_end = turning_begin + pure_regression_on_regression_delay
        pure_regression_on_original_begin = turning_begin + pure_regression_on_regression_delay + regression_transition_epochs
        if global_step < pure_regression_on_regression_end:
            regression_on_regression_weight = prev_weight
        elif global_step >= pure_regression_on_original_begin:
            regression_on_regression_weight = 0
        else:
            regression_on_regression_weight = min((pure_regression_on_original_begin - global_step) \
                /(pure_regression_on_original_begin - pure_regression_on_regression_end), prev_weight)

    return regression_on_regression_weight, 1-regression_on_regression_weight











"""
check if the given values are almost in a horizontal line. 
If the absolute difference of any two consecutive value is larger than the threshold,
then the value is not in a plateau
"""
def check_plateau(values, steps, slope_threshold, step_threshold, mode):   
    if len(values) == 0 or len(values) == 1:
        return False

    # linear regression way
    a, _ = linear_regression(steps, values)
    # print(a)
    if mode == 'high':
        if a < -slope_threshold or values[-1] - values[len(values)-2] < -step_threshold:
            return False
    elif mode == 'low':
        if a > slope_threshold or a < -slope_threshold or abs(values[-1] - values[len(values)-2]) > step_threshold:
            return False

    return True

"""
change patience in the process of prediction
"""
def patience_dynamic(global_steps, preds, cur_patience, check_plateau_thre, increase_decrease_patience_thre):
    remaining_time_list = np.array(preds) - np.array(global_steps)

    # check if the curve is in the plateau, 
    # if not, do nothing
    if not check_plateau(list(remaining_time_list), global_steps, check_plateau_thre):
        return cur_patience

    # if the mean of the remaining time is smaller than a threshold,
    # then make the patience + 1
    if np.mean(remaining_time_list) <= increase_decrease_patience_thre:
        cur_patience += 1
        return cur_patience

    # if the mean of the remaining time is larger than a threshold
    # and the curve is in the plateau
    # then make the patience - 1
    cur_patience -= 1
    return cur_patience