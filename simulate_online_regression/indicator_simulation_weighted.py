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
    for i, w in enumerate(list(weights[:-1])):
        a_, b_, d_ = kwargs['coeffs'][i][0], kwargs['coeffs'][i][1], kwargs['coeffs'][i][2]
        ys.append(a_*np.power(x,b_) + d_*(x-linear_bias))
        if w != 0: # collecting useful coeffs and return them
            coeffs.append((a_, b_, d_))
    ys.append(a*np.power(x,b) + d*(x-linear_bias))
    if weights[-1] != 0:
        coeffs.append((a, b, d))

    stopping_epochs = []
    samples = []
    smoothed_samples = []
    print(var)
    for j in range(len(weights)):
        for i in range(int(num_samples_list[j])):
            # noise = np.random.normal(0, np.sqrt(var), num_points)
            noise = np.random.rayleigh(np.sqrt(2*var/(4-np.pi)), num_points)
            # if lower_limit >= 8000:
            #     noise = np.random.rayleigh(np.sqrt(2/2*var/(4-np.pi)), num_points) #- var*np.sqrt(np.pi/2)
            # else:
            #     noise = np.random.rayleigh(np.sqrt(2/4*var/(4-np.pi)), num_points)
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
    elif reg_method == 'power_num_op':
        # regression by numerical optimization
        fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))))*weights, 2))
        res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
        a = res.x[0]
        b = res.x[1]
        d = 0 # the coefficient of linear part should be 0 if we don't use linear part to fit the curve
    elif reg_method == 'power_linear_num_op':
        fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))+x[2]*(steps-steps[0])))*weights,2))
        res = scipy.optimize.minimize(fun, kwargs['inits'], method=kwargs['method'], bounds=kwargs['bounds'])
        a = res.x[0]
        b = res.x[1]
        d = res.x[2]
    elif reg_method == 'power_linear_in_order':
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
        'power_num_op': power regression by numerical optimization
        'power_linear_num_op': power+linear regression by numerical optimization
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
                            pure_regression_on_regression_end=2000,
                            pure_regression_on_original_begin=4000,
                            regression_transition_epochs=1000,
                            num_val_outliers=10,
                            turning_point_check_win_size=500,
                            turning_point_check_alpha=0.01,
                            # below are some parameters for power_linear_in_order method
                            max_reg_weight=10,
                            pure_simple_end_delay=1500,
                            pure_complex_begin_delay=3000
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

    # a queue to store current indices of train data and train epochs used to do the regression
    # max size is pred_win_size
    train_reg_steps = []
    train_reg_data = []
    # a queue to store current indices of train data and train epochs used to estimate variance
    # max size is noise_est_win_size
    train_noise_est_steps = []
    train_noise_est_data = []
    # a queue to store current indices of train data and train epochs used to find the turning point
    # max size is turning_point_check_win_size
    train_turning_point_check_steps = []
    train_turning_point_check_data = []
    is_turning_point = False
    finish_tuning_point_check_flag = False # if true, don't check anymore
    turning_point_epoch = None # the epoch that is regarded as turning point

    ################### simulation #####################
    global_step = 0
    train_pointer = 0
    val_pointer = 0
    while True:
        if global_step > earlyStoppingStep or val_pointer >= len(data) or val_pointer >= pure_regression_on_train_end:
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
        # train data and train epoch for finding turning point
        if len(train_turning_point_check_steps) >= turning_point_check_win_size:
            train_turning_point_check_steps.pop(0)
            train_turning_point_check_data.pop(0)    
        train_turning_point_check_steps.append(train_epochs[train_pointer])
        train_turning_point_check_data.append(train_data[train_pointer])

        # at the beginning, do the regression on training data/err
        while True:
            if val_pointer >= len(data) or epochs[val_pointer] > train_epochs[train_pointer]:
                break
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

            val_pointer += 1
        
        # start to predict periodically
        # at the beginning, do the regression on training data/err
        if train_pointer >= start_point and (train_pointer - start_point) % on_train_period == 0:
            print("Global Step: {}".format(global_step))

            # check if already reach the turning point
            # two consecutive not rejecting null hypothesis are regarded as turning point begin
            if not finish_tuning_point_check_flag:
                last_is_turning_point = is_turning_point
                is_turning_point, turning_point_p_value = \
                        one_sided_slope_test(train_turning_point_check_steps, train_turning_point_check_data, turning_point_check_alpha)
                print(is_turning_point, turning_point_p_value)
                if last_is_turning_point and is_turning_point:
                    turning_point_epoch = global_step
                    finish_tuning_point_check_flag = True
                    # pure_regression_on_regression_end = global_step
                    # pure_regression_on_original_begin = global_step + regression_transition_epochs
            #     break

            # always use power_num_op on train err
            a, b, d = _regression_method(train_reg_steps, train_reg_data, np.ones_like(train_reg_steps), 'power_num_op', 
                                        shift=0, inits=inits, method=method, bounds=bounds)
            # a, b = power_regression(train_reg_steps, train_reg_data, weights)
            # d = 0
            # use the original - power_regression_on_original to calculate variance
            # TODO: we can also use original - linear_regression_on_original to calculate variance
            # TODO: variance could also be estimated on the validation curve
            var = np.var(train_noise_est_data-(a*np.power(train_noise_est_steps,b) + d*(train_noise_est_steps-train_reg_steps[0])))
            # linear regression
            a1, b1 = linear_regression(train_reg_steps, train_reg_data)
            # var = np.var(np.array(train_noise_est_data)-a1*np.array(train_noise_est_steps)-b1)

            # if finish_tuning_point_check_flag:
            if global_step < pure_regression_on_regression_end:
                regression_on_regression_weight = 1
            elif global_step >= pure_regression_on_original_begin:
                regression_on_regression_weight = 0
            else:
                regression_on_regression_weight = (pure_regression_on_original_begin - global_step) \
                    /(pure_regression_on_original_begin - pure_regression_on_regression_end)
            # else:
            #     regression_on_regression_weight = 1
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
                                                             coeffs=[(al,bl,dl)])
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
            # train data and train epoch for finding turning point
            if len(train_turning_point_check_steps) >= turning_point_check_win_size:
                train_turning_point_check_steps.pop(0)
                train_turning_point_check_data.pop(0)    
            train_turning_point_check_steps.append(train_epochs[train_pointer])
            train_turning_point_check_data.append(train_data[train_pointer])

            train_pointer += 1

        if val_pointer >= on_val_start_point and (val_pointer - on_val_start_point) % period == 0:
            print("Global Step: {}".format(global_step))

            # check if already reach the turning point
            # two consecutive not rejecting null hypothesis are regarded as turning point begin
            if not finish_tuning_point_check_flag:
                last_is_turning_point = is_turning_point
                is_turning_point, turning_point_p_value = \
                        one_sided_slope_test(train_turning_point_check_steps, train_turning_point_check_data, turning_point_check_alpha)
                print(is_turning_point)
                if last_is_turning_point and is_turning_point:
                    turning_point_epoch = global_step
                    finish_tuning_point_check_flag = True
                    # pure_regression_on_regression_end = global_step
                    # pure_regression_on_original_begin = global_step + regression_transition_epochs

            # get the variance and calculate the val_reg_weights
            var = np.var(np.array(val_noise_est_data[:len(val_noise_est_data)-smooth_win_size//2]) - np.array(val_noise_est_smoothed_data))

            # the condition that we begin to do the "power linear in order" method is that
            ## the regression method is power_linear_in_order
            ## after the turning point
            ## after the beginning of doing regression only on val err
            ## after there are enough points after the turning point
            if regression_method == 'power_linear_in_order' and finish_tuning_point_check_flag \
                    and global_step > pure_regression_on_val_begin and global_step >= pure_simple_end_delay+turning_point_epoch:
                cur_max_reg_weight = min(max_reg_weight, max_reg_weight * (global_step - pure_simple_end_delay)/(pure_complex_begin_delay - pure_simple_end_delay))
                val_reg_weights = _weights_generator(val_reg_steps, turning_point_epoch, cur_max_reg_weight)
                linear_part_shift = turning_point_epoch
                val_regression_method = regression_method
                simple_weight = max((global_step - pure_complex_begin_delay)/(pure_simple_end_delay - pure_complex_begin_delay),0)
            else:
                val_reg_weights = np.ones(len(val_reg_steps))
                linear_part_shift = None
                # before doing "power linear in order" method, we just do 'power_linear_num_op' method
                if regression_method == 'power_linear_in_order':
                    val_regression_method = 'power_linear_num_op'
                else:
                    val_regression_method = regression_method
                simple_weight = 1


            # val_reg_weights = np.ones_like(val_reg_steps)
            a, b, d = _regression_method(val_reg_steps, val_reg_data, val_reg_weights, val_regression_method, 
                                        inits=inits, method=method, bounds=bounds, shift=linear_part_shift)
            # use the original - power_regression_on_original to calculate variance
            # var = np.var(val_noise_est_data-(a*np.power(val_noise_est_steps,b) + d*(val_noise_est_steps-val_reg_steps[0])))
            # var = np.var(np.array(val_noise_est_data[:len(val_noise_est_data)-smooth_win_size//2]) - np.array(val_noise_est_smoothed_data))
            # if finish_tuning_point_check_flag:
            #     var /= max((4-(global_step - turning_point_epoch)*(4-1)/turning_point_epoch),1)
            # else:
            #     var /= 4
            if global_step >= pure_regression_on_val_begin:
                # if finish_tuning_point_check_flag:
                if global_step < pure_regression_on_regression_end:
                    regression_on_regression_weight = 1
                elif global_step >= pure_regression_on_original_begin:
                    regression_on_regression_weight = 0
                else:
                    regression_on_regression_weight = (pure_regression_on_original_begin - global_step) \
                        /(pure_regression_on_original_begin - pure_regression_on_regression_end)
                # else:
                #     regression_on_regression_weight = 1

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
                                                                     coeffs=[(al,bl,dl)])
                else:
                    input_weight2 = (1-regression_on_regression_weight)*simple_weight
                    input_weight3 = (1-regression_on_regression_weight)*(1-simple_weight)
                    a_simple, b_simple, d_simple = _regression_method(val_reg_steps, val_reg_data, val_reg_weights, 'power_linear_num_op', 
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
                                                                     coeffs=[(al,bl,dl), (a_simple, b_simple, d_simple)])
                
            else:
                # deal with weights
                train_weight = (epochs[pure_regression_on_val_begin-1] - global_step)\
                    /(epochs[pure_regression_on_val_begin - 1] - epochs[pure_regression_on_train_end-1])
                val_weight = 1 - train_weight
                if finish_tuning_point_check_flag:
                    if global_step < pure_regression_on_regression_end:
                        regression_on_regression_weight = 1
                    elif global_step >= pure_regression_on_original_begin:
                        regression_on_regression_weight = 0
                    else:
                        regression_on_regression_weight = (pure_regression_on_original_begin - global_step) \
                            /(pure_regression_on_original_begin - pure_regression_on_regression_end)
                else:
                    regression_on_regression_weight = 1
                regression_on_original_weight = 1 - regression_on_regression_weight
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
                # always use power_num_op on train err
                al, bl, dl = _regression_method(train_reg_steps, train_reg_data, np.ones_like(train_reg_steps), 'power_num_op', 
                                                shift=0, inits=inits, method=method, bounds=bounds)
                coeffs.append((al, bl, dl))

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
                                                                 coeffs=coeffs)

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


def one_sided_slope_test(x, y, alpha):
    slope, _, _, _, std_err = scipy.stats.linregress(x,y)
    t_value = slope / std_err
    p_value = scipy.stats.t.cdf(t_value, df=len(x)-2)

    return p_value >= alpha, p_value

# The max_weight will assigned to turning_point_epoch // 2
# linear decay to 1 with epoch increase to turning_point_epoch
# or decrease to 1 
def _weights_generator(epochs, turning_point_epoch, max_weight):
    weights = []
    max_weight_epoch = turning_point_epoch // 2
    for e in epochs:
        if e < turning_point_epoch:
            weights.append(max_weight/max_weight_epoch*e)
        else:
            weights.append(max_weight/max_weight_epoch*(turning_point_epoch-e))
    return np.array(weights)