from utils import *

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
import math
import scipy

from kalman_filtering import oneIterPowerKalmanFilter

"""
Predict the early stopping epoch by "adding-noise" method, 
currently assume that noise is rayleigh distributed
input:
    a, b: y = ax^b
    var: the variance of unsmoothed data
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    smooth_win_size: the window size for smoothing
    num_samples: how many samples to generate when using bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit
    lower_limit: only consider the predicted epoch larger than upper_limit
    regression_on_linear_weight: at the early stage, it's better to do linear regression
        and then do a power regression on the linear line. This is the weights for such
        method. Should be in [0,1]
    **kwargs: a,b,d 
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
                                           regression_on_linear_weight=0,
                                           **kwargs
                                           ):
    num_points = int((upper_limit-lower_limit) // num_epochs_between_eval) # the number of noisy points we want to generate

    x = np.linspace(lower_limit, upper_limit, num_points)
    y = a*np.power(x,b) + d*(x-linear_bias)
    y1 = kwargs['al']*np.power(x,kwargs['bl']) + kwargs['dl']*(x-linear_bias)

    num_samples1 = int(num_samples * regression_on_linear_weight)
    num_samples0 = num_samples - num_samples1    
    stopping_epochs = []
    for i in range(num_samples0):
        #         noise = np.random.normal(0, np.sqrt(var), num_points)
        noise = np.random.rayleigh(np.sqrt((4-np.pi)/2*var), num_points) - var*np.sqrt(np.pi/2)
        z = y + noise
        
        if i == 0:
            sample = z
            smoothed_sample = smooth_by_linear_filter(z, smooth_win_size)
    
        try:
            stopping_epochs.append(early_stopping_step(z, min_delta, patience, smooth_win_size, num_epochs_between_eval) + lower_limit)
        except:
            continue

    for i in range(num_samples1):
        #         noise = np.random.normal(0, np.sqrt(var), num_points)
        noise = np.random.rayleigh(np.sqrt((4-np.pi)/2*var), num_points) - var*np.sqrt(np.pi/2)
        z = y1 + noise
        
        if i == 0:
            sample = z
            smoothed_sample = smooth_by_linear_filter(z, smooth_win_size)
    
        try:
            stopping_epochs.append(early_stopping_step(z, min_delta, patience, smooth_win_size, num_epochs_between_eval) + lower_limit)
        except:
            continue
    return np.mean(stopping_epochs), np.array(stopping_epochs), sample, smoothed_sample

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

'''
simulate early stopping prediction by regression
input:
    earlyStoppingStep: ground truth
    data: the data that used to predict the epoch
    smooth_win_size: window size for smoothing
    epochs_between_eval: number of epochs between two evals
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    pred_win_size: the window size for epoch prediction, wrt points not real epochs
    left_tail_size: the window where the data should be weighted (< 1) when doing regression,
    wrt points not real epochs
    period: the period for prediction, wrt points not real epochs
    is_power_linear: if using power + linear to do the regression
    bounds, inits, method: used in numerical optimal
    start_point: at which point we start the first prediction, wrt points not real epochs
    weights_type: the type of weights
    linear: the weights are linear from 0-1 within the left_tail_size
    equal: the weights are 1 within the left_tail_size
    num_samples: how many samples to generate when predicting the epoch by bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit when doing bootstarpping
    is_fit_smoothed: whether we do the regression on the smoothed data, or the original unsmoothed data
    noise_est_win_size: the window size used for noise estimation. If none, it's equal to the pred_win_size
output:
    preds: predicted epochs
    coeffs: regression coeffients (a,b)
    shifts: the distance that we translation the piece curve to make it close to x-axis
    sample: the sythetic data for each piece curve
    smoothed_sample: smoothing the sythetic data
    CIs: 95% confidence intervals
'''
def powerRegressionIndicator(
                            ######################## simulation ######################
                            earlyStoppingStep, 
                            #########################################################
                            data,
                            smooth_win_size,
                            epochs_between_eval,
                            min_delta, patience,
                            pred_win_size,
                            left_tail_size,
                            period,
                            is_power_linear=False,
                            bounds=None, inits=None, method='TNC',
                            start_point=100,
                            weights_type='equal',
                            num_samples=100,
                            upper_limit=2e4,
                            is_fit_smoothed=False,
                            noise_est_win_size=None,
                            pure_regression_on_linear_end=2000,
                            pure_power_regression_begin=3000,
                            ):
    original_data = data
    if not is_fit_smoothed:
        position_bias = 0
    else:
        position_bias = smooth_win_size - 1
        data = smooth_by_linear_filter(data, smooth_win_size)
    if noise_est_win_size == None:
        noise_est_win_size = pred_win_size
        
    ################### simulation #####################
    global_step = position_bias * epochs_between_eval
    ########################################
    
    # generate weights
    w1_size = pred_win_size - left_tail_size # the number of elements we assign weight 1
    if weights_type == 'linear':
        basic_weights0 = np.linspace(0,1,left_tail_size)
    elif weights_type == 'equal':
        basic_weights0 = np.ones(left_tail_size)
    basic_weights1 = np.ones(w1_size)
    basic_weights = np.concatenate((basic_weights0, basic_weights1))
    def weights_generator(length):
        if length <= w1_size:
            rst = np.ones(length)
        elif length <= pred_win_size:
            s = pred_win_size - length
            rst = basic_weights[s:]
        else:
            z = np.zeros(length - pred_win_size)
            rst = np.concatenate((z, basic_weights))

        return rst
    preds = []
    coeffs = []
    shifts = []
    samples = []
    smoothed_samples = []
    CIs = []
    vars_ = []
    predicted_steps = []
    pred = 0

    ################### simulation #####################
    for i in range(len(data)):
        global_step += epochs_between_eval
    
        ###############################
        # save time, should be deleted in practice
        if global_step > earlyStoppingStep:
            break
        ###############################
        num_evals = global_step//epochs_between_eval - position_bias

        if num_evals >= start_point and (num_evals-start_point)%period == 0:
            print(global_step)
            predicted_steps.append(global_step)
            
            # locate the smoothed points at the middle points of each window
            if num_evals < pred_win_size:
                s = 0 + position_bias // 2
            else:
                s = num_evals - pred_win_size + position_bias // 2
            e = num_evals + position_bias // 2
            
            if num_evals < noise_est_win_size:
                s_noise_est = 0
            else:
                s_noise_est = num_evals - noise_est_win_size
            
            steps = (np.arange(s,e)+1) * epochs_between_eval
            steps_noise_est = (np.arange(s_noise_est,e)+1) * epochs_between_eval
            shift = steps[-1]
            y = data[s - position_bias // 2:e - position_bias // 2]
            weights = weights_generator(len(y))
            
            if not is_power_linear:
                # regression by linear regression on log-log scale
                a, b = power_regression(steps, y, weights)
                # regression by numerical optimization
                # fun = lambda x: np.sum(np.power(y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))), 2))
                # res = scipy.optimize.minimize(fun, inits, method=method, bounds=bounds)
                # a = res.x[0]
                # b = res.x[1]
                d = 0 # the coefficient of linear part should be 0 if we don't use linear part to fit the curve
            else:
                fun = lambda x: np.sum(np.power((y-(np.exp(x[1]*np.log(steps)+np.log(x[0]))+x[2]**2*(steps-steps[0])))*weights,2))
                res = scipy.optimize.minimize(fun, inits, method=method, bounds=bounds)
                a = res.x[0]
                b = res.x[1]
                d = res.x[2]**2
            var = np.var(original_data[s_noise_est:e]-(a*np.power(steps_noise_est,b) + d*(steps_noise_est-steps[0])))

            # linear regression
            A = np.vstack([steps, np.ones(len(steps))]).T
            a1, b1 = np.linalg.lstsq(A, y, rcond=-1)[0]
            if a1 < 0:                
                linear_end = math.floor(-b1/a1)
                if b1/a1 == 0:
                    linear_end -= 1
                steps_l = np.arange(steps[0], linear_end+1, epochs_between_eval)
                yl = a1*steps_l+b1
                al, bl = power_regression(steps_l, yl, np.ones(len(steps_l)))
            
            last_pred = pred
            if a1 < 0 and global_step < pure_regression_on_linear_end:
                # var = np.var(original_data[s_noise_est:e]-a1*steps_noise_est-b1)
                pred, stopping_epochs, sample, smoothed_sample = early_stopping_prediction_adding_noise(
                                                                    a,b,d,steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=steps[-1],
                                                                     regression_on_linear_weight=1,
                                                                     al=al, bl=bl, dl=0
                                                                                     )
            elif global_step >= pure_power_regression_begin or a1 >= 0:
                pred, stopping_epochs, sample, smoothed_sample = early_stopping_prediction_adding_noise(
                                                                    a,b,d,steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=steps[-1],
                                                                     regression_on_linear_weight=0,
                                                                     al=0, bl=0, dl=0
                                                                                     )
            elif a1 < 0:
                regression_on_linear_weight = (pure_power_regression_begin - global_step)/(pure_power_regression_begin - pure_regression_on_linear_end)
                pred, stopping_epochs, sample, smoothed_sample = early_stopping_prediction_adding_noise(
                                                                    a,b,d,steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=steps[-1],
                                                                     regression_on_linear_weight=regression_on_linear_weight,
                                                                     al=al, bl=bl, dl=0
                                                                                     )
            print("Predicted Stopping epoch is {}. a = {}, b = {}, d^2={}".format(pred, a, b, d))
            
            _, CI_left, CI_right = mean_confidence_interval(stopping_epochs)
            
            preds.append(pred)
            CIs.append((CI_left, CI_right))
            coeffs.append((a,b,d))
            shifts.append(shift)
            samples.append(sample)
            smoothed_samples.append(smoothed_sample)            
            vars_.append(var)
                        
    return preds, coeffs, shifts, samples, smoothed_samples, CIs, vars_, predicted_steps



'''
simulate early stopping prediction by Kalman Filtering
input:
    earlyStoppingStep: ground truth
    data: the data that used to predict the epoch
    smooth_win_size: window size for smoothing
    num_epochs_between_eval: number of epochs between two evals
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    report_period: the period for prediction, wrt points not real epochs
    pred_win_size: the window size for epoch prediction, wrt points not real epochs
    period: in KF state transformation, how many points dose the sliding window move, wrt points not real epochs
    start_point: at which point we start the first prediction, wrt points not real epochs
    num_samples: how many samples to generate when predicting the epoch by bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit when doing bootstarpping
    noise_est_win_size: the window size used for noise estimation. If none, it's equal to the pred_win_size
output:
    epoch_preds: predicted epochs
    sample: the sythetic data for each piece curve
    smoothed_sample: smoothing the sythetic data
    CIs: 95% confidence intervals
'''
def KFIndicator(
                ######################## simulation ######################
                earlyStoppingStep, 
                #########################################################
                data, 
                smooth_win_size,
                num_epochs_between_eval, 
                min_delta, patience, 
                report_period,
                pred_win_size, 
                period,
                init_d,
                var_ud = 1e-3,
                start_point=100,                         
                num_samples=1000,
                upper_limit=2e4,
                noise_est_win_size=None): # if none, same as pred window size

    upper_lim_points = upper_limit // num_epochs_between_eval
    if noise_est_win_size == None:
        noise_est_win_size = pred_win_size
    
    def _addingNoiseBootstrapping(predicts, R, shift):
        period = R.shape[0]

        stopping_epochs = []
        for i in range(num_samples):
#             noise = np.random.multivariate_normal(np.zeros(period), R, noise_sample_size).reshape((1,-1))[0]
#             noise = np.concatenate([noise, np.random.multivariate_normal(np.zeros(residual), truncated_R)])
            noise = np.random.rayleigh(np.sqrt((4-np.pi)/2*R[0,0]), len(predicts)) - R[0,0]*np.sqrt(np.pi/2)
            measurement = predicts + noise
            smoothed_measurement = smooth_by_linear_filter(measurement, smooth_win_size)
            try:
                stopping_epochs.append(
                    early_stopping_step(measurement, min_delta, patience, smooth_win_size, num_epochs_between_eval) + shift)
            except:
                continue
        return np.mean(stopping_epochs), np.array(stopping_epochs), measurement, smoothed_measurement
    
    epoch_preds = []
    samples = []
    CIs = []
    
    KF_input_queue = []
    d_ests = []
    init_x = []
    predicted_steps = []
    KF = None
    for i, d in enumerate(data):
        global_step = (i+1)*num_epochs_between_eval
        ###############################
        # save time, should be deleted in practice
        if global_step > earlyStoppingStep:
            break
        ###############################
        
        if i < start_point:
            init_x.append(d)
                        
        if i >= pred_win_size:
            KF_input_queue.pop(0) 
        KF_input_queue.append(d)
        
        if i == start_point-1:
            KF = oneIterPowerKalmanFilter(num_epochs_between_eval,
                                          pred_win_size,
                                          period,
                                          init_x,
                                          init_d,
                                          var_ud,
                                          noise_est_win_size=noise_est_win_size)

        if i >= start_point and (i-start_point+1)%period == 0:
            KF.oneIterKF(np.array(KF_input_queue))
        
        if i >= start_point+period-1 and (i+1)%report_period == 0:
            shift = KF.q_epochs[-1]
            d_ests.append(KF.d_est)
            print('Global Step: {}'.format(global_step))
            predicted_steps.append(global_step)
            
            predicts, est_pred = KF.predictionByCurrent(int(upper_lim_points - len(KF.all_estimates)))
            epoch_pred, stopping_epochs, _, sample_measurement = _addingNoiseBootstrapping(predicts, KF.R, shift)
            _, CI_left, CI_right = mean_confidence_interval(stopping_epochs)
        
            epoch_preds.append(epoch_pred)
            CIs.append((CI_left, CI_right))
            samples.append(np.concatenate([KF.all_estimates, sample_measurement]))
            
            print('Predicted early stopping epoch: {}. d^2 = {}'.format(epoch_pred, KF.d_est**2))
        
    return epoch_preds, CIs, samples, d_ests, KF, predicted_steps


"""
Since we can't make the KF methods start too early, we combine the regression method and kf method
input:
    earlyStoppingStep: ground truth
    data: the data that used to predict the epoch
    smooth_win_size: window size for smoothing
    num_epochs_between_eval: number of epochs between two evals
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    report_period: the period for prediction, wrt points not real epochs
    pred_win_size: the window size for epoch prediction, wrt points not real epochs
    period: in KF state transformation, how many points dose the sliding window move, wrt points not real epochs
    start_point: at which point we start the first prediction, wrt points not real epochs
    KF_start_point: the point initial kf. Note that initialing kf can't predict immediately, so still using regression to report at this point
    num_samples: how many samples to generate when predicting the epoch by bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit when doing bootstarpping
    noise_est_win_size: the window size used for noise estimation. If none, it's equal to the pred_win_size
output:
    rst_preds: predicted epochs
    rst_CIs: 95% confidence intervals
"""
def mixKFIndicator(
        ######################## simulation ######################
        earlyStoppingStep, 
        #########################################################
        data, 
        smooth_win_size,
        num_epochs_between_eval, 
        min_delta, patience, 
        report_period,
        pred_win_size, 
        period,
        init_d,
        var_ud = 1e-3,
        start_point=0,
        KF_start_point=100,               
        num_samples=1000,
        upper_limit=2e5,
        noise_est_win_size=None
    ):
    
    regress_data = data[:KF_start_point+report_period]
    rst_preds = []
    rst_CIs = []
    rst_steps = []
    preds, coeffs, _, samples, smoothed_samples, CIs, _, predicted_steps = powerRegressionIndicator(
                                    earlyStoppingStep,
                                    regress_data,
                                    smooth_win_size,
                                    num_epochs_between_eval, 
                                    min_delta,
                                    patience,
                                    pred_win_size = pred_win_size,
                                    left_tail_size = 0,
                                    period = report_period,
                                    is_power_linear=False,
                                    start_point = start_point,
                                    num_samples=num_samples,
                                    upper_limit=upper_limit,
                                    is_fit_smoothed=False,
                                    noise_est_win_size = noise_est_win_size)
    rst_preds += preds
    rst_CIs += CIs
    rst_steps += predicted_steps

    print("Kalman filtering starts")
    preds, CIs, samples, d_ests, KF, predicted_steps = KFIndicator(
                                    earlyStoppingStep,
                                    data,
                                    smooth_win_size,
                                    num_epochs_between_eval, 
                                    min_delta,
                                    patience,
                                    report_period,
                                    pred_win_size = pred_win_size,
                                    period = period,
                                    init_d = init_d,
                                    var_ud = var_ud,
                                    start_point = KF_start_point,
                                    num_samples=num_samples,
                                    upper_limit=upper_limit,
                                    noise_est_win_size=noise_est_win_size)
    rst_preds += preds
    rst_CIs += CIs
    rst_steps += predicted_steps

    return rst_preds, rst_CIs, rst_steps


'''
simulate early stopping prediction by regression
input:
    earlyStoppingStep: ground truth
    data: the data that used to predict the epoch
    smooth_win_size: window size for smoothing
    epochs_between_eval: number of epochs between two evals
    min_delta, patience: no improvement (less than min_delta) for patience epoch is stopping criteria
    pred_win_size: the window size for epoch prediction, wrt points not real epochs
    left_tail_size: the window where the data should be weighted (< 1) when doing regression,
    wrt points not real epochs
    period: the period for prediction, wrt points not real epochs
    is_power_linear: if using power + linear to do the regression
    bounds, inits, method: used in numerical optimal
    start_point: at which point we start the first prediction, wrt points not real epochs
    weights_type: the type of weights
    linear: the weights are linear from 0-1 within the left_tail_size
    equal: the weights are 1 within the left_tail_size
    num_samples: how many samples to generate when predicting the epoch by bootstrapping
    upper_limit: only consider the predicted epoch smaller than upper_limit when doing bootstarpping
    is_fit_smoothed: whether we do the regression on the smoothed data, or the original unsmoothed data
    noise_est_win_size: the window size used for noise estimation. If none, it's equal to the pred_win_size
output:
    preds: predicted epochs
    coeffs: regression coeffients (a,b)
    shifts: the distance that we translation the piece curve to make it close to x-axis
    sample: the sythetic data for each piece curve
    smoothed_sample: smoothing the sythetic data
    CIs: 95% confidence intervals
'''
def method3(
                            ######################## simulation ######################
                            earlyStoppingStep, 
                            #########################################################
                            data,
                            smooth_win_size,
                            epochs_between_eval,
                            min_delta, patience,
                            pred_win_size,
                            left_tail_size,
                            period,
                            is_power_linear=False,
                            bounds=None, inits=None, method='TNC',
                            start_point=100,
                            weights_type='equal',
                            num_samples=100,
                            upper_limit=2e4,
                            is_fit_smoothed=False,
                            noise_est_win_size=None,
                            linear_start_point = 100
                            ):
    original_data = data
    if not is_fit_smoothed:
        position_bias = 0
    else:
        position_bias = smooth_win_size - 1
        data = smooth_by_linear_filter(data, smooth_win_size)
    if noise_est_win_size == None:
        noise_est_win_size = pred_win_size
        
    ################### simulation #####################
    global_step = position_bias * epochs_between_eval
    ########################################

    def _addingNoiseBootstrapping(predicts, k, theta, shift):
        stopping_epochs = []
        for i in range(num_samples):
            noise = np.random.gamma(k,theta, len(predicts)) - k*theta
            measurement = predicts + noise
            smoothed_measurement = smooth_by_linear_filter(measurement, smooth_win_size)
            try:
                stopping_epochs.append(
                    early_stopping_step(measurement, min_delta, patience, smooth_win_size, epochs_between_eval) + shift)
            except:
                continue
        return np.mean(stopping_epochs), np.array(stopping_epochs), measurement, smoothed_measurement
    
    # generate weights
    w1_size = pred_win_size - left_tail_size # the number of elements we assign weight 1
    if weights_type == 'linear':
        basic_weights0 = np.linspace(0,1,left_tail_size)
    elif weights_type == 'equal':
        basic_weights0 = np.ones(left_tail_size)
    basic_weights1 = np.ones(w1_size)
    basic_weights = np.concatenate((basic_weights0, basic_weights1))
    def weights_generator(length):
        if length <= w1_size:
            rst = np.ones(length)
        elif length <= pred_win_size:
            s = pred_win_size - length
            rst = basic_weights[s:]
        else:
            z = np.zeros(length - pred_win_size)
            rst = np.concatenate((z, basic_weights))

        return rst
    preds = []
    coeffs = []
    shifts = []
    samples = []
    smoothed_samples = []
    CIs = []
    vars_ = []
    predicted_steps = []
    pred = 0

    ################### simulation #####################
    for i in range(len(data)):
        global_step += epochs_between_eval
    
        ###############################
        # save time, should be deleted in practice
        if global_step > earlyStoppingStep:
            break
        ###############################
        num_evals = global_step//epochs_between_eval - position_bias

        if num_evals >= start_point and (num_evals-start_point)%period == 0:
            print(global_step)
            predicted_steps.append(global_step)
            
            # locate the smoothed points at the middle points of each window
            if num_evals < pred_win_size:
                s = 0 + position_bias // 2
            else:
                s = num_evals - pred_win_size + position_bias // 2
            e = num_evals + position_bias // 2
            
            if num_evals < noise_est_win_size:
                s_noise_est = 0
            else:
                s_noise_est = num_evals - noise_est_win_size
            
            steps = (np.arange(s,e)+1) * epochs_between_eval
            steps_noise_est = (np.arange(s_noise_est,e)+1) * epochs_between_eval
            shift = steps[-1]
            y = data[s - position_bias // 2:e - position_bias // 2]
            weights = weights_generator(len(y))
            
 
            a, b = power_regression(steps, y, weights)
            # prediction_start_step = 200
            if e - position_bias // 2 > linear_start_point:
                err = y[linear_start_point:] - power_function(steps[linear_start_point:], a, b)
                err_a, err_b = np.linalg.lstsq(np.vstack([steps[linear_start_point:], np.ones(len(steps[linear_start_point:]))]).T, err, rcond=None)[0]
                res = err - err_a*steps[linear_start_point:] - err_b
                flag = err_a > 0
            else:
                res = original_data[s_noise_est:e]-(a*np.power(steps_noise_est,b))
                flag = False
            skew = scipy.stats.skew(res)
            var = np.var(res)
            k = (2/skew)**2
            theta = np.sqrt(np.var(res)/k)

            d = 0 # the coefficient of linear part should be 0 if we don't use linear part to fit the curve

            if flag:
                num_points = int((upper_limit-steps[-1]) // epochs_between_eval)
                x = np.linspace(steps[-1], upper_limit, num_points)
                predicts = power_function(x,a,b) + err_a*x+err_b
                pred, stopping_epochs, sample, smoothed_sample =  _addingNoiseBootstrapping(predicts, k, theta, shift)
            else:
                pred, stopping_epochs, sample, smoothed_sample = early_stopping_prediction_adding_noise(
                                                                    a,b,d,steps[0],var,
                                                                    min_delta,patience,
                                                                     epochs_between_eval,
                                                                     smooth_win_size,
                                                                     num_samples=num_samples,
                                                                     upper_limit=upper_limit,
                                                                     lower_limit=steps[-1]
                                                                                     )
            print("Predicted Stopping epoch is {}. a = {}, b = {}, skew = {}, var = {}, k = {}, theta = {}".format(pred, a, b, skew, var,k,theta))
            
            _, CI_left, CI_right = mean_confidence_interval(stopping_epochs)
            
            preds.append(pred)
            CIs.append((CI_left, CI_right))
            coeffs.append((a,b,d))
            shifts.append(shift)
            samples.append(sample)
            smoothed_samples.append(smoothed_sample)            
            vars_.append(var)
                        
    return preds, coeffs, shifts, samples, smoothed_samples, CIs, vars_, predicted_steps,k,theta
