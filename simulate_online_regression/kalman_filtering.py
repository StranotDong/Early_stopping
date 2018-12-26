import numpy as np
import scipy
import math
from utils import power_function, power_regression
from utils import actual_win_size

class oneIterPowerKalmanFilter:

    """
    Init function:
        input:
            num_epochs_between_val: # of epochs between two validation
                when training
            pred_win_size: # of points used for one time predition
            period: # of points skipped when sliding the window
            init_x: first several data points used for KF init.
            process_noise_var: variance of process noise, a hyperparameter
            num_extra_win_size: the # of extra points needed for estimating R
            noise_est_win_size: the win size needed for estimation the noise, none means it's equal to pred_win_size
        Note:
            Better set esitimation start position larger then the period, i.e.,
            len(init_x) > period. This is because we can't get a period X period
            measurement noise corvariance matrix if data point is less then period.
    """
    def __init__(self,
                 num_epochs_between_val,
                 pred_win_size,
                 period,
                 init_x,
                 init_d,
                 process_noise_var=1e-6,
#                 num_extra_win_size=20,
                 noise_est_win_size=None
                ):
        if len(init_x) < period:
            print("Better set estimation start position larger than period")
        
        if noise_est_win_size == None:
            self.noise_est_win_size = pred_win_size
        else:
            self.noise_est_win_size = noise_est_win_size

        self.numEpsBtwVal = num_epochs_between_val
        self.predWinSize = pred_win_size
        self.currentWinSize = min(len(init_x), self.predWinSize)
        self.pointPeriod = period # in terms of # of points
        self.epochPeriod = period * num_epochs_between_val # in terms of # of epochs

        self.weights = np.ones(self.predWinSize) # the weights when doing regression

        # KF init
        # the # of points for R estimation, if any
#        self.RWinSize = num_extra_win_size + pred_win_size

        ## variance of process noise
        self.var_ud = process_noise_var
        epochs = (np.arange(len(init_x))+1)*self.numEpsBtwVal
        a, b = power_regression(epochs, init_x, np.ones(len(init_x)))
        s_est = power_function(epochs, a, b)
        # A = np.vstack([epochs, np.ones(len(init_x))]).T
        # a, b = np.linalg.lstsq(A, init_x, rcond=-1)[0]
        # s_est = a*epochs + b

        self.epochs = epochs[len(epochs) - self.currentWinSize:]
        self.x = init_x[len(init_x) - self.currentWinSize:]
        self.s_est = s_est[len(s_est) - self.currentWinSize:]
#        self.d_est = self.x[-1] - self.s_est[-1]
        self.d_est = init_d
        ## updated state/signal estimation
        self.sd_est = np.concatenate([self.s_est, np.array([self.d_est])])
        ## updated covariance estimation
        # self.M_est = np.zeros((self.currentWinSize+1,self.currentWinSize+1))
#        dev = self.x - self.s_est
#        dev_d = np.concatenate([dev, np.zeros(1)]).reshape([-1,1])
#        self.M_est = dev_d.dot(dev_d.T)
        self.M_est = np.ones((self.currentWinSize+1,self.currentWinSize+1))

        # the array that store all x and all s for now
        self.all_original_data = np.array(init_x)
        self.all_estimates = s_est



    '''
    Estimate measurement noise corvariance matrix.
    Assume measurement noise have the same distribution in short time
    Generally, assume noise are uncorrelated in different KF iteration
    '''
    def _R_estimation(self):
        if len(self.x) <= self.noise_est_win_size:
            s = 0
        else:
            s = len(self.x) - self.noise_est_win_size
        measure_var = np.var(self.x[s:] - self.s_est[s:])
        # assume noise is Rayleigh distribution
        # measure_var = 1/(2*len(self.x))*np.sum(np.power(self.x-self.s_est,2))
        self.R = measure_var * np.eye(self.pointPeriod)
        # num_whole_points = len(self.all_original_data)
        # actual_R_win_size = min(num_whole_points, self.RWinSize)
        # start = num_whole_points - actual_R_win_size
        # measure_noise = (self.all_original_data[start:] - self.all_estimates[start:]).reshape((-1,1))
        # # print(measure_noise)

        # # if len(measure_noise) >= self.pointPeriod:
        # #     cor_mats = []
        # #     for i in range(len(measure_noise) - self.pointPeriod + 1):
        # #         cor_mats.append(measure_noise[i:i+self.pointPeriod].dot(measure_noise[i:i+self.pointPeriod].T))
        # #     self.R = np.mean(np.array(cor_mats), axis=0)
        # # else:
        # #     # in this case (may only happen at the first KF iteration), 
        # #     # we assume the noise in unsmoothed data is unrelated in different epoch
        # #     # then smoothed data should be a toeplitz matrix
        # cor_mat = measure_noise.dot(measure_noise.T)
        # auto_cors = np.array([])

        # for i in range(self.pointPeriod):
        #     if i < self.currentWinSize:
        #         auto_cors = np.concatenate([auto_cors, np.array([np.mean(cor_mat.diagonal(i))])])
        #     else:
        #         # initial data may be too few to estimate the whole R, just set associated values are 0
        #         auto_cors = np.concatenate([auto_cors, np.array([0])])
        # self.R = scipy.linalg.toeplitz(auto_cors)


    '''
    predicted_state_estimationn
    '''
    def _predicted_state_estimation(self):
        self.a, self.b = power_regression(self.epochs, self.s_est, np.ones(self.currentWinSize))
        # next q points predicted by current regression line
        # self.q_epochs = self.epochs[:self.pointPeriod] + self.currentWinSize * self.numEpsBtwVal
        self.q_epochs = np.arange(self.epochs[-1]+self.numEpsBtwVal, self.epochs[-1]+(self.pointPeriod+1)*self.numEpsBtwVal, self.numEpsBtwVal)
#        self.sq_pred = power_function(self.q_epochs, self.a, self.b) + self.d_est
        self.sq_pred = power_function(self.q_epochs, self.a, self.b) + np.tanh(np.power(self.d_est,2)*(self.q_epochs-self.epochs[-1]))
        # next state predition
        if self.currentWinSize + self.pointPeriod >= self.predWinSize:
            self.s_pred = np.concatenate([self.s_est[len(self.s_est)-(self.predWinSize-self.pointPeriod):], self.sq_pred])
        else:
            self.s_pred = np.concatenate([self.s_est, self.sq_pred])
        self.d_pred = self.d_est
        self.sd_pred = np.concatenate([self.s_pred, np.array([self.d_pred])])

    '''
    calculate the derivative of f
    '''
    def _f_derivative(self):
        log_epochs = np.log(self.epochs)
        var_log_epochs = np.var(log_epochs)
        mean_log_epochs = np.mean(log_epochs)

        coeff = (np.log(self.q_epochs) - mean_log_epochs)/var_log_epochs
        coeff = coeff.reshape((1,-1))
        dev = (log_epochs - mean_log_epochs).reshape((-1,1))
        part1 = 1 + dev.dot(coeff)

        diag = np.diag(np.reciprocal(self.s_est))
        part2 = diag.dot(part1)

        vertical_ones_w = np.ones(self.currentWinSize).reshape((-1,1))
        temp = self.sq_pred.reshape((1,-1))

        # the devirative of f_{1:q}
        dy_1q = (vertical_ones_w.dot(temp))*part2
        # vertical_ones_q = np.ones(self.pointPeriod).reshape((-1,1))
#        df_1q = np.concatenate([dy_1q.T, vertical_ones_q], axis=1)
        df_dd = 2*self.d_est*(1-np.power(np.tanh(np.power(self.d_est, 2)*(self.q_epochs-self.epochs[-1])),2))*(self.q_epochs - self.epochs[-1])
        df_1q = np.concatenate([dy_1q.T, df_dd.reshape((-1,1))], axis=1)

        # get the derivative of whole f
        if self.currentWinSize >= self.predWinSize:
            I = np.eye(self.predWinSize - self.pointPeriod)
            zeros = np.zeros((self.predWinSize-self.pointPeriod, self.pointPeriod))
            zeros2 = np.zeros((self.predWinSize-self.pointPeriod, 1))
            df_qw = np.concatenate([zeros, I, zeros2], axis=1)
            dfd = np.concatenate([np.zeros(self.predWinSize), np.array([1])]).reshape((1, -1))
        elif self.currentWinSize + self.pointPeriod < self.predWinSize:
            I = np.eye(self.currentWinSize)
            zeros = np.zeros((self.currentWinSize,1))
            df_qw = np.concatenate([I, zeros], axis=1)
            dfd = np.concatenate([np.zeros(self.currentWinSize), np.array([1])]).reshape((1, -1))
        else:
            I = np.eye(self.predWinSize - self.pointPeriod)
            zeros = np.zeros((self.predWinSize - self.pointPeriod, self.currentWinSize - self.predWinSize + self.pointPeriod+1))
            df_qw = np.concatenate([I, zeros], axis=1)
            dfd = np.concatenate([np.zeros(self.currentWinSize), np.array([1])]).reshape((1, -1))

        self.F = np.concatenate([df_qw, df_1q, dfd])

    def _predicted_cor_estimation(self):
        if self.currentWinSize + self.pointPeriod >= self.predWinSize:
            Q = np.diag(np.concatenate([np.zeros(self.predWinSize), np.array([self.var_ud])]))
        else:
            Q = np.diag(np.concatenate([np.zeros(self.currentWinSize+self.pointPeriod), np.array([self.var_ud])]))
        self.M_pred = self.F.dot(self.M_est).dot(self.F.T) + Q


    """
    get new true data
    """
    def _update_data_point(self,x):
        self.x = x
        self.previousWinSize = self.currentWinSize
        self.currentWinSize = len(self.x)

        # update the measurement matrix
        if self.currentWinSize >= self.predWinSize:
            self.H = np.concatenate([
                np.zeros((self.pointPeriod, self.predWinSize-self.pointPeriod)),
                np.eye(self.pointPeriod),
                np.zeros((self.pointPeriod,1))],
                axis=1)
        else:
            self.H = np.concatenate([
                np.zeros((self.pointPeriod, self.previousWinSize)),
                np.eye(self.pointPeriod),
                np.zeros((self.pointPeriod,1))],
                axis=1)

        # update new input data
        self.all_original_data = np.concatenate([self.all_original_data, self.x[self.currentWinSize-self.pointPeriod:]])
        if self.previousWinSize >= self.predWinSize:
            self.epochs += self.epochPeriod
        elif self.currentWinSize < self.predWinSize:
            self.epochs = (np.arange(len(x))+1)*self.numEpsBtwVal
        else:
            self.epochs = (np.arange(self.currentWinSize-self.predWinSize, self.currentWinSize)+1) * self.numEpsBtwVal

    '''
    innovation residual
    '''
    def _innovation_residual(self):
        self.y_til = self.x[self.currentWinSize-self.pointPeriod:] - self.H.dot(self.sd_pred)

    '''
    innovation corvariance
    '''
    def _innovation_cor(self):
        self.T = self.H.dot(self.M_pred).dot(self.H.T) + self.R

    '''
    Kalman Gain
    '''
    def _kalman_gain(self):
        self.K = self.M_pred.dot(self.H.T).dot(np.linalg.inv(self.T))

    '''
    updated state estimation
    '''
    def _updated_state_estimation(self):
        self.sd_est = self.sd_pred.reshape((-1,1)) + self.K.dot(self.y_til.reshape(-1,1))
        self.sd_est = self.sd_est.reshape(-1)
        self.s_est = self.sd_est[:-1]
        # self.sq_est = self.s_est[self.predWinSize-self.pointPeriod:]
        self.all_estimates = np.concatenate([self.all_estimates, self.s_est[self.currentWinSize-self.pointPeriod:]])
        self.d_est = self.sd_est[-1]

    '''
    updated corvariance estimation
    '''
    def _updated_cor_estimation(self):
        self.M_est = (np.eye(self.K.shape[0]) - self.K.dot(self.H)).dot(self.M_pred)

    """
    Kalman Filtering For one interation
    input: new real data point
    """
    def oneIterKF(self, x):
        self._R_estimation()
        self._predicted_state_estimation()
        self._f_derivative()
        self._predicted_cor_estimation()
        self._update_data_point(x)
        self._innovation_residual()
        self._innovation_cor()
        self._kalman_gain()
        self._updated_state_estimation()
        self._updated_cor_estimation()


    """
    measurement prediction by current state
    input:
        num: # of predicted points
    """
    def predictionByCurrent(self, num):
        s_queue = list(self.s_est)
        epoch_queue = list(self.epochs)
        predicts = []
        a, b = power_regression(np.array(epoch_queue), np.array(s_queue), np.ones(len(s_queue)))
        for i in range(1,num):
            # if (i-1) % self.pointPeriod == 0:
            #     a, b = power_regression(np.array(epoch_queue), np.array(s_queue), np.ones(len(s_queue)))
            epoch = self.epochs[-1] + i * self.numEpsBtwVal
            if len(epoch_queue) >= self.predWinSize:
                epoch_queue.pop(0)
            epoch_queue.append(epoch)

#            predict = power_function(epoch, a, b) + self.d_est
            predict = power_function(epoch, a, b) + np.tanh(np.power(self.d_est,2)*(epoch-self.epochs[-1]))
            predicts.append(predict)
            if len(s_queue) >= self.predWinSize:
                s_queue.pop(0)
            s_queue.append(predict)

        return np.array(predicts), np.concatenate([self.all_estimates, np.array(predicts)])

