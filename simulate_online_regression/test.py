import json
from utils import *
from kalman_filtering import oneIterPowerKalmanFiltering

with open('val_err.json') as f:
    val_acc = json.load(f)

min_delta = 1e-3
patience = 20
win_size = 5

num_epochs_between_val = 20
pred_win_size = 100
period = 20

val_err = 1 - np.array(val_acc)

# earlyStoppingStep = early_stopping_step(val_err, min_delta, patience, win_size)
# print(earlyStoppingStep, val_err[earlyStoppingStep], np.argmin(val_err), np.min(val_err), len(val_err))

# steps = (np.arange(len(val_err)) + 1)*num_epochs_between_val
# predictedEarlyStoppingStep = early_stopping_prediction(steps, val_err, min_delta, patience)

# print(predictedEarlyStoppingStep)

print(pred_win_size, len(val_err[:pred_win_size]))
KF = oneIterPowerKalmanFilter(num_epochs_between_val,
                                 pred_win_size,
                                 period,
                                 val_err[:pred_win_size])


print(KF.epochs, KF.sd_est, KF.M_est)

KF._R_estimation()
print(KF.R)

KF._predicted_state_estimation()
print(KF.sd_pred, KF.sd_pred.shape)

KF._f_derivative()
print(KF.F)

KF._predicted_cor_estimation()
print(KF.M_pred)

KF._update_data_point(val_err[period:period+pred_win_size])
print(KF.x)

KF._innovation_residual()
print(KF.y_til)

KF._innovation_cor()
print(KF.T, np.linalg.matrix_rank(KF.T))

KF._kalman_gain()
print(KF.K,KF.K.shape)

KF._updated_state_estimation()
print(KF.sd_est, val_err[period:period+pred_win_size])

KF._updated_cor_estimation()
print(KF.M_est)
