import json
from utils import *

with open('val_err.json') as f:
    val_acc = json.load(f)

min_delta = 1e-3
patience = 20
win_size = 5
num_epochs_between_eval = 20

val_err = 1 - np.array(val_acc)

earlyStoppingStep = early_stopping_step(val_err, min_delta, patience, win_size)
print(earlyStoppingStep, val_err[earlyStoppingStep], np.argmin(val_err), np.min(val_err), len(val_err))

steps = (np.arange(len(val_err)) + 1)*num_epochs_between_eval
predictedEarlyStoppingStep = early_stopping_prediction(steps, val_err, min_delta, patience)

print(predictedEarlyStoppingStep)
