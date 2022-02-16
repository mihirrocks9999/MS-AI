from createArray import importArrays
from tensorflow import keras
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.metrics import SpecificityAtSensitivity
from createTFRecord import createTF, createTestTF
from predictClass import predict

noMS_traindir = "Gen 2/MRI Data/OG/Training/noMS"
MS_traindir = "Gen 2/MRI Data/OG/Training/MS"
noMS_testdir = "Gen 2/MRI Data/OG/Test/noMS"
MS_testdir = "Gen 2/MRI Data/OG/Test/MS"
height, width, depth = 181, 217, 181
batch_size = 1 # number of samples that will be propagated through the network. Less = less memory but more inaccurate
prefetch_size = 1 # How many data sets to prefetch for gpu, increase until no speed increases
ratio = 0.7 # ratio of training to validation
printPredict = True # Print each prediction
modelnum = 4 # Select which model to use
epochs = 1 # Number of trainings

# Modify Learning Rate: https://keras.io/api/optimizers/learning_rate_schedules/, for info on what these mean.
Learnertype = "PolynomialDecay" # ExponentialDecay, PolynomialDecay, InverseTimeDecay
initial_learning_rate = 0.1 # Needed for ExponentialDecay, PolynomialDecay, InverseTimeDecay 
end_learning_rate = 0.01 # Needed for PolynomialDecay
decay_steps = 100000 # Needed for ExponentialDecay, PolynomialDecay, InverseTimeDecay
power = 0.5 # Needed for PolynomialDecay
decay_rate = 0.90 # Needed for ExponentialDecay, InverseTimeDecay
staircase = True # Needed for ExponentialDecay, InverseTimeDecay

# Callbacks
checkpoint = True # Use checkpoint callback to save best
earlystopping = True # Stop model early if no growth
monitor = "val_AUC" # What quantity to monitor for early stopping
restore_best_weights = True # Restore the best weights for each epoch
patience = 20 # Epochs with no improvement, training will be stopped.

from Models import get_model_1, get_model_2, get_model_3, get_model_4

inputstring = input("Enter C for create TF Record. Enter T for Train. Enter P for predict. : ")
inputstring = inputstring.lower()
if inputstring.find('c') != -1:
    createTF(noMS_traindir, MS_traindir, height, width, depth)
    print("Done creating Train TFRecord")
    createTestTF(noMS_testdir, MS_testdir, height, width, depth)
    print("Done creating Test TFRecord")
if inputstring.find('t') != -1:
    train_dataset, validation_dataset = importArrays(batch_size, prefetch_size, ratio)
    model = None
    if modelnum == 1:
        model = get_model_1(height, width, depth)
    if modelnum == 2:
        model = get_model_2(height, width, depth)
    if modelnum == 3:
        model = get_model_3(height, width, depth)
    if modelnum == 4:
        model = get_model_4(height, width, depth)
    model.summary()

    # Learning rate overtime
    
    if Learnertype == "ExponentialDecay":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
    elif Learnertype == "PolynomialDecay":
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, decay_steps=decay_steps, end_learning_rate=end_learning_rate, power=power)
    elif Learnertype == "InverseTimeDecay":
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    METRICS = [
    #   TruePositives(name='TruePos'),
    #   FalsePositives(name='FalsePos'),
    #   TrueNegatives(name='TrueNeg'),
    #   FalseNegatives(name='FalseNeg'), 
      BinaryAccuracy(name='Accuracy'),
      Precision(name='Precision'),
      Recall(name='Recall'),
      AUC(name='AUC'),
      SpecificityAtSensitivity(sensitivity=0.8, name='Sensitivity'),
    ]

    # Creating Model
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule) # Type of optimizer used, there are many: https://keras.io/api/optimizers/
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=METRICS,
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "Gen 2/Code/CNN/Custom/bestClassification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, verbose=1) # Stop early if no change, patience is how long it should wait

    callbacks = []
    if checkpoint & earlystopping:
        callbacks = [checkpoint_cb, early_stopping_cb]
    elif checkpoint:
        callbacks = [checkpoint_cb]
    elif earlystopping:
        callbacks = [early_stopping_cb]

    # Train the model, doing validation at the end of each epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    model.save("Gen 2/Code/CNN/Custom/currClassification.h5")
    print("Done Computing")
if inputstring.find('p') != -1:
    predict(printPredict, batch_size, prefetch_size)