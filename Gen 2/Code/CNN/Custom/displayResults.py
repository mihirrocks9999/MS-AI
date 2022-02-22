import pandas as pd
import matplotlib.pyplot as plt

def lossCurve(history):
    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")