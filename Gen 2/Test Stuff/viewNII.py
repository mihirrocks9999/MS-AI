# import nibabel
# import matplotlib.pyplot as plt
# import numpy as np

# struct = nibabel.load("C:/Users/mihir/Desktop/Mihir/Science Fair/Multiple Sclerosis AI/Gen 2/MRI Data/Training/MS/Severe_10.mnc")
# struct_arr = struct.get_fdata() / 1500

# maxval = np.amax(struct_arr)
# minval = np.amin(struct_arr)

# print(maxval)
# print(minval)
# nibabel.viewers.OrthoSlicer3D(struct_arr).show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
history = [0.6933, 0.6634, 0.6243, 0.5783, 0.5663, 0.5535, 0.5232, 0.5152, 0.5345, 0.7655, 0.7864, 0.7566, 0.7875, 0.7563, 0.7965, 0.7764, 0.7564, 0.7563, 0.7556, 0.7463, 0.7956, 0.8012, 0.8164, 0.8244,
0.8174, 0.7984, 0.8264, 0.8265, 0.8356, 0.8215, 0.8111, 0.8566, 0.8332, 0.8446, 0.8035, 0.8236, 0.8356, 0.8254, 0.8225, 0.8325, 0.8456, 0.8465, 0.8356, 0.8555, 0.8532, 0.8453, 0.8444, 0.8345, 0.8445, 0.8563]
history = np.array(history)
display = pd.DataFrame(history)
history1 = [0.5136, 0.5347, 0.5964, 0.5243, 0.6933, 0.6832, 0.6733, 0.6832, 0.6744, 0.6253, 0.6342, 0.6173, 0.5000, 0.5126, 0.5223, 0.6833, 0.6936, 0.7223, 0.7332, 0.6524, 0.6452, 0.6524, 0.6653, 0.6786,
0.7152, 0.7253, 0.7545, 0.6245, 0.6773, 0.7124, 0.7352, 0.7435, 0.7652, 0.7552, 0.7152, 0.5675, 0.6789, 0.7104, 0.7026, 0.6263, 0.6898, 0.6998, 0.7091, 0.7112, 0.7091, 0.7326, 0.7234, 0.7452, 0.7342, 0.7444]
history = np.array(history1)
display1 = pd.DataFrame(history1)
plt.plot(display[0], label = "train")
plt.plot(display1[0], label = "validation")
x1,x2,y1,y2 = plt.axis()  
plt.axis((x1,x2,0.4,1))
plt.ylabel("ROC-AUC")
plt.xlabel("epochs")
plt.legend()
plt.savefig("Gen 2/Code/CNN/Custom/Visualizations/lossHistory2.png")
print("Displaying Loss")