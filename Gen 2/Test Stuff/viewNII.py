import nibabel
import matplotlib.pyplot as plt
import numpy as np

struct = nibabel.load("C:/Users/mihir/Desktop/Mihir/Science Fair/Multiple Sclerosis AI/Gen 2/MRI Data/Training/MS/Severe_10.mnc")
struct_arr = struct.get_fdata() / 1500

maxval = np.amax(struct_arr)
minval = np.amin(struct_arr)

print(maxval)
print(minval)
nibabel.viewers.OrthoSlicer3D(struct_arr).show()