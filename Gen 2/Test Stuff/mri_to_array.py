import nibabel as nib
import numpy as np

struct1 = nib.load("Gen 2/MRI Data/OG/Training/MS/Severe_1.mnc")
struct_arr1 = struct1.get_fdata()

print(struct_arr1.shape)
print(struct_arr1)
nib.viewers.OrthoSlicer3D(struct_arr1).show()

struct_arr1 = struct_arr1**3
struct_arr1 /= 1000

print("bussy")
print(struct_arr1.shape)
print(struct_arr1)
nib.viewers.OrthoSlicer3D(struct_arr1).show()