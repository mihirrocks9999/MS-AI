import nibabel as nib
import numpy as np

struct1 = nib.load("Gen 2/MRI Data/OG/Training/noMS/Normal_1.mnc")
struct_arr1 = struct1.get_fdata()

print(struct_arr1.shape)
print(struct_arr1)
#print(struct_arr1)
# print(struct_arr1.shape)
# x = []
# for i in range(182):
#     x.append(struct_arr1[i:i+1, :217, :181])
# print(x)
# nib.viewers.OrthoSlicer3D(x[120]).show()
nib.viewers.OrthoSlicer3D(struct_arr1).show()