import nibabel
import matplotlib.pyplot as plt

struct = nibabel.load("C:/Users/mihir/Desktop/Mihir/Science Fair/Multiple Sclerosis AI/Gen 2/Test Stuff/brainweb.mnc")
struct_arr = struct.get_fdata()

nibabel.viewers.OrthoSlicer3D(struct_arr).show()