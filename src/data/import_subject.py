import os
import numpy as np
import nibabel as nib
from sklearn.neighbors import LocalOutlierFactor

img = nib.load("./data/raw/sub001/BOLD_1/NIfTI/bold.nii.gz")

# get the time series from a single voxel
ts1 = img.get_fdata()[10,10,10,:]

# LOF
LocalOutlierFactor(n_neighbors=20,
                   algorithm='auto',
                   leaf_size=30,
                   metric='minkowski',
                   p=2,
                   metric_params=None,
                   contamination='auto',
                   novelty=False,
                   n_jobs=None)