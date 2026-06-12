#!/usr/bin/env python

import h5py

def store_sample(target_file, sample_name, hs_image, hs_label):
    assert hs_image is not None and hs_label is not None

    with h5py.File(target_file, "a") as hdf_file:
        if sample_name in hdf_file:
            print(f"{sample_name} already exists ... Skip!")
            return

        group = hdf_file.create_group(sample_name)
        group.create_dataset("image", data=hs_image)
        group.create_dataset("labels", data=hs_label)
