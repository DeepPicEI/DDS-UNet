import ants
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


fp_list = [os.path.join(r,f[i]) for r,d,f in os.walk("./MICCAI_BraTS_2019_Data_Training/") if d==[]
                                for i in range(len(f))]

fp_list = sorted(fp_list)
print(fp_list)


# TODO: optimize the code
fp_list_sep = []
fp_imgs_temp = []
fp_mask_temp = 0
for i in range(len(fp_list)):
    try:
        if fp_list[i].split("/")[-2] == fp_list[i + 1].split("/")[-2]:
            if fp_list[i].split("_")[-1].split(".")[0] in ["flair", "t1", "t1ce", "t2"]:
                fp_imgs_temp.append(fp_list[i])
            if fp_list[i].split("_")[-1].split(".")[0] == "seg":
                fp_mask_temp = fp_list[i]
        else:
            fp_imgs_temp.append(fp_list[i])
            fp_list_sep.append((fp_imgs_temp, fp_mask_temp))
            fp_imgs_temp = []

    except IndexError as e:
        pass

# print(fp_list_sep)

fp_list_sep_gen = (([ants.image_read(fp) for fp in fp_list_sep[i][0]], ants.image_read(fp_list_sep[i][1])) for i in range(len(fp_list_sep)))
print(fp_list_sep_gen)
imgs, mask = next(fp_list_sep_gen)

ants.plot(imgs[0], overlay=mask) # 0: flair, 1: t1, 2: t1_ce, 3:t2
ants.imshow()