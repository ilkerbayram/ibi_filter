#!/usr/bin/env python
import numpy as np


def get_ibis(fname):
    cnt = 0
    with open(fname, "r") as fp:
        ibis = []
        for line in fp.readlines():
            tmp = line.split("\t")
            if tmp[3] != "N":
                cnt += 1
            try:
                ibis.append(float(tmp[2]))
            except:
                continue
    print(f"number of erroneous IBIs : {cnt}")
    return np.array(ibis)
