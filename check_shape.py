#!/usr/bin/env python3
import numpy as np
import sys

for i in sys.argv[1:]:
    f = np.load(i)
    print(i, f.shape)
