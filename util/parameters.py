NPARTICLE = 100
NITERATION = 50
DIM = 15
RANGE = (0, 1)
VMAX = (RANGE[1]-RANGE[0])/2

C1 = 2.05#
C2 = 2.05
WRANGE = (.4, .9)
W = [WRANGE[1] - (WRANGE[1] - WRANGE[0]) * i / NITERATION for i in range(NITERATION)]