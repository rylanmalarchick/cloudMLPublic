# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:09:23 2025

@author: ppantina
"""
import datetime

import numpy as np


def cplTimeConvert(cTimePre, cFileName):
    ##Convert CPL time to Epoch Time
    cTime = np.zeros_like(cTimePre)  ##Julian days
    YY = cFileName.split(".")[0][-2:]  ##read from file
    JJ = np.zeros_like(cTimePre)
    JJfrac = np.zeros_like(cTimePre)
    HH = np.zeros_like(cTimePre)
    MM = np.zeros_like(cTimePre)
    SS = np.zeros_like(cTimePre)
    SSS = np.zeros_like(cTimePre)

    previous_frac = 0  ##track day rollovers
    JJpush = datetime.timedelta(seconds=0)  ##rollover 0 for now

    for i, j in enumerate(cTimePre):  ##cTimePre is Julian day chaos.
        # Store Julian day/frac
        JJ[i] = int(j)
        JJfrac[i] = j - np.floor(j)

        ##JJfrac increment should be positive. If negative, manually add a day.
        if (JJfrac[i] - previous_frac) < -0.2:
            JJpush = datetime.timedelta(seconds=86400)

        ##HH, MM, SS
        HH[i] = JJfrac[i] * 24
        MM[i] = (HH[i] - np.floor(HH[i])) * 60
        SS[i] = (MM[i] - np.floor(MM[i])) * 60

        ##Seconds cannot be 60 in datetime. Store the mod and manually add a minute if so.
        SSS[i] = int(round(SS[i])) % 60
        if round(SS[i]) > 59:
            MMpush = datetime.timedelta(seconds=60)
        else:
            MMpush = datetime.timedelta(seconds=0)

        # Store the datetime, then convert to Epoch
        # Add day/second pushes if needed
        dt = (
            datetime.datetime.strptime(
                YY
                + str(int(JJ[i])).zfill(3)
                + str(int(HH[i])).zfill(2)
                + str(int(MM[i])).zfill(2)
                + str(int(SSS[i])).zfill(2),
                "%y%j%H%M%S",
            ).replace(tzinfo=datetime.timezone.utc)
            + JJpush
            + MMpush
        )
        dt = dt.timestamp()

        # Save, then update the counter.
        cTime[i] = dt
        previous_frac = JJfrac[i]

    return cTime
