#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from data.dengue_infection import DengueInfection


def load(cfg,development=True):
    den = DengueInfection(cfg=cfg,development=development)
    return den
