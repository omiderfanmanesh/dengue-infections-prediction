#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


import warnings

from configs import cfg
from data import load
from data.preprocessing import Encoders, Scalers
from eda.dengue_infection_analyser import DengueInfectionAnalyzer
from eda.dengue_infection_plot import DengueInfectionPlots
from engine.analyser import do_analysis


import numpy as np

np.seterr(all='warn')
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    den = load(cfg)
    den.load_dataset()
    den.drop_cols()

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)

    analyzer = DengueInfectionAnalyzer(dataset=den, cfg=cfg)
    plots = DengueInfectionPlots(dataset=den, cfg=cfg)

    do_analysis(dataset=den, analyzer=analyzer, plots=plots, encoder=encoder, scaler=scaler)


if __name__ == '__main__':
    main()
