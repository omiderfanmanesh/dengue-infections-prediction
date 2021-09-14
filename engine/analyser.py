#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from data.based.based_dataset import BasedDataset
from data.preprocessing import Encoders, Scalers
from eda.based.based_analyzer import BasedAnalyzer
from eda.based.based_plots import BasedPlot


def do_analysis(dataset: BasedDataset, plots: BasedPlot, analyzer: BasedAnalyzer, encoder: Encoders, scaler: Scalers):


    analyzer.description()
    # analyzer.avg_temp_c()
    # plots.ncep_diur_temp_rng_k()
    # plots.corr(data=dataset.df)
    # plots.numerical_features_distribution()
    # plots.month()
    plots.total_cases()

