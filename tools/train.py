#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from configs import cfg
from data.loader import load
from data.preprocessing import Encoders, Scalers, PCA
from engine.trainer import do_fine_tune, do_cross_val, do_train
from model import DecisionTree, LogisticRegression, SVM, RandomForest
from model.based import Model
from model.based.tuning_mode import TuningMode
from utils import RuntimeMode


def main():
    den = load(cfg)  # create dataset object instance
    den.load_dataset()  # load data from csv file
    den.drop_cols()  # drop columns

    model_selection = cfg.BASIC.MODEL  # select the model
    if model_selection == Model.SVM:
        model = SVM(cfg=cfg)
    elif model_selection == Model.DECISION_TREE:
        model = DecisionTree(cfg=cfg)
    elif model_selection == Model.RANDOM_FOREST:
        model = RandomForest(cfg=cfg)
    elif model_selection == Model.LOGISTIC_REGRESSION:
        model = LogisticRegression(cfg=cfg)

    encoder = Encoders(cdg=cfg)  # initialize Encoder object
    scaler = Scalers(cfg=cfg)  # initialize scaler object
    pca = None
    if cfg.BASIC.PCA:  # PCA object will be initialized if you set pca = True in configs file
        pca = PCA(cfg=cfg)

    runtime_mode = cfg.BASIC.RUNTIME_MODE  # mode that you want to run this code
    if runtime_mode == RuntimeMode.TRAIN:
        do_train(cfg=cfg, dataset=den, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.CROSS_VAL:
        do_cross_val(cfg=cfg, dataset=den, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.TUNING:
        do_fine_tune(cfg=cfg, dataset=den, model=model, encoder=encoder, scaler=scaler,
                     method=TuningMode.GRID_SEARCH)
    if runtime_mode == RuntimeMode.FEATURE_IMPORTANCE:
        do_train(cfg=cfg, dataset=den, model=model, encoder=encoder, scaler=scaler, pca=pca, feature_importance=True)


if __name__ == '__main__':
    main()
