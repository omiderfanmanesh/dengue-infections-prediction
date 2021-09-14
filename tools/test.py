import copy

import joblib

from configs import cfg
from data.loader import load
from data.preprocessing import Encoders, Scalers, PCA
from model.based import BasedModel


def WriteOnFile(name, y_eval):
    f = open(name, "w")
    f.write("Id,Predicted\n")
    for index, i in enumerate(y_eval):
        f.write(f"{index},{i}\n")
    f.close

def main():
    model = BasedModel(cfg=cfg)
    model.load()
    den = load(cfg, development=False)  # create dataset object instance
    den.load_dataset()  # load data from csv file
    den.drop_cols()  # drop columns

    encoder = Encoders(cdg=cfg)  # initialize Encoder object
    scaler = Scalers(cfg=cfg)  # initialize scaler object



    if cfg.BASIC.TRANSFORMATION:
        den.df = den.transformation(copy.deepcopy(den.df))

    enc = joblib.load('../output/city_binary_encoding_4.joblib')  # initialize Encoder object
    scl = joblib.load('../output/min_max_scaler_1.joblib')  # initialize scaler object

    # den.df = encoder.encode_by_enc(enc=enc, data=den.df)
    den.df = scaler.scale_by_scl(scl=scl, data=den.df)

    pca = None
    if cfg.BASIC.PCA:  # PCA object will be initialized if you set pca = True in configs file
        pca = PCA(cfg=cfg)
        pca.pca = joblib.load('../output/pca.joblib')
        den.df = pca.pca.transform(den.df)

    # model.model.get_booster().feature_names =None

    pred = model.prediction(den.df)
    WriteOnFile('../output/submission.csv',pred)

if __name__ == '__main__':
    main()
