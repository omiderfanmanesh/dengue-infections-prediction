#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.


import math

import numpy as np
import pandas as pd

from data.based.based_dataset import BasedDataset
from data.based.file_types import FileTypes


class DengueInfection(BasedDataset):
    def __init__(self, cfg, development):
        super(DengueInfection, self).__init__(cfg=cfg, dataset_type=FileTypes.TSV, development=development)


        self.city()

        self.extract_month()
        self.week_start_date()

        self.persiann_precip_mm()

        self.ncep_avg_temp_k()
        self.ncep_diur_temp_rng_k()
        self.ncep_max_air_temp_k()
        self.ncep_min_air_temp_k()
        self.ncep_air_temp_k()
        self.ncep_dew_point_temp_k()

        self.avg_temp_c()
        self.diur_temp_rng_c()
        self.max_temp_c()
        self.min_temp_c()
        self.precip_mm()


    def fill_nan(self, col):
        table = pd.pivot_table(self.df, values=col, index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)

        self.df[col + '_no_nans'] = self.df[col]

        for index, row in self.df.iterrows():
            if math.isnan(row[col]):
                query = table.query(f'year == "{row["year"]}" & month =="{row["month"]}"').reset_index()
                city = row['city']
                value = query[city]

                if value.empty:
                    value = self.df.loc[self.df['year'] == row["year"]][col].mean()
                    self.df.loc[index, [col + '_no_nans']] = value
                    continue
                self.df.loc[index, [col + '_no_nans']] = value[0]

    def extract_month(self):
        self.df['week_start_date'] = pd.to_datetime(self.df['week_start_date'])
        self.df['month'] = self.df['week_start_date'].dt.month

    def kelvin_to_celsius(self, kelvin):
        if kelvin is None:
            return kelvin
        return kelvin - 273.15



    def year(self):
        pass

    def week_of_year(self):
        pass

    def week_start_date(self):
        pass

    def persiann_precip_mm(self):
        self.fill_nan(col='PERSIANN_precip_mm')

    def ncep_air_temp_k(self):
        self.df['NCEP_air_temp_c'] = self.df['NCEP_air_temp_k'].apply(lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_air_temp_c')

    def ncep_avg_temp_k(self):
        self.df['NCEP_avg_temp_c'] = self.df['NCEP_avg_temp_k'].apply(lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_avg_temp_c')

    def ncep_dew_point_temp_k(self):
        """
        dew point temperature in Kelvin degrees measured by NCEP CFSR;
        :rtype: object
        """
        self.df['NCEP_dew_point_temp_c'] = self.df['NCEP_dew_point_temp_k'].apply(
            lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_dew_point_temp_c')

    def ncep_max_air_temp_k(self):
        self.df['NCEP_max_air_temp_c'] = self.df['NCEP_max_air_temp_k'].apply(
            lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_max_air_temp_c')

    def ncep_min_air_temp_k(self):
        self.df['NCEP_min_air_temp_c'] = self.df['NCEP_min_air_temp_k'].apply(
            lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_min_air_temp_c')

    def ncep_precip_kg_per_m2(self):
        self.fill_nan(col='NCEP_precip_kg_per_m2')

    def ncep_humidity_percent(self):
        self.fill_nan(col='NCEP_humidity_percent')

    def ncep_precip_mm(self):
        self.fill_nan(col='NCEP_precip_mm')

    def ncep_humidity_g_per_kg(self):
        self.fill_nan(col='NCEP_humidity_g_per_kg')

    def ncep_diur_temp_rng_k(self):
        self.df['NCEP_diur_temp_rng_c'] = self.df['NCEP_diur_temp_rng_k'].apply(
            lambda k: self.kelvin_to_celsius(kelvin=k))
        self.fill_nan(col='NCEP_diur_temp_rng_c')

    def avg_temp_c(self):
        self.fill_nan(col='avg_temp_c')

    def diur_temp_rng_c(self):
        self.fill_nan(col='diur_temp_rng_c')

    def max_temp_c(self):
        self.fill_nan(col='max_temp_c')

    def min_temp_c(self):
        self.fill_nan(col='min_temp_c')

    def precip_mm(self):
        self.fill_nan(col='precip_mm')

    def total_cases(self):
        pass

    def city(self):
        self.df = self.df[self.df['city'] != 'sj']
