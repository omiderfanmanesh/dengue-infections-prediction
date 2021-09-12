#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

import numpy as np
import pandas as pd

from eda.based import BasedAnalyzer


class DengueInfectionAnalyzer(BasedAnalyzer):
    def __init__(self, dataset, cfg):
        super(DengueInfectionAnalyzer, self).__init__(dataset, cfg)

    def year(self):
        pass

    def week_of_year(self):
        pass

    def week_start_date(self):
        pass

    def persiann_precip_mm(self):
        table = pd.pivot_table(self.df, values='PERSIANN_precip_mm', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        # print(table.query('year == "1990" & month =="5"'))


    def ncep_air_temp_k(self):
        table = pd.pivot_table(self.df, values='NCEP_air_temp_k', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_avg_temp_k(self):
        table = pd.pivot_table(self.df, values='NCEP_avg_temp_k', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_dew_point_temp_k(self):
        table = pd.pivot_table(self.df, values='NCEP_dew_point_temp_k', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_max_air_temp_k(self):
        table = pd.pivot_table(self.df, values='NCEP_max_air_temp_k', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_min_air_temp_k(self):
        table = pd.pivot_table(self.df, values='NCEP_min_air_temp_k', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_precip_kg_per_m2(self):
        table = pd.pivot_table(self.df, values='NCEP_precip_kg_per_m2', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_humidity_percent(self):
        table = pd.pivot_table(self.df, values='NCEP_humidity_percent', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_precip_mm(self):
        table = pd.pivot_table(self.df, values='NCEP_precip_mm', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def ncep_humidity_g_per_kg(self):
        pass

    def ncep_diur_temp_rng_k(self):
        pass

    def avg_temp_c(self):
        table = pd.pivot_table(self.df, values='avg_temp_c', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)
        # avg_temp_c_is_nan = self.df[self.df['avg_temp_c'].isna()]
        # ncep_avg_temp_k = avg_temp_c_is_nan.loc[:,['NCEP_avg_temp_c','avg_temp_c']]
        # print(ncep_avg_temp_k)

    def diur_temp_rng_c(self):
        table = pd.pivot_table(self.df, values='avg_temp_c', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def max_temp_c(self):
        table = pd.pivot_table(self.df, values='avg_temp_c', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def min_temp_c(self):
        table = pd.pivot_table(self.df, values='avg_temp_c', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def precip_mm(self):
        table = pd.pivot_table(self.df, values='avg_temp_c', index=['year', 'month'],
                               columns=['city'], aggfunc=np.mean)
        print(table)

    def total_cases(self):
        pass

    def city(self):
        pass
