#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from eda.based import BasedPlot


class DengueInfectionPlots(BasedPlot):
    def __init__(self, cfg, dataset):
        super(DengueInfectionPlots, self).__init__(dataset=dataset, cfg=cfg)

    def year(self):
        self.rel(x='year',y='total_cases')
        self.box(col='year',title='year distribution')

    def week_of_year(self):
        self.rel(x='weekofyear', y='total_cases')
        self.box(col='weekofyear', title='weekofyear distribution')

    def week_start_date(self):
        self.rel(x='week_start_date', y='total_cases')
        self.box(col='week_start_date', title='week_start_date distribution')

    def persiann_precip_mm(self):
        self.rel(x='PERSIANN_precip_mm', y='total_cases')
        self.box(col='PERSIANN_precip_mm', title='PERSIANN_precip_mm distribution')

    def ncep_air_temp_k(self):
        self.rel(x='NCEP_air_temp_k', y='total_cases')
        self.box(col='NCEP_air_temp_k', title='NCEP_air_temp_k distribution')

    def ncep_avg_temp_k(self):
        self.rel(x='NCEP_avg_temp_k', y='total_cases')
        self.box(col='NCEP_avg_temp_k', title='NCEP_avg_temp_k distribution')

    def ncep_dew_point_temp_k(self):
        self.rel(x='NCEP_dew_point_temp_k', y='total_cases')
        self.box(col='NCEP_dew_point_temp_k', title='NCEP_dew_point_temp_k distribution')

    def ncep_max_air_temp_k(self):
        self.rel(x='NCEP_max_air_temp_k', y='total_cases')
        self.box(col='NCEP_max_air_temp_k', title='NCEP_max_air_temp_k distribution')

    def ncep_min_air_temp_k(self):
        self.rel(x='NCEP_min_air_temp_k', y='total_cases')
        self.box(col='NCEP_min_air_temp_k', title='NCEP_min_air_temp_k distribution')

    def ncep_precip_kg_per_m2(self):
        self.rel(x='NCEP_precip_kg_per_m2', y='total_cases')
        self.box(col='NCEP_precip_kg_per_m2', title='NCEP_precip_kg_per_m2 distribution')

    def ncep_humidity_percent(self):
        self.rel(x='NCEP_humidity_percent', y='total_cases')
        self.box(col='NCEP_humidity_percent', title='NCEP_humidity_percent distribution')

    def ncep_precip_mm(self):
        self.rel(x='NCEP_precip_mm', y='total_cases')
        self.box(col='NCEP_precip_mm', title='NCEP_precip_mm distribution')

    def ncep_humidity_g_per_kg(self):
        self.rel(x='NCEP_humidity_g_per_kg', y='total_cases')
        self.box(col='NCEP_humidity_g_per_kg', title='NCEP_humidity_g_per_kg distribution')

    def ncep_diur_temp_rng_k(self):
        self.rel(x='NCEP_diur_temp_rng_k', y='total_cases')
        self.box(col='NCEP_diur_temp_rng_k', title='NCEP_diur_temp_rng_k distribution')

    def avg_temp_c(self):
        self.rel(x='avg_temp_c', y='total_cases')
        self.box(col='avg_temp_c', title='avg_temp_c distribution')

    def diur_temp_rng_c(self):
        self.rel(x='diur_temp_rng_c', y='total_cases')
        self.box(col='diur_temp_rng_c', title='diur_temp_rng_c distribution')

    def max_temp_c(self):
        self.rel(x='max_temp_c', y='total_cases')
        self.box(col='max_temp_c', title='max_temp_c distribution')

    def min_temp_c(self):
        self.rel(x='min_temp_c', y='total_cases')
        self.box(col='min_temp_c', title='min_temp_c distribution')

    def precip_mm(self):
        self.rel(x='precip_mm', y='total_cases')
        self.box(col='precip_mm', title='precip_mm distribution')

    def total_cases(self):
        self.rel(x='total_cases', y='total_cases')
        self.box(col='total_cases', title='total_cases distribution')

    def city(self):
        self.bar(x='city',y='total_cases')
