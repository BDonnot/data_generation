import json
import random
import os
import pdb
import datetime
from scipy.interpolate import CubicSpline


class CorrNoise:
    def __init__(self, path="consos.json"):
        """Represent the correlated noise"""
        if not os.path.isfile(path) or not os.path.exists(path):
            msg = "Noise: the file \"{}\" is not found.".format(path)
            self.dict = {}
            raise RuntimeError(msg)
        else:
            with open(path, "r") as f:
                self.dict = json.load(f)
        self.date = {}

    def noise(self):
        """
        :return: the current noise choosen from the dictionnary
        """
        res = 1.
        kk = self.dict.keys()
        kk = [el for el in kk if el != "sources"]
        for key in kk:
            subkeys = self.dict[key].keys()
            subkey = random.sample(population=subkeys, k=1)[0]
            res *= (self.dict[key])[subkey]
            self.date[key] = subkey
        return res

    def getdate(self):
        """
        :return: return the current date for the data generated
        """
        return self.date


Noise = CorrNoise


class HourlyNoise(CorrNoise):
    def __init__(self,
                 path="loads.json",
                 month_start="jan",
                 dow_start="mon",
                 hour_start="00:00",
                 initnoise=1.0):
        CorrNoise.__init__(self, path=path)
        self.li_month = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
        self.nb_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        self.li_dow = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        self.li_hour = tuple(["{:02d}:00".format(i) for i in range(24)])
        self.nb_sample_generated = 0
        self.day_this_month = 0


        self.month_id = self._get_month_id(month_start)
        self.dow_id = self._get_dow_id(dow_start)
        self.hour_id = self._get_hour_id(hour_start)

        self.initnoise = initnoise
        self.date = {"month": self.li_month[self.month_id], "day": self.li_dow[self.dow_id],
                     "hour": self.li_hour[self.hour_id]}

    def _get_month_id(self, month):
        try:
            res = self.li_month.index(month)
        except ValueError:
            raise RuntimeError("Unknown month \"{}\" for HourlyNoise".format(month))
        return res

    def _get_dow_id(self, dow):
        try:
            res = self.li_dow.index(dow)
        except ValueError:
            raise RuntimeError("Unknown day of week \"{}\" for HourlyNoise".format(dow))
        return res

    def _get_hour_id(self, hour):
        try:
            res = self.li_hour.index(hour)
        except ValueError:
            raise RuntimeError("Unknown hour of the day \"{}\" for HourlyNoise".format(hour))
        return res

    def noise(self):
        res = self.initnoise
        res *= self.dict['month'][self.li_month[self.month_id]]
        res *= self.dict['day'][self.li_dow[self.dow_id]]
        res *= self.dict['hour'][self.li_hour[self.hour_id]]
        self.date = {"month": self.li_month[self.month_id], "day": self.li_dow[self.dow_id],
                     "hour": self.li_hour[self.hour_id]}
        self._update_ids()
        return res

    def _update_ids(self):
        self.hour_id += 1
        self.nb_sample_generated += 1
        if self.hour_id >= 24:
            self.dow_id += 1
            self.day_this_month += 1
            self.hour_id = 0

            if self.dow_id >= 7:
                # day of week
                self.dow_id = 0

            if self.day_this_month >= self.nb_days[self.month_id]:
                # next month
                self.month_id += 1
                self.day_this_month = 0
                if self.month_id == len(self.li_month):
                    self.month_id = 0


class SubHourlyNoise(CorrNoise):
    def __init__(self,
                 path="loads.json",
                 month_start="jan",
                 dow_start="mon",
                 hour_start="00:00",
                 initnoise=1.0,
                 time_resolution=datetime.timedelta(minutes=5)
                 ):
        CorrNoise.__init__(self, path=path)
        self.hourly_noises = HourlyNoise(path=path, month_start=month_start, dow_start=dow_start,
                                         hour_start=hour_start, initnoise=initnoise)
        self.h1 = datetime.timedelta(hours=1)
        self.time_resolution = time_resolution
        if int(self.h1 / time_resolution) != self.h1 / time_resolution:
            raise RuntimeError("Impossible to interpolate with resolution not comprised in an hour (eg 1min, 2min, 3, 5, 6, 12min etc.) 7min is NOT possible.")
        self.nb_timestep = self.h1 // time_resolution
        self.nb_sample_generated = 0
        self.curr_sample = 0

        self.x = [self.nb_timestep*i for i in range(3)]
        tmp = [(self.hourly_noises.date, self.hourly_noises.noise()) for _ in range(3)]
        self.y = [el for (_, el) in tmp]
        self.dates = [el for (el, _) in tmp]
        self.cs = CubicSpline(self.x, self.y)
        self.date = self.dates[0]
        self.date["min"] = str(self.time_resolution*self.curr_sample)

    def noise(self):
        res = self.cs(self.curr_sample)
        self._update_ids()
        return res

    def _update_ids(self):
        self.curr_sample += 1
        if self.curr_sample >= self.nb_timestep:
            # change of day
            self.curr_sample = 0
            self.dates = self.dates[1:]
            self.y = self.y[1:]
            tmp = self.hourly_noises.noise()
            self.y.append(tmp)
            self.dates.append(self.hourly_noises.date)
            self.cs = CubicSpline(self.x, self.y)
            self.date = self.dates[0]
        self.date["min"] = str(self.time_resolution*self.curr_sample)

