import os
import time
import datetime
# import time
import pdb

import itertools
import bisect
import math

import random
import numpy as np

import multiprocessing as mp
import json
import grid2op

from .CorrelatedNoises import CorrNoise

from grid2op.Action import HelperAction
from grid2op.GameRules import GameRules

from .UncorrNoise import UncorrNoiseConso


class Compute:
    """
    A class to compute 'size' load-flow
    changing injection-plan.
    This is thread safe : the powergrid network is copied
    """
    def __init__(self, size, net, transit=True, withReact=False,
                 param_QP_distrib="./QP_ratio_distrb.json",
                 maxratiodiv=0.1,
                 sigmaConso=0.05,
                 sigmaProd=0.05,
                 corrnoise=None,
                 uncorrNoiseConso=None,
                 dc = False, disco_prod_winter=0.05, disco_prod_summer=0.1,
                 with_powerflow=True):
        """

        Parameters
        ----------
        size
        net: ``grid2op.Backend.Backend``
        transit
        withReact
        maxratiodiv
        sigmaConso
        sigmaProd
        corrnoise
        uncorrNoiseConso
        dc
        disco_prod_winter
        disco_prod_summer
        """

        if corrnoise is None:
            corrnoise = CorrNoise()
        if uncorrNoiseConso is None:
            uncorrNoiseConso = UncorrNoiseConso()

        self.size = size
        self.transit = transit

        self.net = net.copy()
        self.DCnet = net.copy()

        self.dc = dc

        self.conso_names = self.net.name_loads
        self.prod_names = self.net.name_prods
        self.quad_names = self.net.name_lines

        self.n_conso = len(self.conso_names)
        self.n_prod = len(self.prod_names)
        self.n_quads = len(self.quad_names)

        load_p, *_ = self.net.loads_info()
        prod_p, *_ = self.net.generators_info()

        self.inj_plan = np.concatenate((load_p , prod_p))
        self.ref_prodP = 1.5*prod_p
        self.ref_loadsP = load_p

        if withReact:
            self.sizein = 2*(self.n_conso+self.n_prod)
        else:
            self.sizein = self.n_conso + self.n_prod

        self.injs = np.ndarray((self.size, self.sizein))
        self.transits = np.ndarray((self.size, self.n_quads))

        # prepare storing of everything upfront
        # inputs of the lf
        self.inj_act = np.ndarray((self.size, self.n_conso))
        self.inj_react = np.ndarray((self.size, self.n_conso))
        self.prod_plan = np.ndarray((self.size, self.n_prod))
        self.prod_v = np.ndarray((self.size, self.n_prod))

        #outputs of the lf
        self.prod_act = np.ndarray((self.size, self.n_prod))
        self.conso_v = np.ndarray((self.size, self.n_conso))
        self.prod_react = np.ndarray((self.size, self.n_prod))
        self.transit_a = np.ndarray((self.size, self.n_quads))
        self.transit_a_ext = np.ndarray((self.size, self.n_quads))
        self.transit_MW = np.ndarray((self.size, self.n_quads))
        self.transit_MW_ext = np.ndarray((self.size, self.n_quads))

        self.transit_a_dc = np.ndarray((self.size, self.n_quads))
        self.transit_a_ext_dc = np.ndarray((self.size, self.n_quads))
        self.transit_MW_dc = np.ndarray((self.size, self.n_quads))
        self.transit_MW_ext_dc = np.ndarray((self.size, self.n_quads))

        # other stuff for sampling lf inputs
        self.corrnoise = corrnoise  # noise that represents correlated noise, more or less the time of the year
        self.uncorrNoiseConso = uncorrNoiseConso

        self.muconso = np.mean(load_p)
        # self.sigmaConso = 0.01*self.muconso #1% noise relative to average conso
        self.sigmaConso = sigmaConso  # have the uncorrelated noise vary stuff of around 5%
        if self.uncorrNoiseConso.need_init():
            self.uncorrNoiseConso.init(self.muconso, self.sigmaConso)

        # print("creating a \"Compute\" instance wiht sigmaConso: {}".format(sigmaConso))
        self.sigmaProd = sigmaProd  # have the uncorrelated noise vary stuff of around 5%
        # self.muprod = np.mean(self.ref_loadsP)
        self.muprod = np.mean(self.ref_prodP)
        self.losses = 1.02  # loss of 5% total consumption
        self.withReact = withReact
        self.densityFactorQP = {}
        self.cumdist = []
        if withReact:
            dict = {}
            with open(param_QP_distrib) as f:
                dict = json.load(f)
            self.densityFactorQP = dict
            self.cumdist = list(np.cumsum(dict['proba']))
        self.nbdiv = 0
        self.lftime = 0  # ellapsed time computing lf, in seconds
        self.nblf = 0

        self.e = math.exp(1)  # the constant value "e" (exp(1))
        self.maxratiodiv = maxratiodiv
        self.disco_prod_winter = disco_prod_winter
        self.disco_prod_summer = disco_prod_summer

        game_rules = GameRules()
        self.action_space = HelperAction(name_prod=self.prod_names,
                                         name_load=self.conso_names,
                                         name_line=self.quad_names,
                                         subs_info=self.net.subs_elements,
                                         load_to_subid=self.net.load_to_subid,
                                         gen_to_subid=self.net.gen_to_subid,
                                         lines_or_to_subid=self.net.lines_or_to_subid,
                                         lines_ex_to_subid=self.net.lines_ex_to_subid,
                                         load_to_sub_pos=self.net.load_to_sub_pos,
                                         gen_to_sub_pos=self.net.gen_to_sub_pos,
                                         lines_or_to_sub_pos=self.net.lines_or_to_sub_pos,
                                         lines_ex_to_sub_pos=self.net.lines_ex_to_sub_pos,
                                         load_pos_topo_vect=self.net.load_pos_topo_vect,
                                         gen_pos_topo_vect=self.net.gen_pos_topo_vect,
                                         lines_or_pos_topo_vect=self.net.lines_or_pos_topo_vect,
                                         lines_ex_pos_topo_vect=self.net.lines_ex_pos_topo_vect,
                                         game_rules=game_rules)
        self.with_powerflow = with_powerflow

    def recoeverything(self, el, avg):
        """reconnect everything
        If something was disconnected, its new value is drawn to be +- 50% of the avg value"""
        res = el
        if res == 0.:
            res = avg*random.lognormvariate(mu=1., sigma=0.5)/self.e
        return res

    def getprobadecoperiod(self, period):
        res = 0.
        if period["month"] in ["jan", "fev", "mar", "dec", "feb"]:
            res = self.disco_prod_winter
        else:
            res = self.disco_prod_summer
        return res

    def disconnectprod(self, prodval, period):
        """disconnect some production depending on the 'time of the year'
        It supposely reflect the fact that production maintenance happens preferably during summer months.
        So the probability for a production to be 0 depends on the time of the year"""
        recodate = 1.-self.getprobadecoperiod(period=period)
        res = random.uniform(0, 1) >= recodate
        return res

    def modifyprod(self, el, currsumprod, sumconso):
        """
        Sample one production. Equivalent to the uncorrelated noise concept for the loads, but here for the productions
        :param el: the current value of the production to sample
        :param currsumprod: the total production for this run (before being simulated)
        :param sumconso: the total loads, which have already be sampled
        :return: the new production value
        """
        fact = 1.0
        assert currsumprod > 0, "Error of dispatching production"
        ratio = sumconso/currsumprod # targetted ratio, if no production "uncorrelated noise"
        fact = ratio*random.lognormvariate(mu=1., sigma=self.sigmaProd)/self.e
        return el*fact

    def sampleqpratio(self):
        """sample from dict['qp_ratio'],
        with coming from dict['probas'] (factored in self.cumdist)
        """
        # return 0.0
        return self.densityFactorQP['qp_ratio'][bisect.bisect(self.cumdist, random.random() * self.cumdist[-1])]

    def setconsoplanRe(self, actconso):
        """set the new conso plan for reactive power"""
        return [el*self.sampleqpratio() for el in actconso]

    def getconsoplan(self, corrNoise):
        """
        Generate a new consumption planning, only active and reactive power
        :param corrNoise: the correlated noise
        :return: both active and reactive new loads dispatch
        """

        # thisConsoPlan: only active power
        thisConsoPlan = [el for el in self.inj_plan[:self.n_conso]]
        thisConsoPlan = self.uncorrNoiseConso(thisConsoPlan)
        thisConsoPlan = [el*corrNoise for el in thisConsoPlan]
        thisConsoPlanRe = None
        if self.withReact:
            # generate with reactive power if needs to
            thisConsoPlanRe = self.setconsoplanRe(thisConsoPlan)
        return thisConsoPlan, thisConsoPlanRe

    def getprodplan(self, period, consoplan):
        """
        Generate a new production plan, based on the period of the year (related to the correlated noise)
        and the consumptions dispatching.
        Deals only with active productions.
        :param period:
        :param consoplan:
        :return: the generated production plan
        """
        sumConso = np.sum(consoplan)

        thisProdPlan = self.ref_prodP
        thisProdPlan = [self.recoeverything(el, self.muprod) for el in thisProdPlan]

        sumProd = 0
        while sumProd == 0:
            tmp = [el*(1.-self.disconnectprod(el, period)) for el in thisProdPlan]
            # print("tmp {}".format(tmp))
            sumProd = np.sum(tmp)
        thisProdPlan = tmp
        thisProdPlan = [self.modifyprod(el, currsumprod=sumProd, sumconso=sumConso) for el in thisProdPlan]
        sumProd = np.sum(thisProdPlan)

        thisProdPlan = [el*sumConso/sumProd*self.losses for el in thisProdPlan]
        return thisProdPlan

    def modify_injplan(self, reset=True):
        """
            Generate a new grid state (injection dependant only)
            Then laucnch a load-flow and return the results
        :return: all the load-flow interesting variables
        """
        numberofdivergence = 0
        ok = False
        prod_plan = self.ref_prodP
        # to prevent learning from plan where the net did not converge
        corrNoise = self.corrnoise.noise()
        while not ok:
            # print("modify_injplan: corrNoise : {}".format(corrNoise))

            conso_plan, conso_planre = self.getconsoplan(corrNoise)
            prod_plan = self.getprodplan(self.corrnoise.getdate(), conso_plan)

            full_action = self.action_space({"injection": {
                "load_p": conso_plan,
                "load_q": conso_planre,
                "prod_p": prod_plan
            }})

            self.net.apply_action(full_action)
            if self.dc:
                self.DCnet.apply_action(full_action)

            beg__ = datetime.datetime.now()
            if self.with_powerflow:
                ok = self.net.runpf(is_dc=False)
            else:
                ok = True
            end__ = datetime.datetime.now()

            if self.dc:
                if self.with_powerflow:
                    self.DCnet.runpf(is_dc=True)

            self.nblf += 1
            tmp = end__ - beg__
            self.lftime += tmp.total_seconds() # ellapsed time computing lf, in seconds
            if not ok:
                numberofdivergence += 1
                self.nbdiv += 1
                if numberofdivergence >= int(self.size*self.maxratiodiv):
                    raise RuntimeError("Compute.modify_injplan : Diverge too much !")

        if isinstance(self.net, grid2op.BackendPandaPower.PandaPowerBackend) and self.with_powerflow is False:
            # pandapower backend doesn't modify the load nor production if it doesn't compute
            # a powerflow I need to adapt the code a bit.
            inj_act = 1. * self.net._grid.load["p_mw"].values
            inj_react = 1. * self.net._grid.load["q_mvar"].values
            conso_v = -1

            prod_act = 1. * self.net._grid.gen["p_mw"].values
            prod_react = -1
            prod_v = self.net._grid.gen["vm_pu"].values * self.net.prod_pu_to_kv
            if self.net._grid.gen["bus"].iloc[self.net._id_bus_added] == self.net.gen_to_subid[self.net._id_bus_added]:
                # slack bus and added generator are on same bus. I need to add power of slack bus to this one.
                prod_act[self.net._id_bus_added] += self.net._grid._ppc["gen"][self.net._iref_slack, 1]
        else:
            inj_act, inj_react, conso_v = self.net.loads_info()
            prod_act, prod_react, prod_v = self.net.generators_info()

        transit_MW, transit_MVar, v_or, transit_a = self.net.lines_or_info()
        transit_MW_ext, transit_MVar_ext, v_ex_ext, transit_a_ext = self.net.lines_ex_info()

        if self.dc:
            transit_MW_dc, transit_MVar_dc, v_or_dc, transit_a_dc = self.DCnet.lines_or_info()
            transit_MW_ext_dc, transit_MVar_ext_dc, v_or_ext_dc, transit_a_ext_dc = self.DCnet.lines_or_info()
        else:
            transit_a_dc = 0
            transit_a_ext_dc = 0
            transit_MW_dc = 0
            transit_MW_ext_dc = 0

        return inj_act, inj_react, prod_plan, prod_v, prod_act, conso_v, prod_react,\
               transit_a, transit_MW, transit_MW_ext, transit_a_ext, \
                transit_a_dc, \
                transit_MW_dc, \
                transit_MW_ext_dc, \
                transit_a_ext_dc

    def run(self):
        """
        run the number of experiment given at the initialization
        :return: all the interesting variables for all the load-flow computed
        """
        for numbatch in range(self.size):  # 1 epoch
            inj_act, inj_react, prod_plan, prod_v, prod_act, conso_v, prod_react, \
            transit_a, transit_MW, transit_MW_ext, transit_a_ext, \
            transit_a_dc, transit_MW_dc, transit_MW_ext_dc, transit_a_ext_dc \
                = self.modify_injplan()

            self.inj_act[numbatch, :] = inj_act
            self.inj_react[numbatch, :] = inj_react
            self.prod_act[numbatch, :] = prod_act
            self.prod_plan[numbatch, :] = prod_plan
            self.prod_v[numbatch, :] = prod_v
            self.conso_v[numbatch, :] = conso_v
            self.prod_react[numbatch, :] = prod_react

            self.transit_a[numbatch, :] = transit_a
            self.transit_MW[numbatch, :] = transit_MW
            self.transit_a_ext[numbatch, :] = transit_a_ext
            self.transit_MW_ext[numbatch, :] = transit_MW_ext

            self.transit_a_dc[numbatch, :] = transit_a_dc
            self.transit_MW_dc[numbatch, :] = transit_MW_dc
            self.transit_a_ext_dc[numbatch, :] = transit_a_ext_dc
            self.transit_MW_ext_dc[numbatch, :] = transit_MW_ext_dc

        return self.inj_act, self.inj_react, self.prod_plan, self.prod_v, self.prod_act, self.conso_v, self.prod_react, \
               self.transit_a, self.transit_MW, self.transit_MW_ext, self.transit_a_ext, \
               self.transit_a_dc, self.transit_MW_dc, self.transit_MW_ext_dc, self.transit_a_ext_dc

    def compute(self):
        """
        Compute all the experiement, method to be compliant with method from other classes in this file
        :return:
        """
        self.run()

    def get_results(self):
        """
        :return: the interesting variables saved from the computations
        """
        return self.inj_act, self.inj_react, self.prod_plan, self.prod_v, \
               self.prod_act, self.conso_v, self.prod_react, \
               self.transit_a, self.transit_MW, self.transit_MW_ext, self.transit_a_ext, \
               self.transit_a_dc, self.transit_MW_dc, self.transit_MW_ext_dc, self.transit_a_ext_dc


class ComputeOneThread(mp.Process): #Compute_One_Thread(Thread)
    """ A class to compute n LF (changing inj) on a single trhead. This is thread safe"""

    def __init__(self, size, net, transit=True, withReact=False,
                 maxratiodiv=0.1, dc=False, disco_prod_winter=0.05, disco_prod_summer=0.1,
                 corrnoise=None, uncorrNoiseConso=None,
               sigmaConso=0.05, sigmaProd=0.05,
                 param_QP_distrib="./QP_ratio_distrb.json",
                 with_powerflow=True):
        mp.Process.__init__(self)
        self.size = size
        self.dc = dc

        self.net = Compute(size=size, net=net, transit=transit, withReact=withReact,
                           maxratiodiv=maxratiodiv, dc=dc,
                           disco_prod_winter=disco_prod_winter, disco_prod_summer=disco_prod_summer,
                           corrnoise=corrnoise, uncorrNoiseConso=uncorrNoiseConso,
                           sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                           param_QP_distrib=param_QP_distrib,
                           with_powerflow=with_powerflow)

        # input of lf
        self.inj_act = mp.Array("d", self.net.size*self.net.n_conso)
        self.inj_react = mp.Array("d", self.net.size*self.net.n_conso)
        self.prod_plan = mp.Array("d", self.net.size*self.net.n_prod) # the targetted production plan
        self.prod_v = mp.Array("d", self.net.size*self.net.n_prod)
        # output of lf
        self.prod_act = mp.Array("d", self.net.size*self.net.n_prod) # the realistic production plan
        self.conso_v = mp.Array("d", self.net.size*self.net.n_conso)
        self.prod_react = mp.Array("d", self.net.size*self.net.n_prod)
        self.transit_a = mp.Array("d", self.net.size*self.net.n_quads)
        self.transit_MW = mp.Array("d", self.net.size*self.net.n_quads)
        self.transit_a_ext = mp.Array("d", self.net.size*self.net.n_quads)
        self.transit_MW_ext = mp.Array("d", self.net.size*self.net.n_quads)

        if self.dc:
            self.transit_a_dc = mp.Array("d", self.net.size*self.net.n_quads)
            self.transit_MW_dc = mp.Array("d", self.net.size*self.net.n_quads)
            self.transit_a_ext_dc = mp.Array("d", self.net.size*self.net.n_quads)
            self.transit_MW_ext_dc = mp.Array("d", self.net.size*self.net.n_quads)
        else:
            self.transit_a_dc = 0.
            self.transit_MW_dc = 0.
            self.transit_a_ext_dc = 0.
            self.transit_MW_ext_dc = 0.

        self.lftime = mp.Array("d", 1)  # ellapsed time computing lf, in seconds
        self.nbdiv = mp.Array("d", 1)
        self.nblf = mp.Array("d", 1)

    def run(self):
        inj_act, inj_react, prod_plan, prod_v, prod_act, conso_v, prod_react, \
        transit_a, transit_MW, transit_MW_ext, transit_a_ext, \
        transit_a_dc, transit_MW_dc, transit_MW_ext_dc, transit_a_ext_dc = self.net.run()

        self.inj_act[:] = inj_act.reshape(self.net.size*self.net.n_conso)[:]
        self.inj_react[:] = inj_react.reshape(self.net.size*self.net.n_conso)[:]
        self.prod_act[:] = prod_act.reshape(self.net.size*self.net.n_prod)[:]
        self.prod_plan[:] = prod_plan.reshape(self.net.size*self.net.n_prod)[:]
        self.prod_v[:] = prod_v.reshape(self.net.size*self.net.n_prod)[:]
        self.conso_v[:] = conso_v.reshape(self.net.size*self.net.n_conso)[:]
        self.prod_react[:] = prod_react.reshape(self.net.size*self.net.n_prod)[:]
        self.transit_a[:] = transit_a.reshape(self.net.size*self.net.n_quads)[:]
        self.transit_MW[:] = transit_MW.reshape(self.net.size*self.net.n_quads)[:]
        self.transit_a_ext[:] = transit_a_ext.reshape(self.net.size*self.net.n_quads)[:]
        self.transit_MW_ext[:] = transit_MW_ext.reshape(self.net.size*self.net.n_quads)[:]
        if self.dc:
            self.transit_a_dc[:] = transit_a_dc.reshape(self.net.size * self.net.n_quads)[:]
            self.transit_MW_dc[:] = transit_MW_dc.reshape(self.net.size * self.net.n_quads)[:]
            self.transit_a_ext_dc[:] = transit_a_ext_dc.reshape(self.net.size * self.net.n_quads)[:]
            self.transit_MW_ext_dc[:] = transit_MW_ext_dc.reshape(self.net.size * self.net.n_quads)[:]
        else:
            self.transit_a_dc = 0.
            self.transit_MW_dc = 0.
            self.transit_a_ext_dc = 0.
            self.transit_MW_ext_dc = 0.

        self.lftime[0] = self.net.lftime # ellapsed time computing lf, in seconds
        self.nbdiv[0] = self.net.nbdiv
        self.nblf[0] = self.net.nblf

    def get_results(self):
        inj_act = np.ndarray(shape=(self.net.size, self.net.n_conso)
                          , dtype=np.float
                          , buffer=np.array(self.inj_act[:]))
        inj_react = np.ndarray(shape=(self.net.size, self.net.n_conso)
                          , dtype=np.float
                          , buffer=np.array(self.inj_react[:]))
        prod_act = np.ndarray(shape=(self.net.size, self.net.n_prod)
                          , dtype=np.float
                          , buffer=np.array(self.prod_act[:]))
        prod_plan = np.ndarray(shape=(self.net.size, self.net.n_prod)
                          , dtype=np.float
                          , buffer=np.array(self.prod_plan[:]))
        prod_v = np.ndarray(shape=(self.net.size, self.net.n_prod)
                          , dtype=np.float
                          , buffer=np.array(self.prod_v[:]))
        conso_v = np.ndarray(shape=(self.net.size, self.net.n_conso)
                          , dtype=np.float
                          , buffer=np.array(self.conso_v[:]))
        prod_react = np.ndarray(shape=(self.net.size, self.net.n_prod)
                          , dtype=np.float
                          , buffer=np.array(self.prod_react[:]))
        transit_a = np.ndarray(shape=(self.net.size, self.net.n_quads)
                          , dtype=np.float
                          , buffer=np.array(self.transit_a[:]))
        transit_MW = np.ndarray(shape=(self.net.size, self.net.n_quads)
                          , dtype=np.float
                          , buffer=np.array(self.transit_MW[:]))
        transit_a_ext = np.ndarray(shape=(self.net.size, self.net.n_quads)
                          , dtype=np.float
                          , buffer=np.array(self.transit_a_ext[:]))
        transit_MW_ext = np.ndarray(shape=(self.net.size, self.net.n_quads)
                          , dtype=np.float
                          , buffer=np.array(self.transit_MW_ext[:]))

        if self.dc:
            transit_a_dc = np.ndarray(shape=(self.net.size, self.net.n_quads)
                                   , dtype=np.float
                                   , buffer=np.array(self.transit_a_dc[:]))
            transit_MW_dc = np.ndarray(shape=(self.net.size, self.net.n_quads)
                                    , dtype=np.float
                                    , buffer=np.array(self.transit_MW_dc[:]))
            transit_a_ext_dc = np.ndarray(shape=(self.net.size, self.net.n_quads)
                                       , dtype=np.float
                                       , buffer=np.array(self.transit_a_ext_dc[:]))
            transit_MW_ext_dc = np.ndarray(shape=(self.net.size, self.net.n_quads)
                                        , dtype=np.float
                                        , buffer=np.array(self.transit_MW_ext_dc[:]))
        else:
            transit_a_dc = np.zeros(shape=(1))
            transit_MW_dc = np.zeros(shape=(1))
            transit_a_ext_dc = np.zeros(shape=(1))
            transit_MW_ext_dc = np.zeros(shape=(1))
        return inj_act, inj_react, prod_plan, prod_v, prod_act, conso_v, prod_react, \
               transit_a, transit_MW, transit_MW_ext, transit_a_ext, \
               transit_a_dc, transit_MW_dc, transit_MW_ext_dc, transit_a_ext_dc


class ExecPar:
    def __init__(self, n_core, size, net, transit=True, withReact=False,
                 maxratiodiv=0.1, dc=False
                 , disco_prod_winter=0.05, disco_prod_summer=0.1,
                 corrnoise=None,
                 uncorrNoiseConso=None,
                 sigmaConso=0.05,
                 sigmaProd=0.05,
                 param_QP_distrib="./QP_ratio_distrb.json",
                 with_powerflow=True):
        """
        Parallel execution, over 'n_core'
        of load-flows. This is data parallelism.
        Each process handle a part of 'size' LF """

        self.size = size
        self.thread_pool = []
        self.sizes = [int(size/n_core) for _ in range(n_core)]
        if (size % n_core) != 0:
            remaining = size % n_core
            for i in range(remaining):
                self.sizes[i] += 1
        for i in range(n_core):
            current = ComputeOneThread(size=self.sizes[i],
                                       net=net,
                                       transit=transit,
                                       withReact=withReact,
                                       maxratiodiv=maxratiodiv,
                                       dc=dc,
                                       disco_prod_winter=disco_prod_winter,
                                       disco_prod_summer=disco_prod_summer,
                                       corrnoise=corrnoise,
                                       uncorrNoiseConso=uncorrNoiseConso,
                                       sigmaConso=sigmaConso,
                                       sigmaProd=sigmaProd,
                                       param_QP_distrib=param_QP_distrib,
                                       with_powerflow=with_powerflow)
            self.thread_pool.append(current)

        end__ = time.perf_counter()
        self.lftime = end__ - end__ # ellapsed time computing lf, in seconds
        self.nbdiv = 0
        self.nblf = 0

    def start(self):
        """
        Launch the computation
        :return:
        """
        for el in self.thread_pool:
            el.start()

    def join(self):
        """ Join all the thread: this function terminate when all thread terminate"""
        for el in self.thread_pool:
            el.join()

    def compute(self):
        """ Launch the parrallel computation"""
        self.start()
        self.join()

    def get_results(self):
        """
        :return: the results of the computation in the order:
         inj_acts, inj_reacts, prod_plans, prod_vs, \
               prod_acts, conso_vs, prod_reacts, transit_as, transit_MWs
        """
        inj_acts, inj_reacts, prod_plans, prod_vs, \
        prod_acts, conso_vs, prod_reacts,\
        transit_as, transit_MWs, transit_MW_exts, transit_a_exts, \
        transit_as_dc, transit_MWs_dc, transit_MW_exts_dc, transit_a_exts_dc = (self.thread_pool[0]).get_results()

        self.nbdiv = self.thread_pool[0].nbdiv[0]
        self.lftime = self.thread_pool[0].lftime[0] # ellapsed time computing lf, in seconds
        self.nblf = self.thread_pool[0].nblf[0]

        for el in self.thread_pool[1:]:
            inj_act, inj_react, prod_plan, prod_v, \
            prod_act, conso_v, prod_react, \
            transit_a, transit_MW, transit_MW_ext, transit_a_ext, \
            transit_a_dc, transit_MW_dc, transit_MW_ext_dc, transit_a_ext_dc = el.get_results()

            inj_acts = np.concatenate((inj_acts, inj_act))
            inj_reacts = np.concatenate((inj_reacts, inj_react))
            prod_acts = np.concatenate((prod_acts, prod_act))
            prod_plans = np.concatenate((prod_plans, prod_plan))
            prod_vs = np.concatenate((prod_vs, prod_v))

            conso_vs = np.concatenate((conso_vs, conso_v))
            prod_reacts = np.concatenate((prod_reacts, prod_react))

            transit_as = np.concatenate((transit_as, transit_a))
            transit_MWs = np.concatenate((transit_MWs, transit_MW))
            transit_MW_exts = np.concatenate((transit_MW_exts, transit_MW_ext))
            transit_a_exts = np.concatenate((transit_a_exts, transit_a_ext))

            transit_as_dc = np.concatenate((transit_as_dc, transit_a_dc))
            transit_MWs_dc = np.concatenate((transit_MWs_dc, transit_MW_dc))
            transit_MW_exts_dc = np.concatenate((transit_MW_exts_dc, transit_MW_ext_dc))
            transit_a_exts_dc = np.concatenate((transit_a_exts_dc, transit_a_ext_dc))

            self.nbdiv += el.nbdiv[0]
            self.lftime += el.lftime[0] # ellapsed time computing lf, in seconds
            self.nblf += el.nblf[0]
        return inj_acts, inj_reacts, prod_plans, prod_vs, \
               prod_acts, conso_vs, prod_reacts, \
               transit_as, transit_MWs, transit_MW_exts, transit_a_exts, \
               transit_as_dc, transit_MWs_dc, transit_MW_exts_dc, transit_a_exts_dc


def save_dataframe(array, nm_array, name, colnames, path):
    import pandas as pd
    tmp = pd.DataFrame(array)
    tmp = tmp.rename(columns={i: l for i, l in enumerate(colnames)})
    nm_ = "{}_{}.csv.bz2".format(name, nm_array) if name != "train" else "{}.csv.bz2".format(nm_array)
    tmp.to_csv(os.path.join(path, nm_), index=False, sep=";", float_format='%.1f')


def computeData(size, path, name, num_cores, net,
                transit=True, withReact=False, override=False, dictcalc={},
                maxratiodiv=0.1, dc=False, disco_prod_winter=0.05, disco_prod_summer=0.1,
                corrnoise=None,
                uncorrNoiseConso=None,
                savecsv=False,
               sigmaConso=0.05, sigmaProd=0.05,
                 param_QP_distrib="./QP_ratio_distrb.json",
                with_powerflow=True
                ):
    """ require size_in (integer)
    n_quads(integer)
    ref_inj_plan (vector / list)
    transit=True : transit results in MW, else in A (or a multiple of it)
    """

    if override or (not (os.path.exists(os.path.join(path,"_".join([name, "loads_p"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "loads_q"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "prod_p_target"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "prod_v"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "prod_p"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "conso_v"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "prod_q"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "transits_a"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "transits_MW"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "flowsext_a"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "flowsext_MW"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "transits_a_dc"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "transits_MW_dc"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "flowsext_a_dc"])+".npy")) and
            os.path.exists(os.path.join(path, "_".join([name, "flowsext_MW_dc"])+".npy"))
                         )):
        # if we want to override previous data
        # of if they do not exists
        if num_cores > 1:
            all_situ = ExecPar(size=size, n_core=num_cores, net=net,
                               transit=transit, withReact=withReact, maxratiodiv=maxratiodiv,
                               dc=dc, disco_prod_winter=disco_prod_winter, disco_prod_summer=disco_prod_summer,
                               corrnoise=corrnoise, uncorrNoiseConso=uncorrNoiseConso,
                               sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                               param_QP_distrib=param_QP_distrib, with_powerflow=with_powerflow)
        else:
            all_situ = Compute(size=size, net=net, transit=transit, withReact=withReact,
                               dc=dc, disco_prod_winter=disco_prod_winter, disco_prod_summer=disco_prod_summer,
                               corrnoise=corrnoise, uncorrNoiseConso=uncorrNoiseConso,
                               sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                               param_QP_distrib=param_QP_distrib, with_powerflow=with_powerflow)
        all_situ.compute()

        inj_acts, inj_reacts, prod_plans, prod_vs, prod_acts, conso_vs, prod_reacts, \
        transit_as, transit_MWs, transit_MW_exts, transit_a_exts, \
        transit_as_dc, transit_MWs_dc, transit_MW_exts_dc, transit_a_exts_dc \
            = all_situ.get_results()

        dictcalc[name] = {"nbdiv": all_situ.nbdiv
                      , "lftime": all_situ.lftime
                      , "nblf": all_situ.nblf}
        if not savecsv:
            np.save(os.path.join(path, "_".join([name, "loads_p"])), inj_acts)
            np.save(os.path.join(path, "_".join([name, "loads_q"])), inj_reacts)
            np.save(os.path.join(path, "_".join([name, "prod_p_setpoint"])), prod_plans)
            np.save(os.path.join(path, "_".join([name, "prod_v"])), prod_vs)

            if with_powerflow:
                np.save(os.path.join(path, "_".join([name, "prod_p"])), prod_acts)
                np.save(os.path.join(path, "_".join([name, "loads_v"])), conso_vs)
                np.save(os.path.join(path, "_".join([name, "prod_q"])), prod_reacts)
                np.save(os.path.join(path, "_".join([name, "flows_a"])), transit_as)
                np.save(os.path.join(path, "_".join([name, "flows_MW"])), transit_MWs)
                np.save(os.path.join(path, "_".join([name, "flowsext_a"])), transit_a_exts)
                np.save(os.path.join(path, "_".join([name, "flowsext_MW"])), transit_MW_exts)
                np.save(os.path.join(path, "_".join([name, "flows_a_dc"])), transit_as_dc)
                np.save(os.path.join(path, "_".join([name, "flows_MW_dc"])), transit_MWs_dc)
                np.save(os.path.join(path, "_".join([name, "flowsext_a_dc"])), transit_a_exts_dc)
                np.save(os.path.join(path, "_".join([name, "flowsext_MW_dc"])), transit_MW_exts_dc)
        else:
            save_dataframe(array=inj_acts, nm_array="load_p", name=name, colnames=net.name_loads, path=path)
            save_dataframe(array=inj_reacts, nm_array="load_q", name=name, colnames=net.name_loads, path=path)
            save_dataframe(array=prod_vs, nm_array="prod_v", name=name, colnames=net.name_prods, path=path)
            save_dataframe(array=prod_plans, nm_array="prod_p_setpoint", name=name, colnames=net.name_prods, path=path)

            if with_powerflow:
                save_dataframe(array=transit_as, nm_array="flows_a", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=prod_acts, nm_array="prod_p", name=name, colnames=net.name_prods, path=path)
                save_dataframe(array=prod_reacts, nm_array="prod_q", name=name, colnames=net.name_prods, path=path)
                save_dataframe(array=conso_vs, nm_array="loads_v", name=name, colnames=net.name_loads, path=path)
                save_dataframe(array=transit_MWs, nm_array="flows_MW", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_a_exts, nm_array="flowsext_a", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_MW_exts, nm_array="flowsext_MW", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_as_dc, nm_array="flows_a_dc", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_MWs_dc, nm_array="flows_MW_dc", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_a_exts_dc, nm_array="flowsext_a_dc", name=name, colnames=net.name_lines, path=path)
                save_dataframe(array=transit_MW_exts_dc, nm_array="flowsext_MW_dc", name=name, colnames=net.name_lines, path=path)


def computeAll(sizes, path_save, num_cores, net, comp_transit=True,
               withReact=False, suff="", override=False, dictcalc={},
               quiet=False, maxratiodiv=0.1, dc=False,
               disco_prod_winter=0.05, disco_prod_summer=0.1,
                corrnoise=None, uncorrNoiseConso=None,
               savecsv=False,
               sigmaConso=0.05, sigmaProd=0.05,
                 param_QP_distrib="./QP_ratio_distrb.json",
               with_powerflow=True):
    """
    Compute the load-flows for taining, test and validation set.
    :param sizes:
    :param path_save:
    :param num_cores:
    :param net:
    :param comp_transit:
    :param withReact:
    :param suff:
    :param override:
    :param dictcalc:
    :param quiet: do you print computation timing
    :return:
    """
    suffixes = ""
    if suff != "":
        suffixes = "_"+suff

    now = time.time()
    beginning = now
    computeData(size=sizes[0]
                , path=path_save
                , name="train"+suffixes
                , num_cores=num_cores
                , net=net
                , transit=comp_transit
                , withReact=withReact
                , override=override
                , dictcalc=dictcalc
                , maxratiodiv=maxratiodiv
                , dc=dc
                , disco_prod_winter=disco_prod_winter
                , disco_prod_summer=disco_prod_summer,
                corrnoise=corrnoise,
                uncorrNoiseConso=uncorrNoiseConso,
                savecsv=savecsv, sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                param_QP_distrib=param_QP_distrib,
                with_powerflow=with_powerflow)
    if not quiet:
        print("Computed the training set (size", sizes[0], ") in %.2fs" % (time.time() - now))
    now = time.time()
    computeData(size=sizes[1]
                , path=path_save
                , name="test"+suffixes
                , num_cores=num_cores
                , net=net
                , transit=comp_transit
                , withReact=withReact
                , override=override
                , dictcalc=dictcalc
                , maxratiodiv=maxratiodiv
                , dc=dc
                , disco_prod_winter=disco_prod_winter
                , disco_prod_summer=disco_prod_summer,
                corrnoise=corrnoise,
                uncorrNoiseConso=uncorrNoiseConso,
                savecsv=savecsv, sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                param_QP_distrib=param_QP_distrib,
                with_powerflow=with_powerflow)
    if not quiet:
        print("Computed the test set (size",  sizes[1], ") in %.2fs" % (time.time() - now))
    now = time.time()
    computeData(size=sizes[2]
                , path=path_save
                , name="val"+suffixes
                , num_cores=num_cores
                , net=net
                , transit=comp_transit
                , withReact=withReact
                , override=override
                , dictcalc=dictcalc
                , maxratiodiv=maxratiodiv
                , dc=dc
                , disco_prod_winter=disco_prod_winter
                , disco_prod_summer=disco_prod_summer,
                corrnoise=corrnoise,
                uncorrNoiseConso=uncorrNoiseConso,
                savecsv=savecsv, sigmaConso=sigmaConso, sigmaProd=sigmaProd,
                param_QP_distrib=param_QP_distrib,
                with_powerflow=with_powerflow)
    if not quiet:
        print("Computed the validation set (size",  sizes[2], ") in %.2fs" % (time.time() - now))
        print("Done in %.2f" % (time.time() - beginning))
