import os
import time
import random
import pdb

from grid2op.BackendPandaPower import PandaPowerBackend
from pgdg import computeAll, SubHourlyNoise, UncorrNoiseConso, Noise, computeData

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Launch N-1 computation')
    parser.add_argument('--nnode', dest='nnode',
                        default=30,
                        help='Number of nodes for the powergrid to consider')
    parser.add_argument('--numcores', dest='numcores',
                        default=1,
                        help='Number of processes used for the computation')
    parser.add_argument('--inputfile', dest='inputfile',
                        default=None,
                        help='Number of processes used for the computation')
    parser.add_argument('--size', dest='size',
                        default=None,
                        help='Size of the training set generated')
    parser.add_argument('--path', dest='path',
                        default=None,
                        help='Path where the data will be stored')
    parser.add_argument('--with_powerflow', dest='with_powerflow',
                        default="True", type=str,
                        help='Do you compute a powerflow with the generated loads and generation (True, default, but slow) or not.')

    args = parser.parse_args()
    nnodes = int(args.nnode)
    numcores = int(args.numcores)
    with_powerflow = True if args.with_powerflow == "True" or args.with_powerflow == "T" or args.with_powerflow == "1" else False

    if args.size is None:
        if nnodes == 30:
            size = int(1e4)
        else:
            size = int(5e3)
    else:
        size = int(args.size)

    if args.path is None:
        path_save = "./data_generated"
    else:
        path_save = os.path.join(args.path)
    path_save = os.path.abspath(path_save)

    if not os.path.exists(path_save):
        os.mkdir(path=path_save)
        print("Creating path \"{}\" to store the data".format(path_save))

    if args.inputfile is None:
        fIn = os.path.abspath("data/case{}.json".format(nnodes))
    else:
        fIn = os.path.abspath(args.inputfile)

    print("Computing Train - Test - Val for \"{}\"".format(fIn))
    print("Saving results in {}".format(path_save))
    # sys.exit("Enough for today")

    net = PandaPowerBackend()
    net.load_grid(path=fIn)

    random.seed(42)
    corrnoise = Noise(path="param/consos.json")  # represents the correlated noise
    uncorrNoiseConso = UncorrNoiseConso() # represents the uncorrelated noise

    beginning = time.time()

    comp_transit = False  # True : MW, False : Amps
    withReact = True  # True: modify also the reactive power for loads
    override = True

    name = "{}_bus".format(nnodes)
    num_cores = numcores
    net = net
    transit = comp_transit
    dict_metacalc = {}
    dict_metacalc["bc"] = {}

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    computeData(size=size,
               path=path_save,
                name=name,
                num_cores=num_cores,
               net=net,
               withReact=withReact,
               dictcalc=dict_metacalc["bc"],
               override=override,
               maxratiodiv=10,
               corrnoise=corrnoise,
               uncorrNoiseConso=uncorrNoiseConso,
               savecsv=True,
               # sigmaConso=0.05/(corrnoise.nb_timestep/2),
               # sigmaProd=0.05/(corrnoise.nb_timestep/2),
               param_QP_distrib="param/QP_ratio_distrb.json",
                with_powerflow=with_powerflow
               )

    outfile = open(os.path.join(path_save, 'computation_infos.json'), 'w')
    path_save_tmp = path_save
    with open(os.path.join(path_save_tmp, 'computation_infos_tmp.json'), 'w') as outtmp:
        json.dump(dict_metacalc, outtmp)




