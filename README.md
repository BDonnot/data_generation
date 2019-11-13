# data_generation

## Introduction
This repository exhibit the code used during my phd to generate load and generation profile for the IEEE case 118.

The main entry point of this package is with the script [SampleInjection.py](SampleInjection.py).
See :
```bash
python SampleInjection.py --help
```
for more information

We will use the following vocabulary:

- a powergrid has loads subscripted with index $i$
- a powergrid has generators subscripted with index $j$
- we want to generate powergrid injections (loads or generators values) for different "time step" $t$

This file briefly explain the generation process that I used during my PhD. It is far from
perfect and comes without any guarantee.

This package is in early beta for now. It's barely documented beside this ReadMe.

More advanced methods to generate data will probably be added in the future.

## Dependencies
This package heavily uses `pandas`, `numpy` and can use `scipy`. It also require
to install the `Grid2Op` package which is a framework to perform easy grid manipulation
built on top of `PandaPower`.

To install this package (required to run [SampleInjection.py](SampleInjection.py) script) 
simply do, from a command line:
```bash
git clone https://github.com/BDonnot/data_generation
cd data_generation
pip3 install -U .
```

The package will be installed in the python3 environment and will be accessible as
```python
import pgdg
```
`pgdg` stands for "Power Grid Data Generator".

## Loads generation process

### Active load

The consumption of load $i$ of a powergrid  at time $t$ is the multiplication of multiple factor:
 - $h_t$ that depends on the hour of day (daily load pattern)
 - $d_t$ that depends on the day of the week (weekly load pattern)
 - $w_t$ that depends on the week of the year (yearly load pattern)
 - $l_{i,t}$ which is an individual factor that depends on the load $i$. $l_i \sim LogNormal$
 
$h_t * d_t * w_t$ is the same, at a given time $t$ for all the load of the powergrid, it is 
then called the "correlated noise".
It is calibrated using the paramter files [consos.json](param/consos.json) containing data computed
from the French load of 2012.

The file [CorrelatedNoises.py](pgdg/CorrelatedNoises.py) gives some implementation of 
such "noises". The main difference between all of them is:

- `CorrNoise`: generates totally random noises. Two consecutive calls of "CorrNoise.noise()"
will give  totally unrelated values. It should not be used for generated temporal data.
- `HourlyNoise`: generates hourly noises. Two consecutive calls of "HourlyNoise.noise()" 
will give two consecutive values of $h_t * d_t * w_t$. For example, if the a call to "HourlyNoise.noise()"
represents the time stamp of the 12 december 2012 at **02**:00 am then the next call to this
function will generate the "correlated noise" of the 12 december 2012 at **03**:00 am. 
It should be use if some temporal properties are expected.
- `SubHourlyNoise` can downscale (using cubic polynomial interpolation) the noise to 
get data at an higher frequency than 1h, usually 5 mins for example.

$l_{t,i}$ is chosen for each load at each "time step"

 ### Reactive load
 
 Once the active load $c_{i,t}$ is known for each load $i$, the reactive load is computed using
 the distribution of the ratio $P/Q$ calibrated using the French grid states of 2012. This 
 dataset is not publically available, only the calibrated data are stored in
 [QP_ratio_distrib.json](param/QP_ratio_distrb.json).
 
 ## Generators setpoint values
 No model of unit commitment or econimic dispatch is run in this script. Thus the generator
 data might look totally irrealistic especially when looking at the time series generated.
 For example, the "ramp" of the generators are completely neglected: a generator can 
 be connected at a time step, then disconnected, then reconnected again for example.
 
 Generating the value of generator $g_{j,t}$ is a three stage process:
  
  - first it is choosen randomly if the generator is connected of not. If it's not connected
  then it ouputs 0 (see `Compute.disconnectprod` in file [GenerateData](pgdg/GenerateData.py)). Otherwise:
  - For all participating generators a value is sampled a first time so that the equilibrium 
  $P=C+L$ is encouraged. Each generator is assigned a production that is drawn from a
  LogNormal distribution (see `Compute.modifyprod` in file [GenerateData](pgdg/GenerateData.py))
  - then the equilibrium for steady state is enforced by scaling everything. Note that this
  step might break the assumption that a production is above $P_{min}$ and bellow $P_{max}$.
  
## Voltages and flows
By default the voltages are not modified from the original test case.

If asked too, this software can compute the flows resulting from the application of the given 
productions and consumption. In this case it uses the package 
[Grid2op](https://github.com/rte-france/Grid2Op). In that case data from flows, voltages 
as well as real production values (as opposed to the setpoint value described in
 `Generators setpoint values`) will also be saved.

Note that in case of divergence, another noise is generated. $h_t * d_t * w_t$  is kept, 
but everything else is re generated.



 