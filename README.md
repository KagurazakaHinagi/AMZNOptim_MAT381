# AMaZoN OPTIMization for local delivery

> This is a Spring 25 Math 381 Course Project of University of Washington.

## User Guide

### (Recommended) Google Colab Demo

WIP

### Local Installation

1. Install `git` and optain AMZNOptim repository

```bash
# Use HTTPS
git clone https://github.com/KagurazakaHinagi/AMZNOptim_MAT381.git
# or SSH
git clone git@github.com:KagurazakaHinagi/AMZNOptim_MAT381.git
```

2. (Recommended) Set up Virtual Environment

```bash
# Use Conda
conda create -n amznoptim python=3.12
conda activate amznoptim
# or Python venv
python3 -m venv .venv
source .venv/bin/activate
```

3. Install `amznoptim` package

```bash
pip install .
# For development, use
pip install -e '.[dev]'
```

4. For example usage, please refer to [example folder](example)

### Google Maps Platform

To run this project, you need to have your own Google Maps Platform account that enables the following APIs:

- Routes API
- Address Validation API
- Maps Static API

You can acquire an account with monthly free trials at [here](https://developers.google.com/maps).

### Data Preparation

WIP

## Constraint Programming Spec

### Single-Depot VRP

#### Decision Variables

- $x_{i, j, k} \in \\{0, 1\\}$: truck $k$ travels directly from stop $i$ to stop $j$.
- $y_{o, k} \in \\{0, 1\\}$: order $o$ is served by truck $k$.
- $u_{i, k} \ge 0$: "timestamp" when truck $k$ arrives at stop $i$.
- $z_{k} \in \\{0, 1\\}$: truck $k$ is used.
- $\text{vis}_{j, k} \in \\{0, 1\\}$: (auxiliary variable) stop $j$ is visited by vehicle $k$.

#### Parameters

- $t_{i, j}$: travel time from stop $i$ to stop $j$.
- $\hat{t}_j$: stopover time at stop $j$.
- $d_{i, j}$: distance from stop $i$ to stop $j$.
- $w_o, v_o$: weight and volume of package $o$.
- $W_k, V_k$: weight and volume capacity of vehicle $k$.
- $P_o$: priority score for order $o$, calculated by the order waiting time.
- $D_k$: maximum cruising range of vehicle $k$.
- $T$: maximum driver duty time.
- $K$: total number of available trucks.
- $N$: total number of distinct addresses in the order list.
- $O$: total number of orders in the order.
- $s(o)$: stop index of order $o$.

#### Objective and constraints

```math
\begin{aligned}
    \text{minimize} &\space \sum_{k=1}^{K}\sum_{i=0}^N\sum_{j=0}^N (t_{i, j}x_{i, j, k}
                        +\hat{t}_{j} y_{j, k})
                        -\alpha \sum_{k=1}^K\sum_{o=1}^{O} P_oy_{o, k}
                        +\beta \sum_{k=1}^K z_k & \cr
    \text{s.t.}     &\space \sum_{i=0,i\ne s(o)}^N x_{i, s(o),k} \ge y_{o, k},
                        \quad \forall o, k & \text{(1a)} \cr
                    &\space \sum_{j=1}^N x_{0, j, k}=\sum_{i=1}^N x_{i, 0, k}=z_k,
                        \quad \forall k & \text{(1b)} \cr
                    &\space \sum_{i=0,i\ne h}^N x_{i, h, k} =\sum_{j=0,j\ne h}^N x_{h, j, k},
                        \quad \forall k, h\in [n] & \text{(1c)} \cr
                    &\space \sum_{o=1}^{O} w_o y_{o,k} \le W_k, \space \sum_{o=1}^{O} v_o y_{o,k} \le V_k,
                        \quad \forall k & \text{(2)} \cr
                    &\space z_k=\max \{y_{o, k}: o \in \{1, \dots, O\}\},
                        \quad \forall k & \text{(3)} \cr
                    &\space u_{i,k}+1 \le u_{j,k} + N\cdot (1 - x_{i, j, k}),
                        \quad \forall i \ne j,k & \text{(4)} \cr
                    &\space \sum_{i=0}^N \sum_{j=0}^N t_{i, j}x_{i, j, k}+\sum_{j \in S_k} \hat{t}_j
                        \cdot \text{vis}_{j,k} \le T,
                        \quad \forall k & \text{(5a)} \cr
                    &\space \text{vis}_{j,k} \ge y_{o,k},
                        \quad \forall k,o:s(o)=j & \text{(5b)} \cr
                    &\space \text{vis}_{j,k} \le \sum_{o:s(o)=j} y_{o,k},
                        \quad \forall j\in S_k,k & \text{(5c)} \cr
                    &\space \sum_{i=0}^N \sum_{j=0}^N d_{i,j}x_{i,j,k}\le D_k,
                        \quad \forall k & \text{(6)} \cr
                    &\space \sum_{k=1}^K y_{o,k}=1,
                        \quad \forall o & \text{(7)} \cr
                    &\space \sum_{k=1}^K \sum_{i=0}^N x_{i,j,k}=1,
                        \quad \forall j & \text{(8)} \cr
                    &\space x_{i,j,k},y_{o,k},z_k,\text{vis}_{j,k}\in \{0,1\} \cr
                    &\space u_{i,k} \in [0,N]
\end{aligned}
```

##### Explanation of constraints

1. Routing Constraints:\
    (1a). Ensure the served orders have incoming edges to their stops.\
    (1b-1c). Enforce flow conservaton separately for depot and non-depot nodes.
2. Vehicle Capacity Constraints:\
    Each vehicle cannot load packages that the total exceeds the maximum capacity of weight or volume.
3. Vehicle Usage Constraint:\
    If any package is assigned to a specific vehicle, then that vehicle need to be marked as chosen.
4. Subtour Elimination (Miller-Tucker-Zemlin):\
    Prevent disconnected loops by constraining on the order of the visiting time.\
    See [Wikipedia page](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation).
5. Time Constraints:\
    (5a). The total travel time + stopover time must not exceeds the driver duty time.\
    (5b-5c). A stop must be visited iff some package corresponding to that stop is assigned to the vehicle.
6. Distance Constraint:\
    The total traveling distance must not exceed the vehicle's cruising range.
7. Package Assignment Constraint:
    Each package is assigned to exactly one vehicle.
8. Stop Assignment Constraint:
    Each stop is visited exactly once.

### Multi-Depot VRP

WIP

## References

- WIP
