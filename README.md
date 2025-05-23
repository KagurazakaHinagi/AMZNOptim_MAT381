# AMaZoN OPTIMization for local delivery

> This is a Spring 25 Math 381 Course Project of University of Washington.

## User Guide

### (Recommended) Google Colab Demo

WIP

### Local Installation

WIP

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

- $x_{i, j, k} \in \\{0, 1\\}$: truck $k$ travels directly from $i$ to $j$.
- $y_{i, k} \in \\{0, 1\\}$: node $i$ is served by truck $k$.
- $u_{i, k} \ge 0$: "timestamp" when truck $k$ arrives at node $i$.
- $z_{k} \in \\{0, 1\\}$: truck $k$ is used.

#### Parameters

- $t_{i, j}$: travel time from $i$ to $j$.
- $\hat{t}_j$: stopover time at address $j$.
- $w_i, v_i$: weight and volume of package $i$.
- $W_k, V_k$: weight and volume capacity of vehicle $k$.
- $P_i$: priority score for address $i$, calculated by the order waiting time (see below)
- $T$: maximum driver duty time.
- $K$: total number of available trucks.
- $N$: total number of distinct addresses in the order list.

#### Objective and constraints

```math
\begin{aligned}
    \text{minimize} &\space \sum_{k=1}^{K}\sum_{i=0}^N\sum_{j=0}^N (t_{i, j}x_{i, j, k}
                        +\hat{t}_{j} y_{j, k})
                        -\alpha \sum_{k=1}^K\sum_{i=1}^N P_iy_{i, k}
                        +\beta \sum_{k=1}^K z_k & \cr
    \text{s.t.}     &\space \sum_{j=0}^N x_{i, j, k}=y_{i, k}=\sum_{j=0}^N x_{j, i, k},
                        \quad \forall i, k & \text{(1)} \cr
                    &\space \sum_{i=1}^N w_{i}y_{i, k} \le W_k,
                        \quad \sum_{i=1}^N v_{i}y_{i, k} \le V_k,
                        \quad \forall k & \text{(2)} \cr
                    &\space \sum_{j=1}^N x_{0, j, k} = \sum_{i=1}^N x_{i, 0, k} = z_k,
                        \quad \forall k & \text{(3)} \cr
                    &\space u_{i, k} + 1 \le u_{j, k} + N \cdot (1 - x_{i, j, k}),
                        \quad \forall i \neq j, k & \text{(4)} \cr
                    &\space \sum_{k=1}^K y_{i, k} \le 1,
                        \quad \forall i & \text{(5)} \cr
                    &\space \sum_{k=1}^K\sum_{i=0}^N\sum_{j=0}^N (t_{i, j}x_{i, j, k}
                        + \hat{t}_{j}y_{j, k}) \le T & \text{(6)}
\end{aligned}
```

##### Explanation of constraints

1. Linking visits to form a route:\
    If some stop $i$ is visited by vehicle $k$, then there is exactly one arc enters and leave that stop.
2. Capacity Constraint:\
    When loading the packages onto vehicle $k$, the total volume and weight could not exceed the payload and the truck volume of that vehicle.
3. Depot visits in the route:\
    If truck $k$ is used, then it leaves and returns to the depot once.
4. Subtour Elimination (Miller-Tucker-Zemlin):\
    Prevent disconnected loops by constraining on the order of visiting time.
    See [Wikipedia page](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation).
5. Single visit to each stop:\
    Each stop cannot be visited more than one times. Otherwise it's inefficient.
6. Duty time:\
    The total traveling time + stopover time cannot exceed the driver's maximum duty time.

### Multi-Depot VRP

WIP

## References

- WIP
