# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Average cost of energy by proportional sharing of supply costs.

Within a snapshot, all energy supplied to a bus is mixed, so every withdrawal from
that bus pays the same average cost. Branches (links, processes, lines and
transformers) pass the costs of their inputs plus their own costs on to their
primary output, which is the supplying port with the lowest index (e.g. `bus1` of a
link in forward flow, `bus0` in reverse flow). Further outputs (e.g. oxygen of an
electrolyser) are free. Storages pass on the average costs of the energy they hold.

Since the energy flows are known from the optimisation, all cost relations are
linear. The average costs of all buses and the costs of the energy held by all
storages are solved for all snapshots at once:

    bus b, snapshot t:      c[b,t] W[b,t] - Σ_inputs E_in[j,t] c[j,t] - Σ_storages d[s,t] K[s,t-1] = L[b,t]
    storage s, snapshot t:  K[s,t] - (1 - d[s,t]) K[s,t-1] - E_charge[s,t] c[b,t] = H[s,t]

with the average cost `c` of bus `b` [currency/MWh], the withdrawal `W` from the
bus, the cost `K` of the energy held by storage `s` [currency], the share `d` of the
held energy that is discharged, the local supply costs `L` and the holding costs `H`
of the storage. Energy flowing in loops needs no special treatment, as long as it
reaches a final withdrawal. Costs of energy that only circulates between buses (e.g.
surplus energy burned in lossy links) cannot be passed on, so their average costs are
undefined (NaN) and the costs are not attributed.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.linalg import MatrixRankWarning, spsolve

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)

# Energy below this threshold is treated as solver noise [MWh].
EPS = 1e-6

# Operational costs scaling with the dispatch, attributed to the energy supplied in the
# same snapshot.
DISPATCH_COST_TYPES = ["marginal_cost", "marginal_cost_quadratic"]
# Operational costs for holding or spilling energy, added to the cost of stored energy.
STORAGE_HOLDING_COST_TYPES = ["marginal_cost_storage", "spill_cost"]
# Unit commitment costs, spread over all energy supplied by the component like capital
# costs.
COMMITMENT_COST_TYPES = ["start_up_cost", "shut_down_cost", "stand_by_cost"]
# Costs occurring once per event, all other costs scale with the snapshot weighting.
UNWEIGHTED_COST_TYPES = ["start_up_cost", "shut_down_cost"]

# Attributes of the stored energy, its cyclicity and its initial value.
STORAGE_ATTRS = {
    "StorageUnit": (
        "state_of_charge",
        "cyclic_state_of_charge",
        "state_of_charge_initial",
    ),
    "Store": ("e", "e_cyclic", "e_initial"),
}


def calculate_average_cost(n: Network) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the average cost of energy of all flows between components and buses.

    Parameters
    ----------
    n : Network
        Solved network.

    Returns
    -------
    energy : pd.DataFrame
        Energy supplied to the bus (positive) or withdrawn from it (negative) for each
        snapshot, weighted by the objective snapshot weightings. Columns are a
        MultiIndex with the levels `component`, `name` and `bus`.
    cost : pd.DataFrame
        Average cost of the energy of each flow and snapshot in currency/MWh, NaN for
        snapshots without flow and for free outputs of branches. Same shape as
        `energy`.

    """
    _check_supported(n)
    df_energy, flow_port = _get_flows(n)
    columns = df_energy.columns
    energy = df_energy.to_numpy(dtype=float)
    n_snapshots, n_flows = energy.shape
    supply = energy > EPS
    withdrawal = energy < -EPS

    buses = n.c.buses.static.index
    n_buses = len(buses)
    flow_bus = buses.get_indexer(columns.get_level_values("bus"))
    flow_component = columns.get_level_values("component")
    units = list(
        dict.fromkeys(
            zip(flow_component, columns.get_level_values("name"), strict=True)
        )
    )
    unit_flows: dict[tuple[str, str], list[int]] = {unit: [] for unit in units}
    for i, unit in enumerate(
        zip(flow_component, columns.get_level_values("name"), strict=True)
    ):
        unit_flows[unit].append(i)

    # Only the supplying port with the lowest index of a branch carries costs.
    primary_supply = supply.copy()
    for (c, _), flows in unit_flows.items():
        if c not in n.branch_components:
            continue
        supplied = np.zeros(n_snapshots, dtype=bool)
        for i in sorted(flows, key=lambda i: flow_port[i]):
            primary_supply[:, i] &= ~supplied
            supplied |= supply[:, i]

    # Local supply costs and holding costs of storages per snapshot and flow [currency].
    local_cost, holding_cost = _get_local_cost(n, energy, primary_supply, unit_flows)

    # Unknowns: average cost c[b,t] at position b * T + t, followed by the cost of the
    # energy held by the storages K[s,t].
    storage_flows = [i for i, c in enumerate(flow_component) if c in STORAGE_ATTRS]
    n_bus_unknowns = n_buses * n_snapshots
    n_unknowns = n_bus_unknowns + len(storage_flows) * n_snapshots

    def idx_bus(bus: int | np.ndarray, t: np.ndarray) -> np.ndarray:
        return bus * n_snapshots + t

    def idx_storage(i_storage: int, t: np.ndarray) -> np.ndarray:
        return n_bus_unknowns + i_storage * n_snapshots + t

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rhs = np.zeros(n_unknowns)

    # Bus rows. Buses without withdrawal get the row c[b,t] = 0.
    flow_to_bus = sp.csr_matrix(
        (np.ones(n_flows), (np.arange(n_flows), flow_bus)), shape=(n_flows, n_buses)
    )
    bus_withdrawal = np.asarray(flow_to_bus.T @ np.where(withdrawal, -energy, 0.0).T)
    rows.append(np.arange(n_bus_unknowns))
    cols.append(np.arange(n_bus_unknowns))
    values.append(np.where(bus_withdrawal > EPS, bus_withdrawal, 1.0).ravel())
    bus_local_cost = np.asarray(flow_to_bus.T @ local_cost.T)
    rhs[:n_bus_unknowns] = np.where(bus_withdrawal > EPS, bus_local_cost, 0.0).ravel()

    # The primary output of a branch carries the costs of all its inputs.
    branch_input_pairs: list[tuple[int, int]] = []
    for (c, _), flows in unit_flows.items():
        if c not in n.branch_components:
            continue
        for i_out in flows:
            for i_in in flows:
                if i_in == i_out:
                    continue
                t = np.flatnonzero(primary_supply[:, i_out] & withdrawal[:, i_in])
                if t.size == 0:
                    continue
                branch_input_pairs.append((i_out, i_in))
                rows.append(idx_bus(flow_bus[i_out], t))
                cols.append(idx_bus(flow_bus[i_in], t))
                values.append(energy[t, i_in])

    # Storage rows and the cost of discharged energy in the bus rows.
    discharge_share = np.zeros((n_snapshots, len(storage_flows)))
    previous = np.full((n_snapshots, len(storage_flows)), -1)
    for i_storage, i in enumerate(storage_flows):
        positions, held_available, held_after = _get_storage_tracking(
            n, flow_component[i], columns.get_level_values("name")[i]
        )
        e = energy[positions, i]
        discharging = e > EPS
        charging = e < -EPS
        # The standing loss of a snapshot affects all held energy, so the discharged
        # share refers to the energy left after the loss.
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(
                held_available > EPS,
                np.clip((held_available - held_after) / held_available, 0.0, 1.0),
                1.0,
            )
        share = np.where(discharging, share, 0.0)
        discharge_share[positions, i_storage] = share
        # The tracking starts without costs, so the first position has no predecessor.
        previous[positions[1:], i_storage] = positions[:-1]

        idx_k = idx_storage(i_storage, positions)
        rows += [idx_k, idx_k[1:]]
        cols += [idx_k, idx_storage(i_storage, positions[:-1])]
        values += [np.ones(n_snapshots), -(1.0 - share[1:])]

        t_charge = positions[charging]
        rows.append(idx_storage(i_storage, t_charge))
        cols.append(idx_bus(flow_bus[i], t_charge))
        values.append(energy[t_charge, i])

        k_discharge = np.flatnonzero(discharging[1:]) + 1
        rows.append(idx_bus(flow_bus[i], positions[k_discharge]))
        cols.append(idx_storage(i_storage, positions[k_discharge - 1]))
        values.append(-share[k_discharge])

        rhs[idx_storage(i_storage, np.arange(n_snapshots))] = holding_cost[:, i]

    matrix = sp.csc_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_unknowns, n_unknowns),
    )
    logger.debug(
        "Average cost: solving %d unknowns with %d non-zeros.", n_unknowns, matrix.nnz
    )

    # Costs that cannot reach a final withdrawal have no average cost. Their rows are
    # replaced by c = 0, so that the rest of the system, which does not depend on
    # them, can be solved.
    trapped = _get_trapped_unknowns(matrix)
    if trapped.any():
        trapped_bus = trapped[:n_bus_unknowns].reshape(n_buses, n_snapshots)
        logger.warning(
            "Costs of %.2f currency are not attributed, as energy circulates between "
            "the buses %s in %d snapshots without being withdrawn by a final consumer "
            "(e.g. surplus energy burned in lossy links). Their average costs are "
            "undefined (NaN).",
            rhs[trapped].sum(),
            list(buses[trapped_bus.any(axis=1)]),
            trapped_bus.any(axis=0).sum(),
        )
        coo = matrix.tocoo()
        keep = ~trapped[coo.row]
        matrix = sp.csc_matrix(
            (
                np.concatenate((coo.data[keep], np.ones(trapped.sum()))),
                (
                    np.concatenate((coo.row[keep], np.flatnonzero(trapped))),
                    np.concatenate((coo.col[keep], np.flatnonzero(trapped))),
                ),
            ),
            shape=(n_unknowns, n_unknowns),
        )
        rhs[trapped] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            solution = np.asarray(spsolve(matrix, rhs), dtype=float)
        except MatrixRankWarning as e:
            msg = (
                "The average cost can not be determined, the linear system is singular."
            )
            raise ValueError(msg) from e
    if not np.isfinite(solution).all():
        msg = "The average cost can not be determined, the linear system has no finite solution."
        raise ValueError(msg)
    solution[trapped] = np.nan

    bus_cost = solution[:n_bus_unknowns].reshape(n_buses, n_snapshots).T
    storage_cost = solution[n_bus_unknowns:].reshape(len(storage_flows), n_snapshots).T

    # Total supply costs per snapshot and flow [currency].
    supply_cost = local_cost
    for i_out, i_in in branch_input_pairs:
        mask = primary_supply[:, i_out] & withdrawal[:, i_in]
        supply_cost[mask, i_out] -= energy[mask, i_in] * bus_cost[mask, flow_bus[i_in]]
    for i_storage, i in enumerate(storage_flows):
        has_previous = previous[:, i_storage] >= 0
        supply_cost[has_previous, i] += (
            discharge_share[has_previous, i_storage]
            * storage_cost[previous[has_previous, i_storage], i_storage]
        )

    cost = np.full_like(energy, np.nan)
    cost[withdrawal] = bus_cost[:, flow_bus][withdrawal]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost[primary_supply] = (supply_cost / energy)[primary_supply]

    return df_energy, pd.DataFrame(cost, index=df_energy.index, columns=columns)


def _get_trapped_unknowns(matrix: sp.csc_matrix) -> np.ndarray:
    """Get the unknowns whose costs cannot reach a final withdrawal.

    Costs are passed on without losses: an off-diagonal entry in row v and column u
    means that costs of u go to v. The column sum is the part of the costs of u that
    leaves the system through final withdrawals (loads, energy left in storages at the
    end). If energy only circulates between buses, the costs of those buses cannot
    leave, the linear system is singular and their average costs are undefined.
    """
    n_unknowns = matrix.shape[0]
    exits = np.flatnonzero(np.asarray(matrix.sum(axis=0)).ravel() > EPS)
    # Graph with an edge v -> u for costs going from u to v, plus a node connected to
    # all unknowns with an exit. All unknowns reached from that node pass their costs
    # on to a final withdrawal.
    coo = matrix.tocoo()
    off_diagonal = coo.row != coo.col
    graph = sp.csr_matrix(
        (
            np.ones(off_diagonal.sum() + exits.size),
            (
                np.concatenate(
                    (coo.row[off_diagonal], np.full(exits.size, n_unknowns))
                ),
                np.concatenate((coo.col[off_diagonal], exits)),
            ),
        ),
        shape=(n_unknowns + 1, n_unknowns + 1),
    )
    reached = breadth_first_order(
        graph, n_unknowns, directed=True, return_predecessors=False
    )
    trapped = np.ones(n_unknowns + 1, dtype=bool)
    trapped[reached] = False
    return trapped[:n_unknowns]


def _check_supported(n: Network) -> None:
    if n.has_scenarios:
        msg = "The average cost is not supported for stochastic networks yet."
        raise NotImplementedError(msg)
    if n.has_investment_periods:
        msg = "The average cost is not supported for networks with investment periods yet."
        raise NotImplementedError(msg)
    if not n.is_solved:
        msg = "The average cost can only be calculated for a solved network."
        raise ValueError(msg)
    for c in n.controllable_branch_components:
        static = n.c[c].static
        delays = static.filter(regex=r"^delay\d*$")
        if not static.empty and (delays.fillna(0) != 0).any().any():
            msg = f"The average cost is not supported for {c} components with a delay yet."
            raise NotImplementedError(msg)


def _get_flows(n: Network) -> tuple[pd.DataFrame, np.ndarray]:
    """Get the energy of all flows between components and buses.

    Returns the energy supplied to the bus for each snapshot and flow and the port
    index of each flow. Flows without energy are dropped, ports of a component that
    connect to the same bus are summed.
    """
    weightings = n.snapshot_weightings.objective
    frames = []
    for c in sorted(n.one_port_components | n.branch_components):
        static = n.c[c].static
        if static.empty:
            continue
        is_branch = c in n.branch_components
        for port in n.c[c].ports if is_branch else [""]:
            attr = f"p{port}"
            if attr not in n.c[c].dynamic or n.c[c].dynamic[attr].empty:
                continue
            p = n.c[c].dynamic[attr].reindex(columns=static.index).fillna(0.0)
            sign = -1.0 if is_branch else static["sign"]
            energy = p.mul(sign, axis=1).mul(weightings, axis=0)
            bus = static[f"bus{port}"]
            keep = (bus.fillna("") != "") & (energy.abs() > EPS).any()
            energy = energy.loc[:, keep]
            energy.columns = pd.MultiIndex.from_arrays(
                [
                    [c] * keep.sum(),
                    energy.columns,
                    bus[keep],
                    [int(port) if port else 0] * keep.sum(),
                ],
                names=["component", "name", "bus", "port"],
            )
            frames.append(energy)

    if not frames:
        msg = "The network has no energy flows."
        raise ValueError(msg)
    df = pd.concat(frames, axis=1)
    ports = (
        df.columns.to_frame(index=False)
        .groupby(["component", "name", "bus"], sort=False)["port"]
        .min()
    )
    df = df.T.groupby(level=["component", "name", "bus"]).sum().T
    return df, ports.reindex(df.columns).to_numpy()


def _get_opex(n: Network, cost_types: list[str]) -> pd.DataFrame:
    """Get operational costs per component and snapshot [currency].

    Index is (component, name), columns are the snapshots.
    """
    df_opex = pd.DataFrame(columns=n.snapshots, dtype=float)
    for cost_type in cost_types:
        opex = n.statistics.opex(
            groupby="name",
            groupby_time=False,
            cost_types=cost_type,
            nice_names=False,
            drop_zero=False,
            round=0,
        )
        if not isinstance(opex, pd.DataFrame) or opex.empty:
            continue
        # Without aggregation over time, the costs are given per hour.
        opex = opex.reindex(columns=n.snapshots).fillna(0.0).astype(float)
        if cost_type not in UNWEIGHTED_COST_TYPES:
            opex = opex.mul(n.snapshot_weightings.objective, axis=1)
        df_opex = opex if df_opex.empty else df_opex.add(opex, fill_value=0.0)
    return df_opex


def _get_local_cost(
    n: Network,
    energy: np.ndarray,
    primary_supply: np.ndarray,
    unit_flows: dict[tuple[str, str], list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Get the local supply costs and the holding costs of storages [currency].

    Capital costs, fixed operation and maintenance costs and unit commitment costs are
    spread over all primary supply of a component. Dispatch costs are attributed to the
    primary supply of the same snapshot. Storages add dispatch costs of snapshots
    without discharge and holding costs to the costs of the held energy.
    """
    fixed_cost = pd.concat(
        [
            n.statistics.capex(
                groupby="name", nice_names=False, drop_zero=False, round=0
            ),
            n.statistics.fom(
                groupby="name", nice_names=False, drop_zero=False, round=0
            ),
            _get_opex(n, COMMITMENT_COST_TYPES).sum(axis=1),
        ]
    )
    fixed_cost = fixed_cost.groupby(level=[0, 1]).sum().astype(float)
    opex_dispatch = _get_opex(n, DISPATCH_COST_TYPES)
    opex_holding = _get_opex(n, STORAGE_HOLDING_COST_TYPES)

    def get_row(df: pd.DataFrame, unit: tuple[str, str]) -> np.ndarray:
        if unit not in df.index:
            return np.zeros(len(n.snapshots))
        return df.loc[unit].to_numpy(dtype=float)

    primary_energy = np.where(primary_supply, energy, 0.0)
    local_cost = np.zeros_like(energy)
    holding_cost = np.zeros_like(energy)
    unallocated = 0.0
    for unit, flows in unit_flows.items():
        if (cost := fixed_cost.get(unit, 0.0)) != 0:
            if (unit_energy := primary_energy[:, flows].sum()) > EPS:
                local_cost[:, flows] += primary_energy[:, flows] * cost / unit_energy
            else:
                unallocated += cost

        dispatch_cost = get_row(opex_dispatch, unit)
        for i in flows:
            local_cost[:, i] += np.where(primary_supply[:, i], dispatch_cost, 0.0)
        if unit[0] in STORAGE_ATTRS:
            # Storages have a single flow.
            i = flows[0]
            holding_cost[:, i] += np.where(primary_supply[:, i], 0.0, dispatch_cost)
            holding_cost[:, i] += get_row(opex_holding, unit)

    if unallocated:
        logger.info(
            "Fixed costs of %.2f currency are not allocated, as the components do not "
            "supply any energy.",
            unallocated,
        )
    return local_cost, holding_cost


def _get_storage_tracking(
    n: Network, c: str, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the tracking order and the held energy of each snapshot [MWh].

    Returns the snapshot positions in tracking order, the held energy left after the
    standing loss of each snapshot (before charging and discharging), and the held energy
    at the end of each snapshot.

    For a cyclic storage, the tracking starts after the first snapshot at the minimum
    energy level and only the energy above the minimum is tracked, so it starts empty.
    For a non-cyclic storage, the tracking starts at the first snapshot and the initial
    energy is free of costs.
    """
    level_attr, cyclic_attr, initial_attr = STORAGE_ATTRS[c]
    n_snapshots = len(n.snapshots)
    static = n.c[c].static
    dynamic = n.c[c].dynamic[level_attr]
    # Energy level at the end of each snapshot [MWh].
    level = (
        dynamic[name].reindex(n.snapshots).fillna(0.0).to_numpy(dtype=float)
        if name in dynamic
        else np.zeros(n_snapshots)
    )

    if static.at[name, cyclic_attr]:
        # The level at the start of the first snapshot is the level at the end of the
        # last snapshot.
        level_before = np.concatenate((level[-1:], level[:-1]))
        start = int(np.argmin(level_before))
        baseline = float(level.min())
    else:
        start = 0
        initial = float(static.at[name, initial_attr])
        level_before = np.concatenate(([initial], level[:-1]))
        baseline = min(initial, float(level.min()))

    # Share of the energy level that remains after the standing loss of each snapshot.
    standing_loss = n.get_switchable_as_dense(c, "standing_loss")[name]
    retained = (1 - standing_loss.to_numpy(dtype=float)) ** (
        n.snapshot_weightings.stores.to_numpy(dtype=float)
    )
    held_available = np.clip(retained * level_before - baseline, 0.0, None)

    positions = np.concatenate((np.arange(start, n_snapshots), np.arange(start)))
    held_after = level[positions] - baseline
    held_available = held_available[positions]
    if static.at[name, cyclic_attr]:
        # The tracking starts empty at the minimum level.
        held_available[0] = 0.0
    return positions, held_available, held_after
