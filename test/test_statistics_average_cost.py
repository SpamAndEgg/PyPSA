# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.statistics.average_cost import calculate_average_cost


def load_cost(n):
    """Total cost paid by all loads."""
    energy, cost = calculate_average_cost(n)
    loads = energy.columns[energy.columns.get_level_values("component") == "Load"]
    return float((-energy[loads] * cost[loads]).sum().sum())


def system_cost(n):
    return float(n.statistics.system_cost(round=0).sum())


def network_with_buses(snapshots, *buses):
    n = pypsa.Network(snapshots=snapshots)
    for bus in buses:
        n.add("Bus", bus)
    return n


def test_link_efficiency_marginal_cost_and_free_secondary_output():
    """Costs of the input and the link go to bus1, the output at bus2 is free.

    The link withdraws 1.25 MWh electricity at 10 currency/MWh and its marginal cost of
    2 currency/MWh applies to the withdrawal at bus0: (1.25 * 10 + 1.25 * 2) / 1 = 15.
    """
    n = network_with_buses(range(2), "electricity", "hydrogen", "oxygen")
    n.add("Generator", "grid", bus="electricity", p_nom=10, marginal_cost=10)
    n.add(
        "Link",
        "electrolysis",
        bus0="electricity",
        bus1="hydrogen",
        bus2="oxygen",
        efficiency=0.8,
        efficiency2=0.1,
        marginal_cost=2,
        p_nom=10,
    )
    n.add("Load", "hydrogen demand", bus="hydrogen", p_set=1)
    n.add("Load", "oxygen demand", bus="oxygen", p_set=0.125)
    n.optimize()

    df = n.statistics.average_cost(round=10)
    assert df.at[
        ("Load", "hydrogen demand", "hydrogen"), "withdrawal"
    ] == pytest.approx(15)
    assert df.at[("Load", "oxygen demand", "oxygen"), "withdrawal"] == pytest.approx(0)
    assert np.isnan(df.at[("Link", "electrolysis", "oxygen"), "supply"])
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_process():
    """A process passes its input costs and its marginal cost on to its output."""
    n = network_with_buses(range(2), "electricity", "heat")
    n.add("Generator", "grid", bus="electricity", p_nom=10, marginal_cost=10)
    n.add(
        "Process",
        "boiler",
        bus0="electricity",
        bus1="heat",
        rate0=-1,
        rate1=0.8,
        marginal_cost=2,
        p_nom=10,
    )
    n.add("Load", "heat demand", bus="heat", p_set=1)
    n.optimize()

    df = n.statistics.average_cost(round=10)
    assert df.at[("Load", "heat demand", "heat"), "withdrawal"] == pytest.approx(15)
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_cyclic_energy_flow():
    """Energy circulating between bus 1 and bus 2 passes on all costs.

    Each snapshot, the load on bus 1 (1 MWh) and link 1 (0.25 MWh) withdraw from bus 1,
    which is supplied by gen 1 (0.65 MWh at 1 currency/MWh), link 4 (0.1 MWh of free
    energy) and link 2 (0.5 MWh from bus 2). Bus 2 is supplied by link 1 (0.25 MWh plus
    a marginal cost of 1 currency/MWh) and link 3 (0.25 MWh of free energy):

        c2 * 0.5 = 0.25 * c1 + 0.25
        c1 * 1.25 = 0.65 + 0.5 * c2  ->  c1 = 0.9
    """
    n = network_with_buses(range(3), "bus 1", "bus 2", "bus 3")
    n.add("Generator", "gen 1", bus="bus 1", p_nom=1, marginal_cost=1)
    n.add("Generator", "gen 3", bus="bus 3", p_nom=1)
    n.add("Load", "load", bus="bus 1", p_set=1)
    n.add(
        "Link",
        "link 1",
        bus0="bus 1",
        bus1="bus 2",
        p_nom=1,
        p_min_pu=0.25,
        marginal_cost=1,
    )
    n.add("Link", "link 2", bus0="bus 2", bus1="bus 1", p_nom=1, p_min_pu=0.5)
    n.add(
        "Link",
        "link 3",
        bus0="bus 3",
        bus1="bus 2",
        p_nom=1,
        p_min_pu=0.25,
        p_max_pu=0.25,
    )
    n.add("Link", "link 4", bus0="bus 3", bus1="bus 1", p_nom=0.1)
    n.optimize()

    df = n.statistics.average_cost(round=10)
    assert df.at[("Load", "load", "bus 1"), "withdrawal"] == pytest.approx(0.9)
    assert df.at[("Link", "link 2", "bus 2"), "withdrawal"] == pytest.approx(0.95)
    assert load_cost(n) == pytest.approx(system_cost(n))


@pytest.mark.parametrize("component", ["StorageUnit", "Store"])
def test_storage(component):
    """Energy stored in the cheap first snapshot is discharged in the second one."""
    n = network_with_buses(range(2), "electricity")
    n.add("Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[10, 30])
    n.add("Load", "load", bus="electricity", p_set=1)
    if component == "StorageUnit":
        n.add(
            "StorageUnit",
            "storage",
            bus="electricity",
            p_nom=1,
            max_hours=1,
            cyclic_state_of_charge=True,
        )
    else:
        n.add("Store", "storage", bus="electricity", e_nom=1, e_cyclic=True)
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(cost[("Load", "load", "electricity")], [10, 10])
    np.testing.assert_allclose(cost[(component, "storage", "electricity")], [10, 10])


def test_storage_efficiency_and_marginal_cost():
    """Storage losses and the dispatch marginal cost raise the cost of discharged energy.

    1.5625 MWh charged at 10 currency/MWh are discharged as 1 MWh, which costs
    15.625 currency/MWh plus the marginal cost of 1 currency/MWh.
    """
    n = network_with_buses(range(2), "electricity")
    n.add("Generator", "grid", bus="electricity", p_nom=3, marginal_cost=[10, 30])
    n.add("Load", "load", bus="electricity", p_set=1)
    n.add(
        "StorageUnit",
        "storage",
        bus="electricity",
        p_nom=2,
        max_hours=1,
        efficiency_store=0.8,
        efficiency_dispatch=0.8,
        marginal_cost=1,
        cyclic_state_of_charge=True,
    )
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(cost[("Load", "load", "electricity")], [10, 16.625])
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_standing_loss_non_cyclic():
    """Energy held longer suffers more standing losses and costs more.

    1 MWh charged for 10 currency loses 10 % per hour. In hour 3, half of the remaining
    0.81 MWh is discharged, so half of the account pays for 0.405 MWh: 12.35 currency/MWh.
    The other half loses another 10 % and is discharged in hour 4: 5 / 0.3645 = 13.72.
    """
    n = network_with_buses(range(4), "electricity")
    n.add(
        "Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[10, 50, 50, 50]
    )
    n.add("Load", "load", bus="electricity", p_set=[0, 0, 0.405, 0.3645])
    n.add(
        "StorageUnit",
        "battery",
        bus="electricity",
        p_nom=1,
        max_hours=1,
        standing_loss=0.1,
        cyclic_state_of_charge=False,
    )
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(
        cost[("Load", "load", "electricity")].iloc[2:], [10 / 0.81, 10 / 0.729]
    )
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_standing_loss_cyclic_wraps_around():
    """A cyclic storage charged at the end of the horizon supplies its beginning.

    The storage charges 1 MWh for 10 currency in hour 4 and discharges it in hours 1 and
    2, with 10 % standing loss per hour: 5 / 0.45 = 11.11 and 5 / 0.405 = 12.35.
    """
    n = network_with_buses(range(4), "electricity")
    n.add(
        "Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[50, 50, 50, 10]
    )
    n.add("Load", "load", bus="electricity", p_set=[0.45, 0.405, 0, 0])
    n.add(
        "StorageUnit",
        "battery",
        bus="electricity",
        p_nom=1,
        max_hours=1,
        standing_loss=0.1,
        cyclic_state_of_charge=True,
    )
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(
        cost[("Load", "load", "electricity")].iloc[:2], [10 / 0.9, 10 / 0.81]
    )
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_electrolyser_with_capital_cost_and_storage():
    """Hydrogen from an electrolyser with capital cost, partly stored with losses.

    Hour 1: 1.25 MWh electricity at 10 currency/MWh plus 2.5 capital cost make 1 MWh
    hydrogen for 15 currency/MWh. Half is stored and loses 10 % until hour 2:
    7.5 / 0.45 = 16.67 currency/MWh.
    """
    n = network_with_buses(range(2), "electricity", "hydrogen")
    n.add("Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[10, 50])
    n.add(
        "Link",
        "electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        efficiency=0.8,
        p_nom=1.25,
        capital_cost=2,
    )
    n.add(
        "StorageUnit",
        "hydrogen storage",
        bus="hydrogen",
        p_nom=1,
        max_hours=1,
        standing_loss=0.1,
    )
    n.add("Load", "hydrogen demand", bus="hydrogen", p_set=[0.5, 0.45])
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(
        cost[("Load", "hydrogen demand", "hydrogen")], [15, 7.5 / 0.45]
    )
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_marginal_cost_storage():
    """Costs for holding energy are paid when the energy is discharged."""
    n = network_with_buses(range(2), "electricity")
    n.add("Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[10, 30])
    n.add("Load", "load", bus="electricity", p_set=1)
    n.add(
        "Store",
        "storage",
        bus="electricity",
        e_nom=1,
        e_cyclic=True,
        marginal_cost_storage=2,
    )
    n.optimize()

    assert n.statistics.opex(cost_types="marginal_cost_storage", round=0).sum() > 0
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_non_cyclic_storage():
    """The initial energy of a non-cyclic storage is free."""
    n = network_with_buses(range(2), "electricity")
    n.add("Generator", "grid", bus="electricity", p_nom=2, marginal_cost=[10, 30])
    n.add("Load", "load", bus="electricity", p_set=1)
    n.add(
        "StorageUnit",
        "storage",
        bus="electricity",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
    )
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(
        cost[("Load", "load", "electricity")], [10, 0], atol=1e-9
    )


def test_capital_cost():
    """Capital costs are spread over the energy supplied in the modelled snapshots."""
    n = network_with_buses(range(2), "electricity")
    n.add(
        "Generator",
        "grid",
        bus="electricity",
        p_nom=2,
        marginal_cost=5,
        capital_cost=10,
    )
    n.add("Load", "load", bus="electricity", p_set=1)
    n.optimize()

    df = n.statistics.average_cost(round=10)
    assert df.at[("Load", "load", "electricity"), "withdrawal"] == pytest.approx(15)
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_snapshot_weightings():
    """Operational costs per snapshot are weighted by the snapshot weightings."""
    n = network_with_buses(range(4), "electricity")
    n.snapshot_weightings.loc[:, :] = 0.25
    n.add(
        "Generator", "grid", bus="electricity", p_nom=10, marginal_cost=[10, 20, 30, 40]
    )
    n.add("Load", "load", bus="electricity", p_set=4)
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(cost[("Load", "load", "electricity")], [10, 20, 30, 40])
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_stand_by_cost():
    """Unit commitment costs are spread over all energy supplied by the component.

    Stand-by costs of 5 currency/h are 10 currency for 2 MWh hydrogen, so the hydrogen
    costs 10 / 0.8 + 5 = 17.5 currency/MWh.
    """
    n = network_with_buses(range(2), "electricity", "hydrogen")
    n.add("Generator", "grid", bus="electricity", p_nom=10, marginal_cost=10)
    n.add(
        "Link",
        "electrolysis",
        bus0="electricity",
        bus1="hydrogen",
        efficiency=0.8,
        p_nom=10,
        committable=True,
        stand_by_cost=5,
    )
    n.add("Load", "hydrogen demand", bus="hydrogen", p_set=1)
    n.optimize()

    df = n.statistics.average_cost(round=10)
    assert df.at[
        ("Load", "hydrogen demand", "hydrogen"), "withdrawal"
    ] == pytest.approx(17.5)
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_start_up_and_shut_down_cost():
    """Start-up and shut-down costs occur once per event and are spread like capex.

    The electrolyser starts once (10 currency) and shuts down once (4 currency) for
    2 MWh hydrogen, so the hydrogen costs 10 / 0.8 + 14 / 2 = 19.5 currency/MWh.
    """
    n = network_with_buses(range(3), "electricity", "hydrogen")
    n.add("Generator", "grid", bus="electricity", p_nom=10, marginal_cost=10)
    n.add(
        "Link",
        "electrolysis",
        bus0="electricity",
        bus1="hydrogen",
        efficiency=0.8,
        p_nom=2,
        p_min_pu=0.5,
        committable=True,
        up_time_before=0,
        start_up_cost=10,
        shut_down_cost=4,
    )
    n.add("Load", "hydrogen demand", bus="hydrogen", p_set=[1, 1, 0])
    n.optimize()

    assert n.statistics.opex(cost_types="start_up_cost", round=0).sum() == 10
    assert n.statistics.opex(cost_types="shut_down_cost", round=0).sum() == 4
    df = n.statistics.average_cost(round=10)
    assert df.at[
        ("Load", "hydrogen demand", "hydrogen"), "withdrawal"
    ] == pytest.approx(19.5)
    assert load_cost(n) == pytest.approx(system_cost(n))


def test_meshed_ac_dc_network(ac_dc_solved):
    """Loads pay all system costs in a meshed network with lines and links."""
    n = ac_dc_solved
    assert load_cost(n) == pytest.approx(system_cost(n))

    df = n.statistics.average_cost()
    assert list(df.columns) == ["withdrawal", "supply"]
    assert df.index.names == ["component", "name", "bus"]
    assert set(df.index.get_level_values("component")) >= {
        "Generator",
        "Line",
        "Link",
        "Load",
    }

    df_time = n.statistics.average_cost(groupby_time=False)
    assert list(df_time.columns) == list(n.snapshots)

    df_dc = n.statistics.average_cost(bus_carrier="DC")
    assert (n.buses.carrier[df_dc.index.get_level_values("bus")] == "DC").all()


def test_not_supported(ac_dc_network, ac_dc_stochastic):
    with pytest.raises(ValueError, match="solved network"):
        ac_dc_network.statistics.average_cost()
    with pytest.raises(NotImplementedError, match="stochastic"):
        ac_dc_stochastic.statistics.average_cost()
    collection = pypsa.NetworkCollection(
        [ac_dc_network, ac_dc_network.copy()], index=pd.Index(["a", "b"])
    )
    with pytest.raises(NotImplementedError, match="network collections"):
        collection.statistics.average_cost()


def test_not_part_of_statistics_summary(ac_dc_solved, monkeypatch):
    """The expensive calculation must not run for `n.statistics()`."""

    def fail(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr("pypsa.statistics.average_cost.calculate_average_cost", fail)
    assert "Average Cost" not in ac_dc_solved.statistics().columns


def test_loads_pay_marginal_cost_with_time_varying_input():
    snapshots = pd.RangeIndex(3)
    n = network_with_buses(snapshots, "electricity", "heat")
    n.add("Generator", "grid", bus="electricity", p_nom=10, marginal_cost=[1, 2, 3])
    n.add(
        "Link",
        "heat pump",
        bus0="electricity",
        bus1="heat",
        efficiency=[2, 3, 4],
        p_nom=10,
    )
    n.add("Load", "heat demand", bus="heat", p_set=1)
    n.optimize()

    _, cost = calculate_average_cost(n)
    np.testing.assert_allclose(
        cost[("Load", "heat demand", "heat")], [0.5, 2 / 3, 0.75]
    )
