# Wells

## Well output (time_data)

### Well Time Data Units
* Time: day
* BHP: bar
* BHT: Kelvin
* Molar rate: kmol/day
* Mass rate: kg/day
* Volumetric rate: m^3/day
* Advective heat rate: kJ/day


### Well Time Data Key Guide

This guide explains how to access stored well time data (series) using the appropriate dictionary keys.\
Each key follows a **specific naming convention** depending on the type of time data you want to have.

---

### Key Naming Formats

#### 1. **Perforation Rates**

For accessing time-series rates **at individual perforations**:

`well_<well_name>_perf_<perf_number>_<rate_type>_rate_<phase_or_component_name> `

Where:

* `<well_name>` is the name of the well.
* `<perf_number>` is the perforation index (e.g., 0, 1, 2, ...).
* `<rate_type>` is one of:
  * `molar`
  * `mass`
  * `volumetric`
  * `advective_heat`
* `<phase_or_component_name>` is taken from the phase names (e.g., `water`, `gas`, `oil`) or component names (e.g., `CO2`, `C1`).


Examples:

* `well_I1_perf_0_molar_rate_gas`
* `well_P1_perf_2_mass_rate_C1`
* `well_I2_perf_1_advective_heat_rate_water`


---

#### 2. **Well Rates (Calculated by Summing Rates Over All Perforations)**

For accessing **total well rates by summing rates over all perforations**:

`well_<well_name>_<rate_type>_rate_<phase_or_component_name>_by_sum_perfs `

Examples:

* `well_I1_molar_rate_gas_by_sum_perfs`
* `well_P1_mass_rate_C1_by_sum_perfs`
* `well_I2_advective_heat_rate_water_by_sum_perfs`


---

#### 3. Well rates (Calculated at **Wellhead)**

For accessing **rates at the wellhead**:

`well_<well_name>_<rate_type>_rate_<phase_or_component_name>_at_wh `

Examples:

* `well_I1_molar_rate_gas_at_wh`
* `well_P1_mass_rate_C1_at_wh`
* `well_I2_advective_heat_rate_water_at_wh`


---

#### 4. **Bottom-Hole Pressure (BHP) and Temperature (BHT)**

For accessing **Bottom-Hole Pressure** and **Bottom-Hole Temperature**:

`well_<well_name>_BHP
well_<well_name>_BHT
`

Examples:

* `well_I1_BHP`
* `well_P2_BHT`


---

### Notes

* Molar rates are not valid for dead-oil models.
* `BHT` is available for thermal scenarios; otherwise, it is a constant set to the initial reservoir temperature.
* `advective_heat` is only available for thermal scenarios.
* For advective heat rate, the following dead state (reference state for energy injection or production) is used:
  * P = 1 atm;
  * T = 15 deg C.
  
## Multi-segment wells

There is a `multi_segment` argument in `add_perforation` function.

If `multi_segment = False`

When using multiple perforations with `multi_segment = False`, be cautious that the wellbore will consist of only **one segment**, and fluid flows between the wellbore segment and multiple reservoir cells through perforations.

![image.png](images/wells/add_perf_mswell_false.png){width=913 height=81}

If depths of reservoir cells are set correctly, there will be flow at perforations due to gravity, which does **not happen in reality**.

An example is given below with two perforations:

![image.png](images/wells/mswell_false.png)

Therefore, to avoid flow due to gravity at perforations, it is better to always use `multi_segment = True`. In addition, the depths of the reservoir cells need to be specified correctly.

If `multi_segment = True`

When using multiple perforations with `multi_segment = True`, the wellbore consists of **multiple segments** (one segment in front of each perforated reservoir cell), and this is a more accurate representation of fluid flow in the wellbore.  
Be careful to use the correct depths for reservoir cells; otherwise, fluid flow between the reservoir and wellbore will be inaccurate.

![image.png](images/wells/add_perf_mswell_true.png){width=892 height=79}

An example is given below with two perforations:

![image.png](images/wells/mswell_true.png)