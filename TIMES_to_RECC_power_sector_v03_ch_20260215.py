"""
by Christian Hauenstein, Marcel Geller, February 2026

Import TIMES power sector results data and prepare in-, outflow and stock parameter files:
1) Import TIMES results and lifetime data 
and create arrays for stock, inflow and outflow by scenario, region, technology and year (2015-2060),
and for lifetimes by scenario, technology and year (2015-2060) 
    1.1) For Lifetimes: differentiat between "Base" and "Slow" scenario where available, otherwise use "Base" values for all scenarios
    1.2) Create outflow array based on stock and inflow balance (outflow = inflow - stock change)

2) Convert inflow, outflow and lifetime arrays to annual values (from 5-year time steps) 

3) Now create age-cohort data for stock based on outflows of exisiting and nuclear capacities
3.1) Create initial 2015 stock from 2016-2060 outflows of existing and nuclear capacities
3.2) Recreate 2020 stock based on above stock_2015_srpc_ex_nuc to derive 2016-2020 inflows\
# - outflows of existing and nuclear capacities during 2016-2020 (2016-2020 outflows, pre 2016 age-cohorts) \
# + inflows based on outflows of existing capacities during 2016-2020 (post 2020 outflows with age-cohorts 2016-2020) \
# + inflows based on VAR_NCap for nuclear during 2016-2020 \
# + difference of this new 2020 stock and TIMES 2020 stock, assigned to age-cohorts 2016-2020 (all other in 2060 remaining capacities already added to 2015 stock)
3.3) Create outflow age-cohort array for all other technologies (non-existing and non-nuclear) based on inflows and lifetimes

4.) Export 2015 stock with age-cohorts, annual inflows, and annual outflows with age-cohorts
mapping and aggregation of scenario, region and process names from TIMES to RECC technology and region names
"""

# Import packages
import pandas as pd
import numpy as np
#import os

##### 1) Import TIMES power sector stock and inflow data, years 2015-2060 (only 5yr time steps available)
df_TIMES_stock = pd.read_excel("Power_Sector_17_02_2026.xlsx", skiprows=2, usecols="A:C,F:O") #exclude years 2010 and 2011
df_TIMES_inflow = pd.read_excel("Power_Sector_17_02_2026.xlsx", skiprows=2, usecols="A:C,R:AA") #exclude 2011
df_TIMES_lt = pd.read_excel("Lifetimes_TIMES_RECC_adjusted_V1.0_17_02_2026.xlsx", skiprows=1)

# Replace NaN entries with 0 in year columns 2015-2060 (if present)
df_TIMES_stock.iloc[:, 3:] = df_TIMES_stock.iloc[:, 3:].fillna(0)
df_TIMES_inflow.iloc[:, 3:] = df_TIMES_inflow.iloc[:, 3:].fillna(0)
df_TIMES_lt.iloc[:, 1:] = df_TIMES_lt.iloc[:, 1:].fillna(0)

##### Create 4-D NumPy arrays: Scenario x Region x Process x Year for stock and inflow #####
scenarios = df_TIMES_stock['Scenario'].astype(str).unique().tolist()
regions = df_TIMES_stock['Region'].astype(str).unique().tolist()
processes = df_TIMES_stock['Process'].astype(str).unique().tolist()
years = [c for c in df_TIMES_stock.columns if (isinstance(c, str) and c.isdigit() and 2015 <= int(c) <= 2060) or (isinstance(c, int) and 2015 <= c <= 2060)]

# Ensure df_TIMES_inflow year-column headers (columns 4-13) match the `years` list
# Columns 0-2 are kept (Scenario, Region, Process); replace the next 10 columns
if len(years) == len(df_TIMES_inflow.columns[3:13]):
    df_TIMES_inflow.columns = list(df_TIMES_inflow.columns[:3]) + list(years)
else:
    # fallback: attempt in-place assignment of the slice if lengths align
    df_TIMES_inflow.columns.values[3:3+len(years)] = list(years)

TIMES_stock_srpt = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)
TIMES_inflow_srpt = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)

sc_idx = {v: i for i, v in enumerate(scenarios)}
rg_idx = {v: i for i, v in enumerate(regions)}
pr_idx = {v: i for i, v in enumerate(processes)}

# Fill TIMES_stock from df_TIMES_stock year columns
for _, row in df_TIMES_stock.iterrows():
    s = str(row['Scenario'])
    r = str(row['Region'])
    p = str(row['Process'])
    try:
        vals = row[years].astype(float).values
    except Exception:
        vals = np.array([float(x) if pd.notna(x) else 0.0 for x in row[years]])
    TIMES_stock_srpt[sc_idx[s], rg_idx[r], pr_idx[p], :] = vals

# Fill TIMES_inflow from df_TIMES_inflow year columns
for _, row in df_TIMES_inflow.iterrows():
    s = str(row['Scenario'])
    r = str(row['Region'])
    p = str(row['Process'])
    try:
        vals = row[years].astype(float).values
    except Exception:
        vals = np.array([float(x) if pd.notna(x) else 0.0 for x in row[years]])
    TIMES_inflow_srpt[sc_idx[s], rg_idx[r], pr_idx[p], :] = vals

# Expose index mapping for later reference
TIMES_stock_index = {
    'Scenario': scenarios,
    'Region': regions,
    'Process': processes,
    'Year_columns': list(years),
}

##### 1.1) Create 3-D NumPy array: Scenario x Process x Year for lifetimes #####
TIMES_lifetime_spt = np.zeros((len(scenarios), len(processes), len(years)), dtype=float)

# df_TIMES_lt only has entries for a Base and a Slow scenario 
# first fill in lt values for all scenarios based Base scenario values.
# in case of missing values in Base scenario, fill with 50 years (individual nuclear plants missing)
# Populate TIMES_lifetime_spt using the 'Base' scenario from df_TIMES_lt
# If a process is missing in df_TIMES_lt, use 50 years as fallback.

# Build mapping of process -> lifetime values for the Base scenario
base_mask = df_TIMES_lt['scenario'].astype(str).str.strip().str.lower() == 'base'
df_base = df_TIMES_lt[base_mask]
# Prepare year column names as strings to index df_TIMES_lt
year_cols = years
lt_map = {}
for _, row in df_base.iterrows():
    proc = str(row['process'])
    try:
        vals = row[year_cols].astype(float).values
    except Exception:
        vals = np.array([float(x) if pd.notna(x) else 50.0 for x in row[year_cols]])
    lt_map[proc] = vals

# Fill TIMES_lifetime_spt for every scenario/process using Base values
for i_s, s in enumerate(scenarios):
    for i_p, p in enumerate(processes):
        vals = None
        # direct match
        if p in lt_map:
            vals = lt_map[p]
        else:
            # case-insensitive match fallback
            for k in lt_map.keys():
                if k.lower() == str(p).lower():
                    vals = lt_map[k]
                    break
        if vals is None:
            TIMES_lifetime_spt[i_s, i_p, :] = 50.0
        else:
            if len(vals) == len(years):
                TIMES_lifetime_spt[i_s, i_p, :] = vals
            else:
                arr = np.full(len(years), 50.0)
                L = min(len(vals), len(years))
                arr[:L] = vals[:L]
                TIMES_lifetime_spt[i_s, i_p, :] = arr

# Overwrite lifetimes for scenarios whose name contains 'slow' or 'nsc'
# using values from the 'Slow' scenario in df_TIMES_lt, but only for
# processes present in the 'Slow' scenario. Do not overwrite otherwise.
slow_scenarios = [s for s in scenarios if ('slow' in str(s).lower()) or ('nsc' in str(s).lower()) or ('ncs' in str(s).lower())] #'01_ssp2nsc', '01_ssp2slow', '02_19taxncs' and '02_19taxslow'

slow_mask = df_TIMES_lt['scenario'].astype(str).str.strip().str.lower() == 'slow'
df_slow = df_TIMES_lt[slow_mask]
year_cols = years
slow_map = {}
for _, row in df_slow.iterrows():
    proc = str(row['process'])
    try:
        vals = row[year_cols].astype(float).values
    except Exception:
        vals = np.array([float(x) if pd.notna(x) else np.nan for x in row[year_cols]])
    slow_map[proc] = vals

for i_s, s in enumerate(scenarios):
    name = str(s).lower()
    if ('slow' in name) or ('nsc' in name) or ('ncs' in name): #'01_ssp2nsc', '01_ssp2slow', '02_19taxncs' and '02_19taxslow'
        for i_p, p in enumerate(processes):
            vals = None
            if p in slow_map:
                vals = slow_map[p]
            else:
                for k in slow_map.keys():
                    if k.lower() == str(p).lower():
                        vals = slow_map[k]
                        break
            if vals is None:
                # process not present in Slow scenario — do not overwrite
                continue
            # assign values, preserving existing entries if length mismatch
            if len(vals) == len(years):
                TIMES_lifetime_spt[i_s, i_p, :] = vals
            else:
                L = min(len(vals), len(years))
                TIMES_lifetime_spt[i_s, i_p, :L] = vals[:L]

# Expose index mapping for later reference
TIMES_lifetime_index = {
    'Scenario': scenarios,
    'Process': processes,
    'Year_columns': list(years),
}


##### 1.2) Compute outflows from stock and inflow (for all technologies, including new technologies for comprehensiveness here) ######
TIMES_outflow_srpt = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)
# no outflows in 2015
TIMES_outflow_srpt[:,:,:,1:] = TIMES_stock_srpt[:,:,:,:-1] - (TIMES_stock_srpt[:,:,:,1:] - TIMES_inflow_srpt[:,:,:,1:])
#  do not remove negative outflows here, only during processing later
'''# set negative outflows to 0 ()
TIMES_outflow_srpt_negative = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)
TIMES_neg_outflow_mask = TIMES_outflow_srpt < 0
TIMES_outflow_srpt_negative[TIMES_neg_outflow_mask] = TIMES_outflow_srpt[TIMES_neg_outflow_mask]

TIMES_outflow_srpt[TIMES_outflow_srpt < 0] = 0
'''

##### 2) Convert inflow, outflow and lifetime arrays to annual values (from 5-year time steps)  #####
# For stock we only use 2015 stock and assign age-cohort distribution based on outflows
Nt = 46 # number of years from 2015 to 2060 inclusive
Nc = 161 # number of age-cohorts from 1900 to 2060 inclusive
SwitchTime = 116 # index of year 2016 in Nc (first year with in- and outflows in RECC)

TIMES_inflow_srpt_annual = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_outflow_srpt_annual = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_lifetime_spt_annual = np.zeros((len(scenarios), len(processes), Nt), dtype=float)

# 2015 lifetime values unchanged:
TIMES_lifetime_spt_annual[:,:,0] = TIMES_lifetime_spt[:,:,0]

# 2016-2060 values: assign same value for each of the 5 years in the time step, and divide by 5 to get annual values for inflows and outflows.
counter = 0
for T in range(1,len(years)): # T for five year time steps, start from 1 as no inflow/outflow in 2015
    TIMES_inflow_srpt_annual[:,:,:,counter+T:counter+T+5] = TIMES_inflow_srpt[:,:,:,T][:,:,:,np.newaxis] / 5.0
    TIMES_outflow_srpt_annual[:,:,:,counter+T:counter+T+5] = TIMES_outflow_srpt[:,:,:,T][:,:,:,np.newaxis] / 5.0
    # For lifetimes assign the same value for each of the 5 years in the time step
    TIMES_lifetime_spt_annual[:,:,counter+T:counter+T+5] = TIMES_lifetime_spt[:,:,T][:,:,np.newaxis]
    # Update counter for next iteration
    counter += 4

#### 3) Now create age-cohort data for stock based on outflows of exisiting and nuclear capacities
# ("existing" capacities have no (VAR_NCap) inflow in TIMES; nuclear does not differentiate "exisiting" and new, but is 
# reactor specific, except for general process "nuclear liftime extension"
# all nuclear outflows during period 2016-2060 have been added before 2016, and all new inflows have lifetime beyond 2060).
# Inflows 2016-2020 are based on outflows of "existing" capacities (only PV and Wind, all other have lifetime > 45yr) and VAR_NCap for nuclear.
# Then add difference of 
# For remaining difference of 2020 stock created based on outflows, and TIMES 2020 stock, assign age-cohort 1900 and add to 2015 stock.
# For age-cohorts assigned based on outflows and lifetime, set oldest possible age-cohort to 1901, to differentiat from capacities assigned via 2020 delta.

# Create an array with the same process dimension as `processes`, but
# keep values only for processes that contain 'existing' or 'nuclear'.
proc_mask = [isinstance(p, str) and ("existing" in p.lower() or "nuclear" in p.lower()) for p in processes]
proc_idx_ex_nuc = [i for i, m in enumerate(proc_mask) if m]

# Array shape: Scenario x Region x Processes x Nt (same process ordering as `processes`)
TIMES_outflow_srpt_annual_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_inflow_srpt_annual_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_stock_srpt_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)
TIMES_stock_srpt_all_other = np.zeros((len(scenarios), len(regions), len(processes), len(years)), dtype=float)
TIMES_outflow_srpt_annual_all_other = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_inflow_srpt_annual_all_other = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)


# Copy values for matching processes; non-matching processes remain zero
for i_s in range(len(scenarios)):
    for i_r in range(len(regions)):
        for i_p in proc_idx_ex_nuc:
            TIMES_outflow_srpt_annual_ex_nuc[i_s, i_r, i_p, :] = TIMES_outflow_srpt_annual[i_s, i_r, i_p, :]
            TIMES_inflow_srpt_annual_ex_nuc[i_s, i_r, i_p, :] = TIMES_inflow_srpt_annual[i_s, i_r, i_p, :]
            TIMES_stock_srpt_ex_nuc[i_s, i_r, i_p, :] = TIMES_stock_srpt[i_s, i_r, i_p, :]

for i_s in range(len(scenarios)):
    for i_r in range(len(regions)):
        for i_p in range(len(processes)):
            if i_p not in proc_idx_ex_nuc:
                TIMES_stock_srpt_all_other[i_s, i_r, i_p, :] = TIMES_stock_srpt[i_s, i_r, i_p, :]
                TIMES_inflow_srpt_annual_all_other[i_s, i_r, i_p, :] = TIMES_inflow_srpt_annual[i_s, i_r, i_p, :]
                TIMES_outflow_srpt_annual_all_other[i_s, i_r, i_p, :] = TIMES_outflow_srpt_annual[i_s, i_r, i_p, :]

# set negative outflows to 0 (and store negative values in separate array for reference)
TIMES_outflow_srpt_annual_ex_nuc_negative = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float)
TIMES_neg_outflow_mask = TIMES_outflow_srpt_annual_ex_nuc < 0
TIMES_outflow_srpt_annual_ex_nuc_negative[TIMES_neg_outflow_mask] = TIMES_outflow_srpt_annual_ex_nuc[TIMES_neg_outflow_mask]

TIMES_outflow_srpt_annual_ex_nuc[TIMES_outflow_srpt_annual_ex_nuc < 0] = 0 # nuclear inflows covered via VAR_Ncap

# assign age-cohorts to outflows of existing and nuclear capacities, create 2015 stock with age-cohorts
TIMES_outflow_srpt_annual_ex_nuc_c = np.zeros((len(scenarios), len(regions), len(processes), Nt, Nc), dtype=float) # age-cohort array for outflows of existing and nuclear capacities
stock_2015_srpc_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nc), dtype=float)
# 3.1) Create initial 2015 stock from 2016-2060 outflows of existing and nuclear capacities
for i_t in range(1,Nt): # start from 1 as no outflow in 2015
    for i_s in range(len(scenarios)):
        for i_p in proc_idx_ex_nuc:
            outflow = np.zeros(len(regions))
            outflow[:] = TIMES_outflow_srpt_annual_ex_nuc[i_s, :, i_p, i_t]
            lifetime = TIMES_lifetime_spt_annual[i_s, i_p, i_t]
            # Determine age-cohort of this outflow based on lifetime
            age_cohort = int(SwitchTime - 1 + i_t - lifetime)
            if age_cohort < 1:
                age_cohort = 1
            TIMES_outflow_srpt_annual_ex_nuc_c[i_s, :, i_p, i_t, age_cohort] = outflow
            if age_cohort <= (SwitchTime -1): # add all age-cohorts up to 2015 to 2015 stock
                stock_2015_srpc_ex_nuc[i_s, :, i_p, age_cohort] += outflow[:]

#Add 2015 TIMES-stock_2015_srpc_ex_nuc difference to 2015 stock (only existing and nuclear), with age-cohort 1900 (capacities without outflow before 2060)
delta_2015_ex_nuc2 = TIMES_stock_srpt_ex_nuc[:,:,:,0] - stock_2015_srpc_ex_nuc[:,:,:,:].sum(axis=3)
#delta_2015_ex_nuc2_neg_values = delta_2015_ex_nuc2.copy() # only for checking
#delta_2015_ex_nuc2_neg_values[delta_2015_ex_nuc2_neg_values > 0] = 0
# negative delta 2015 values: in all scenarios roughly the same total; relevance differs across regions
# negative delta 2015 values due to early retirements (thus, part of 2015 stock based on outflows, but not of TIMES 2015 stock), and nuclear lifetime extension.
# set any negative values in delta_2015_ex_nuc2 to 0 (mostly an issue of nuclear lifetime extension; not relevant as not taken over into RECC technologies later)
delta_2015_ex_nuc2[delta_2015_ex_nuc2 < 0] = 0
stock_2015_srpc_ex_nuc[:,:,:,0] += delta_2015_ex_nuc2[:,:,:] # add any remaining difference to age-cohort 1900 (index 0)
# Check: Delta_2015_final = stock_2015_srpc_ex_nuc.sum() - TIMES_stock_srpt_ex_nuc[:,:,:,0].sum()
# Delta_2015_final >> 0 because stock_2015_srpc_ex_nuc contains early retired capacities that actually were installed after 2015, but age-cohorts assigned to outflows indicate 2015 existance.
# For now (Feb 16, 2026), we accept this overestimation of the 2015 stock.

# 3.2) Recreate 2020 stock based on above stock_2015_srpc_ex_nuc to derive 2016-2020 inflows\
# - outflows of existing and nuclear capacities during 2016-2020 (2016-2020 outflows, pre 2016 age-cohorts) \
# + inflows based on outflows of existing capacities during 2016-2020 (post 2020 outflows with age-cohorts 2016-2020) \
# + inflows based on VAR_NCap for nuclear during 2016-2020 \
# + difference of this new 2020 stock and TIMES 2020 stock, assigned to age-cohorts 2016-2020 (all other in 2060 remaining capacities already added to 2015 stock)
stock_2020_srpc_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nc), dtype=float)
stock_2020_srpc_ex_nuc = stock_2015_srpc_ex_nuc.copy() # start with 2015 stock
stock_2020_srpc_ex_nuc[:,:,:,:] -= TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,1:6,:].sum(axis=3) # subtract outflows of existing and nuclear capacities during 2016-2020 (pre 2016 age-cohorts)
stock_2020_srpc_ex_nuc[:,:,:,SwitchTime:SwitchTime+5] += TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,:,SwitchTime:SwitchTime+5].sum(axis=3) # add inflows based on outflows of existing and nuclear capacities during (2016-2020 age-cohorts, post 2020 outflows; this is only Wind an PV)
stock_2020_srpc_ex_nuc[:,:,:,SwitchTime:SwitchTime+5] += TIMES_inflow_srpt_annual_ex_nuc[:,:,:,1:6] # add inflows based on VAR_NCap for nuclear during 2016-2020 (this is only nuclear extension + ENFR_FLAMANVILLE-3)
delta_2020_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes)))
delta_2020_ex_nuc[:,:,:] = TIMES_stock_srpt_ex_nuc[:,:,:,1] - stock_2020_srpc_ex_nuc[:,:,:,:].sum(axis=3) # difference of this new 2020 stock and TIMES 2020 stock
# delta_2020_ex_nuc contains negative values --> only add positive values to age-cohorts 2016-2020 (negative deltas because of overestimation of 2015 stock, see above)
delta_2020_ex_nuc[delta_2020_ex_nuc < 0] = 0
stock_2020_srpc_ex_nuc[:,:,:,SwitchTime:SwitchTime+5] += delta_2020_ex_nuc[:,:,:][:,:,:,np.newaxis]/5 # assigne delta_2020 to age-cohorts 2016-2020 (all other in 2060 remaining capacities already added to 2015 stock)

# 2016-2020 inflows based on outflows with age-cohort 2016-2020 + VAR_NCap for nuclear during 2016-2020 + delta 2020 (2016-2020 inflows of "existing"|nuclear not covered elsewhere)
inflow_2016_2020_srpt_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float) # inflows 2016-2020 based on outflows of existing capacities and VAR_NCap for nuclear
inflow_2016_2020_srpt_ex_nuc[:,:,:,1:6] = TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,:,SwitchTime:SwitchTime+5].sum(axis=3) # age cohorts 2016-2020 (sum over all t) based on outflows
inflow_2016_2020_srpt_ex_nuc[:,:,:,1:6] += TIMES_inflow_srpt_annual_ex_nuc[:,:,:,1:6] # add VAR_NCap for nuclear during 2016-2020 (this is only nuclear extension + ENFR_FLAMANVILLE-3)
inflow_2016_2020_srpt_ex_nuc[:,:,:,1:6] +=  delta_2020_ex_nuc[:,:,:][:,:,:,np.newaxis]/5 # add delta_2020_ex_nuc to age-cohorts 2016-2020 
# Check Delta_2016_2020_inflows = inflow_2016_2020_srpt_ex_nuc[:,:,:,1:6].sum() - TIMES_inflow_srpt_annual_ex_nuc[:,:,:,1:6].sum()
# Delta_2016_2020_inflows >> 0 because TIMES does not track inflows for existing capacities 2016-2020

inflow_2021_2060_srpt_ex_nuc = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float) # inflows 2021-2060 based on outflows of existing capacities and VAR_NCap for nuclear
inflow_2021_2060_srpt_ex_nuc[:,:,:,6:] = TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,:,SwitchTime+5:].sum(axis=3) # age cohorts 2021-2060 (sum over all t) based on outflows, should be zero
inflow_2021_2060_srpt_ex_nuc[:,:,:,6:] += TIMES_inflow_srpt_annual_ex_nuc[:,:,:,6:] # add VAR_NCap for nuclear during 2021-2060 (this is only nuclear extension + ENFR_FLAMANVILLE-3, but this has lifetime beyond 2060 so only inflow in 2021-2060)


# Check: for existing and nuclear technologies, 2015 stock + 2016-2020 inflows - 2016-2020 outflows should equal 2020 stock (with some tolerance for numerical issues)
stock_2015_plus_inflow_minus_outflow_2020_ex_nuc = stock_2015_srpc_ex_nuc.sum() + inflow_2016_2020_srpt_ex_nuc[:,:,:,1:6].sum() - TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,1:6,:].sum()
stock_2020_diff_ex_nuc = stock_2020_srpc_ex_nuc.sum() - stock_2015_plus_inflow_minus_outflow_2020_ex_nuc
print(f"Check: 2015 stock + 2016-2020 inflows - 2016-2020 outflows = 2020 stock: {stock_2015_plus_inflow_minus_outflow_2020_ex_nuc} vs {stock_2020_srpc_ex_nuc.sum()} (difference: {stock_2020_diff_ex_nuc})")      
# Check: 2015 stock + 2016-2060 inflows - 2016-2060 outflows should equal 2060 stock
stock_2015_plus_inflow_minus_outflow_2060_ex_nuc = np.zeros((len(scenarios),len(regions),len(processes)))
stock_2015_plus_inflow_minus_outflow_2060_ex_nuc[:,:,:] = stock_2015_srpc_ex_nuc[:,:,:,:].sum(axis=3) + inflow_2016_2020_srpt_ex_nuc[:,:,:,:].sum(axis=3) + inflow_2021_2060_srpt_ex_nuc[:,:,:,:].sum(axis=3) - TIMES_outflow_srpt_annual_ex_nuc_c[:,:,:,1:,:].sum(axis=3).sum(axis=3)
stock_2060_diff_ex_nuc = np.zeros((len(scenarios),len(regions),len(processes)))
stock_2060_diff_ex_nuc[:,:,:] = TIMES_stock_srpt_ex_nuc[:,:,:,9] - stock_2015_plus_inflow_minus_outflow_2060_ex_nuc
print(f"Check (only existing nad nuclear): 2015 stock + 2016-2060 inflows - 2016-2060 outflows = 2060 stock: {stock_2015_plus_inflow_minus_outflow_2060_ex_nuc.sum()} vs {TIMES_stock_srpt_ex_nuc[:,:,:,9].sum()} (difference: {stock_2060_diff_ex_nuc.sum()})")
## --> process 'EUNUC3rd10 [Nuclear third LWR]' in regions 'FR', 'ES' and 'SE'  is main cause for deviation 
## Issue with 'EUNUC3rd10 [Nuclear third LWR]': VAR_Ncap proceeds corresponing VAR_Cap by one period in TIMES results.
## This means, there is an inflow, but stock change is zero, leading to a positive outflow of (inflow - stock change), while negative outflows in following years are set to zero.

# 3.3) Create outflow age-cohort array for all other technologies (non-existing and non-nuclear) based on inflows and lifetimes
inflow_2016_2060_srpt_all_other = np.zeros((len(scenarios), len(regions), len(processes), Nt), dtype=float) # inflows 2016-2060 based on TIMES inflows (VAR_Ncap) 2016-2060
outflow_2016_2060_srpt_all_other_c = np.zeros((len(scenarios), len(regions), len(processes), Nt, Nc), dtype=float) # outflows 2016-2060 based on inflows and lifetimes for non-existing and non-nuclear technologies
for i_s in range(len(scenarios)):
    for i_p in range(len(processes)):
        if i_p not in proc_idx_ex_nuc: # only for non-existing and non-nuclear technologies
            for i_t in range(1,Nt): # start from 1 as no inflow/outflow in 2015
                inflow = np.zeros(len(regions))
                inflow[:] = TIMES_inflow_srpt_annual[i_s, :, i_p, i_t]
                lifetime = TIMES_lifetime_spt_annual[i_s, i_p, i_t]
                # Determine age-cohort of this outflow based on lifetime
                age_cohort = SwitchTime - 1 + i_t # age-cohort is year of inflow (t)
                outflow_year = i_t + int(lifetime) # year of outflow based on inflow year and lifetime
                if outflow_year < Nt: # only consider outflows until 2060
                    outflow_2016_2060_srpt_all_other_c[i_s, :, i_p, outflow_year, age_cohort] += inflow[:]
                inflow_2016_2060_srpt_all_other[i_s, :, i_p, i_t] = inflow[:]
# Check: inflow_2016_2060_srpt_all_other.sum() - TIMES_inflow_srpt_annual_all_other.sum() [excluding nuc and existing] = 0. Correct!
inflow_diff_all_other = inflow_2016_2060_srpt_all_other[:,:,:,1:] - TIMES_inflow_srpt_annual_all_other[:,:,:,1:]
# Check outflow_2016_2060_srpt_all_other_c vs  TIMES_outflow_srpt_annual_all_other:
outflow_diff_all_other = np.zeros((len(scenarios), len(regions), len(processes), Nt))
outflow_diff_all_other[:,:,:,:] = outflow_2016_2060_srpt_all_other_c[:,:,:,:,:].sum(axis=4) - TIMES_outflow_srpt_annual_all_other[:,:,:,:]
#Early retirements not captured, thus generally lower outflow of non-existing and non-nuclear technologies computed than in TIMES.
# For scenarios '02_19taxncs' and '02_19taxslow', computed outflows slightly bigger than TIMES. Reason: for technologies with lifetime extension lifetime slightly underestimated.

# Check: inflow_2016_2060_srpt_all_other.sum() - outflow_2016_2060_srpt_all_other_c.sum() = 2060-2015 stock for non-existing and non-nuclear technologies (with some tolerance for numerical issues)
stock_2015_plus_inflow_minus_outflow_2060_all_other = np.zeros((len(scenarios),len(regions),len(processes)))
stock_2015_plus_inflow_minus_outflow_2060_all_other[:,:,:] = inflow_2016_2060_srpt_all_other[:,:,:,1:].sum(axis=3) - outflow_2016_2060_srpt_all_other_c[:,:,:,1:,SwitchTime:].sum(axis=3).sum(axis=3)
stock_2060_diff_all_other = np.zeros((len(scenarios),len(regions),len(processes)))
stock_2060_diff_all_other[:,:,:] = TIMES_stock_srpt_all_other[:,:,:,9] - stock_2015_plus_inflow_minus_outflow_2060_all_other
## Difference of -69 GW to +91 GW per scenario (for all scenarios less than 2% deviation)
## Early retirements not captured, thus generally larger 2060 stock computed than TIMES 2060 stock for non-existing and non-nuclear technologies.
# For scenarios '02_19taxncs' and '02_19taxslow', computed 2060 stock slightly smaller than TIMES stock. Reason: for technologies with lifetime extension lifetime slightly underestimated.

###### 4.) Export 2015 stock with age-cohorts, annual inflows, and annual outflows with age-cohorts ######
'''# First export for comparison with TIMES technology and region resolution.
# Export `stock_2015_srpc_ex_nuc` to CSV with `Scenario = '01_ssp2v3'`, `Region`, `Process` in columns A:C
# and age-cohorts as column headers (row 1 starting at column D).
try:
    Nc_export = stock_2015_srpc_ex_nuc.shape[3]
except Exception:
    Nc_export = stock_2015_srpc_ex_nuc.shape[-1]
age_years = list(range(1900, 1900 + Nc_export))
age_cols = [str(y) for y in age_years]
rows = []
target_scenario = '01_ssp2v3'
for i_s, s in enumerate(scenarios):
    if str(s).strip() != target_scenario:
        continue
    for i_r, r in enumerate(regions):
        for i_p, p in enumerate(processes):
            vals = stock_2015_srpc_ex_nuc[i_s, i_r, i_p, :].tolist()
            rows.append([s, r, p] + vals)
cols = ['Scenario', 'Region', 'Process'] + age_cols
df_stock2015 = pd.DataFrame(rows, columns=cols)
out_fname = 'TIMES_2015_stock_comp_w_age-cohorts.csv'
df_stock2015.to_csv(out_fname, index=False)

'''

## mapping of scenario, region and process names from TIMES to RECC technology and region names needed for export of parameter files for RECC model.
# Technologies are aggregated according to the mapping
# computed TIMES stock 2015 array: 'stock_2015_srpc_ex_nuc'
# computed TIMES inflows 2016-2060, all technologies:
inflow_2016_2060_srpt_all = inflow_2016_2020_srpt_ex_nuc + inflow_2021_2060_srpt_ex_nuc + inflow_2016_2060_srpt_all_other
# computed TIMES outflows 2016-2060, all technologies:
outflow_2016_2060_srptc_all = TIMES_outflow_srpt_annual_ex_nuc_c + outflow_2016_2060_srpt_all_other_c

#balance check
stock2060_check_RECC_vs_TIMES = stock_2015_srpc_ex_nuc[:,:,:,:].sum(axis=3) + inflow_2016_2060_srpt_all[:,:,:,:].sum(axis=3) - outflow_2016_2060_srptc_all[:,:,:,:,:].sum(axis=(3,4)) - TIMES_stock_srpt_ex_nuc[:,:,:,9] - TIMES_stock_srpt_all_other[:,:,:,9]
print(f"Balance check for 2060 stock: 2015 stock + 2016-2060 inflows - 2016-2060 outflows - TIMES 2060 stock = {stock2060_check_RECC_vs_TIMES.sum()}")
print(f"Balance check per scenario: {stock2060_check_RECC_vs_TIMES[:,:,:].sum(axis=(1,2))}")

# load scenario scenario_mapping.xlsx file with mapping of TIMES scenario names to RECC scenario names
scenario_mapping_df = pd.read_excel('scenario_mapping.xlsx')

times_to_recc_technology_mapping ={
    "EAUTOGENBIO00 [EAUT.Electricity Autoproduction.BIO.00]":"Biomass|Other (Not Elsewhere Specified)",
    "EAUTOGENGAS00 [EAUT.Electricity Autoproduction.GAS.00]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EAUTOGENOIL00 [EAUT.Electricity Autoproduction.OIL.00]":"Oil|Other (Not Elsewhere Specified)",
    "EAUTOGENWASTE00 [EAUT.Electricity Autoproduction.Waste.00]":"Biomass|Other (Not Elsewhere Specified)",
    "ECHP_biomass_thermal [Existing CHP plant - biomass_thermal]":"Biomass|Other (Not Elsewhere Specified)",
    "ECHP_coal_thermal [Existing CHP plant - coal_thermal]":"Coal|Other (Not Elsewhere Specified)",
    "ECHP_HFO_thermal [Existing CHP plant - HFO_thermal]":"Oil|Other (Not Elsewhere Specified)",
    "ECHP_LFO_thermal [Existing CHP plant - LFO_thermal]":"Oil|Other (Not Elsewhere Specified)",
    "ECHP_naturalgas_CCGT [Existing CHP plant - naturalgas_CCGT]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "ECHP_naturalgas_OCGT [Existing CHP plant - naturalgas_OCGT]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "ECHP_naturalgas_thermal [Existing CHP plant - naturalgas_thermal]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EEPP_biomass_thermal [Existing Electricity plant - biomass_thermal]":"Biomass|Other (Not Elsewhere Specified)",
    "EEPP_coal_thermal [Existing Electricity plant - coal_thermal]":"Coal|Other (Not Elsewhere Specified)",
    "EEPP_geothermal [Existing Electricity plant - geothermal]":"Geothermal|Other (Not Elsewhere Specified)",
    "EEPP_HFO_thermal [Existing Electricity plant - HFO_thermal]":"Oil|Other (Not Elsewhere Specified)",
    "EEPP_LFO_thermal [Existing Electricity plant - LFO_thermal]":"Oil|Other (Not Elsewhere Specified)",
    "EEPP_lignite_thermal [Existing Electricity plant - lignite_thermal]":"Coal|Other (Not Elsewhere Specified)",
    "EEPP_naturalgas_CCGT [Existing Electricity plant - naturalgas_CCGT]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EEPP_naturalgas_OCGT [Existing Electricity plant - naturalgas_OCGT]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EEPP_naturalgas_thermal [Existing Electricity plant - naturalgas_thermal]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EEPP_PV [Existing Electricity plant - PV]":"Solar|PV|Other (Not Elsewhere Specified)",
    "EEPP_windON [Existing Electricity plant - windON - onshore]":"Wind|Onshore|Other (Not Elsewhere Specified)",
    "ESTHYDPS101 [Pumped Hydro ELC Storage: DayNite]":"Hydro|Other (Not Elsewhere Specified)",
    "EUCCGASCCSpos20 [CCGT Combined Cycle Gas Turbine + CCS Seq post combustion]":"Gas|Combined Cycle|w/ CCS",
    "EUCCGTGAS15 [Gas Turbine Combined Cycle Gas Advanced]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EUGEOORC01 [Geothermal hydrothermal with ORC]":"Geothermal|Other (Not Elsewhere Specified)",
    "EUHYDDAM00 [Existing Hydro Dams]":"Hydro|Other (Not Elsewhere Specified)",
    "EUHYDLAKELC01 [Lake large scale cheap hydroelectricity > 10 MW]":"Hydro|Other (Not Elsewhere Specified)",
    "EUHYDLAKELE01 [Lake large scale expensive hydroelectricity > 10 MW]":"Hydro|Other (Not Elsewhere Specified)",
    "EUHYDRUN00 [Existing Run-of-river hydro]":"Hydro|Other (Not Elsewhere Specified)",
    "EUHYDRUN01 [Run of River hydroelectricity]":"Hydro|Other (Not Elsewhere Specified)",
    "EUIGCCWOOCCS01 [Biomass Integrated Gasification CC + CCS Seq post combustion]":"Biomass|w/ CCS",
    "EUOCGTGASA01 [OCGT Peak Device Gas Advanced]":"Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    "EUPVSOLL101 [Solar PV Utility c-Si, flat]":"Solar|PV|Other (Not Elsewhere Specified)",
    "EUPVSOLS201 [Solar PV Commercial c-Si, flat]":"Solar|PV|Other (Not Elsewhere Specified)",
    "EUSTHFOsup01 [Steam Turbine Fuel Oil Supercritical]":"Oil|Other (Not Elsewhere Specified)",
    "EUWINONH01 [Wind onshore CF 25 or more]":"Wind|Onshore|Other (Not Elsewhere Specified)",
    "EUWINONL01 [Wind onshore CF 15 to 20]":"Wind|Onshore|Other (Not Elsewhere Specified)",
    "EUWINONM01 [Wind onshore CF 20 to 25]":"Wind|Onshore|Other (Not Elsewhere Specified)",
    "EEPP_windOFF [Existing Electricity plant - windOFF - offshore]":"Wind|Offshore|Other (Not Elsewhere Specified)",
    "ENBE_DOEL-1_00 [Nuclear Power Plant: DOEL-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_DOEL-2_00 [Nuclear Power Plant: DOEL-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_DOEL-3_00 [Nuclear Power Plant: DOEL-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_DOEL-4_00 [Nuclear Power Plant: DOEL-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_TIHANGE-1_00 [Nuclear Power Plant: TIHANGE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_TIHANGE-2_00 [Nuclear Power Plant: TIHANGE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBE_TIHANGE-3_00 [Nuclear Power Plant: TIHANGE-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUOCEWAV01 [Wave (nearshore)]":"Hydro|Other (Not Elsewhere Specified)",
    "EUOCEWAV02 [Wave (offshore)]":"Hydro|Other (Not Elsewhere Specified)",
    "EUSTWOOCCS01 [Fluidized Bed Biomass + CCS Seq post combustion]":"Biomass|w/ CCS",
    "EUWINOFH01 [Wind offshore 3 deeper waters (<60m)]":"Wind|Offshore|Other (Not Elsewhere Specified)",
    "ECHP_lignite_thermal [Existing CHP plant - lignite_thermal]":"Coal|Other (Not Elsewhere Specified)",
    "ENBG_BELENE-1 [Nuclear Power Plant: BELENE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBG_BELENE-2 [Nuclear Power Plant: BELENE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBG_KOZLODUY-5_00 [Nuclear Power Plant: KOZLODUY-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBG_KOZLODUY-6_00 [Nuclear Power Plant: KOZLODUY-6]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENBG_KOZLODUY-7 [Nuclear Power Plant: KOZLODUY-7]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUSTWOO01 [Fluidized Bed Boiler Biomass + steam turbine]":"Biomass|Other (Not Elsewhere Specified)",
    "EUWINOFV01 [Wind offshore 4 floating (<100m)]":"Wind|Offshore|Other (Not Elsewhere Specified)",
    "ENCH_BEZNAU-1_00 [Nuclear Power Plant: BEZNAU-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCH_BEZNAU-2_00 [Nuclear Power Plant: BEZNAU-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCH_GOESGEN_00 [Nuclear Power Plant: GOESGEN]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCH_LEIBSTADT_00 [Nuclear Power Plant: LEIBSTADT]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCH_MUEHLEBERG_00 [Nuclear Power Plant: MUEHLEBERG]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUICDST101 [Peak Device Diesel Conventional]":"Oil|Other (Not Elsewhere Specified)",
    "EUCSPSOL401 [Solar CSP Solar Tower 12-15h storage]":"Solar|CSP",
    "EAUTOGENSOLID00 [EAUT.Electricity Autoproduction.SOLID.00]":"Biomass|Other (Not Elsewhere Specified)",
    "ENCZ_DUKOVANY-1_00 [Nuclear Power Plant: DUKOVANY-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_DUKOVANY-2_00 [Nuclear Power Plant: DUKOVANY-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_DUKOVANY-3_00 [Nuclear Power Plant: DUKOVANY-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_DUKOVANY-4_00 [Nuclear Power Plant: DUKOVANY-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_DUKOVANY-5 [Nuclear Power Plant: DUKOVANY-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_TEMELIN-1_00 [Nuclear Power Plant: TEMELIN-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_TEMELIN-2_00 [Nuclear Power Plant: TEMELIN-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_TEMELIN-3 [Nuclear Power Plant: TEMELIN-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENCZ_TEMELIN-4 [Nuclear Power Plant: TEMELIN-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "EEPP_CSP [Existing Electricity plant - CSP]":"Solar|CSP",
    "ENDE_BIBLIS-A_00 [Nuclear Power Plant: BROKDORF (KBR)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_BIBLIS-B_00 [Nuclear Power Plant: EMSLAND (KKE)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_BROKDORF_KBR_00 [Nuclear Power Plant: PHILIPPSBURG-2 (KKP 2)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_BRUNSBUETTEL_00 [Nuclear Power Plant: GROHNDE (KWG)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_EMSLAND_KKE_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_GRAFENRHEINFELD_KKG_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_GROHNDE_KWG_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_GUNDREM-B_GUN-B_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_GUNDREM-C_GUN-C_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_ISAR-1_00 [Nuclear Power Plant: GUNDREMMINGEN-B (GUN-B)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_ISAR-2_KKI2_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_KRUEMMEL_00 [Nuclear Power Plant: NECKARWESTHEIM-2 (GKN 2)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_NECKARWEST-2_GKN2_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_NECKARWESTHEIM-1_00 [Nuclear Power Plant: GRAFENRHEINFELD (KKG)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_PHILIPPSBURG-1_00 [Nuclear Power Plant: ISAR-2 (KKI 2)]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_PHILIPPSBURG-2_KKP2_00 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENDE_UNTERWESER_00 [Nuclear Power Plant: GUNDREMMINGEN-C (GUN-C)]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUOCETID02 [Tidal energy range]":"Hydro|Other (Not Elsewhere Specified)",
    "EEPP_coal_CCGT [Existing Electricity plant - coal_CCGT]":"Coal|Other (Not Elsewhere Specified)",
    "EEPP_OCE [Existing Electricity plant - OCE - offshore]":"Wind|Offshore|Other (Not Elsewhere Specified)",
    "ENES_ALMARAZ-1_00 [Nuclear Power Plant: ALMARAZ-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_ALMARAZ-2_00 [Nuclear Power Plant: ALMARAZ-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_ASCO-1_00 [Nuclear Power Plant: ASCO-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_ASCO-2_00 [Nuclear Power Plant: ASCO-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_COFRENTES_00 [Nuclear Power Plant: COFRENTES]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_SANTAMARIADEGARONA_00 [Nuclear Power Plant: SANTA MARIA DE GARONA]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_TRILLO-1_00 [Nuclear Power Plant: TRILLO-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENES_VANDELLOS-2_00 [Nuclear Power Plant: VANDELLOS-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUNUC3rd10 [Nuclear third LWR]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_LOVIISA-1_00 [Nuclear Power Plant: LOVIISA-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_LOVIISA-2_00 [Nuclear Power Plant: LOVIISA-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_OLKILUOTO-1_00 [Nuclear Power Plant: OLKILUOTO-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_OLKILUOTO-2_00 [Nuclear Power Plant: OLKILUOTO-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_OLKILUOTO-3 [Nuclear Power Plant: OLKILUOTO-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_OLKILUOTO-4 [Nuclear Power Plant: OLKILUOTO-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFI_PYHA-YOKI [Nuclear Power Plant: PYHA-YOKI]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BELLEVILLE-1_00 [Nuclear Power Plant: BELLEVILLE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BELLEVILLE-2_00 [Nuclear Power Plant: BELLEVILLE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BLAYAIS-1_00 [Nuclear Power Plant: BLAYAIS-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BLAYAIS-2_00 [Nuclear Power Plant: BLAYAIS-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BLAYAIS-3_00 [Nuclear Power Plant: BLAYAIS-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BLAYAIS-4_00 [Nuclear Power Plant: BLAYAIS-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BUGEY-2_00_00 [Nuclear Power Plant: BUGEY-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BUGEY-3_00_00 [Nuclear Power Plant: BUGEY-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BUGEY-4_00_00 [Nuclear Power Plant: BUGEY-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_BUGEY-5_00_00 [Nuclear Power Plant: BUGEY-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CATTENOM-1_00 [Nuclear Power Plant: CATTENOM-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CATTENOM-2_00 [Nuclear Power Plant: CATTENOM-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CATTENOM-3_00 [Nuclear Power Plant: CATTENOM-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CATTENOM-4_00 [Nuclear Power Plant: CATTENOM-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHINON-B-1_00 [Nuclear Power Plant: CHINON-B-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHINON-B-2_00 [Nuclear Power Plant: CHINON-B-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHINON-B-3_00 [Nuclear Power Plant: CHINON-B-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHINON-B-4_00 [Nuclear Power Plant: CHINON-B-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHOOZ-B-1_00 [Nuclear Power Plant: CHOOZ-B-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CHOOZ-B-2_00 [Nuclear Power Plant: CHOOZ-B-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CIVAUX-1_00 [Nuclear Power Plant: CIVAUX-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CIVAUX-2_00 [Nuclear Power Plant: CIVAUX-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CRUAS-1_00 [Nuclear Power Plant: CRUAS-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CRUAS-2_00 [Nuclear Power Plant: CRUAS-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CRUAS-3_00 [Nuclear Power Plant: CRUAS-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_CRUAS-4_00 [Nuclear Power Plant: CRUAS-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_DAMPIERRE-1_00 [Nuclear Power Plant: DAMPIERRE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_DAMPIERRE-2_00 [Nuclear Power Plant: DAMPIERRE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_DAMPIERRE-3_00 [Nuclear Power Plant: DAMPIERRE-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_DAMPIERRE-4_00 [Nuclear Power Plant: DAMPIERRE-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_FESSENHEIM-1_00 [Nuclear Power Plant: FESSENHEIM-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_FESSENHEIM-2_00 [Nuclear Power Plant: FESSENHEIM-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_FLAMANVILLE-1_00 [Nuclear Power Plant: FLAMANVILLE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_FLAMANVILLE-2_00 [Nuclear Power Plant: FLAMANVILLE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_FLAMANVILLE-3 [Nuclear Power Plant: FLAMANVILLE-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GOLFECH-1_00 [Nuclear Power Plant: GOLFECH-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GOLFECH-2_00 [Nuclear Power Plant: GOLFECH-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-1_00 [Nuclear Power Plant: GRAVELINES-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-2_00 [Nuclear Power Plant: GRAVELINES-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-3_00 [Nuclear Power Plant: GRAVELINES-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-4_00 [Nuclear Power Plant: GRAVELINES-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-5_00 [Nuclear Power Plant: GRAVELINES-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_GRAVELINES-6_00 [Nuclear Power Plant: GRAVELINES-6]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_NOGENT-1_00 [Nuclear Power Plant: NOGENT-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_NOGENT-2_00 [Nuclear Power Plant: NOGENT-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PALUEL-1_00 [Nuclear Power Plant: PALUEL-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PALUEL-2_00 [Nuclear Power Plant: PALUEL-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PALUEL-3_00 [Nuclear Power Plant: PALUEL-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PALUEL-4_00 [Nuclear Power Plant: PALUEL-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PENLY-1_00 [Nuclear Power Plant: PENLY-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PENLY-2_00 [Nuclear Power Plant: PENLY-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_PHENIX_00 [Nuclear Power Plant: PHENIX]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_STALBAN-1_00 [Nuclear Power Plant: ST. ALBAN-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_STALBAN-2_00 [Nuclear Power Plant: ST. ALBAN-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_STLAURENT-B-1_00 [Nuclear Power Plant: ST. LAURENT-B-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_STLAURENT-B-2_00 [Nuclear Power Plant: ST. LAURENT-B-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_TRICASTIN-1_00 [Nuclear Power Plant: TRICASTIN-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_TRICASTIN-2_00 [Nuclear Power Plant: TRICASTIN-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_TRICASTIN-3_00 [Nuclear Power Plant: TRICASTIN-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENFR_TRICASTIN-4_00 [Nuclear Power Plant: TRICASTIN-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUSTCOHsup01 [Supercritical Pulverised Coal]":"Coal|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-1_00 [Nuclear Power Plant: PAKS-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-2_00 [Nuclear Power Plant: PAKS-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-3_00 [Nuclear Power Plant: PAKS-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-4_00 [Nuclear Power Plant: PAKS-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-5 [Nuclear Power Plant: PAKS-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENHU_PAKS-6 [Nuclear Power Plant: PAKS-6]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUGEOF01 [Geothermal hydrothermal with flash power plants]":"Geothermal|Other (Not Elsewhere Specified)",
    "ENLT_VISAGINAS-1 [Nuclear Power Plant: VISAGINAS-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUPCCOHCCSoxy20 [Super-critical Pulverised Coal + CCS Seq Oxyfuel]":"Coal|w/ CCS",
    "ENNL_BORSSELE_00 [Nuclear Power Plant: BORSSELE]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENNL_BORSSELE-2 [Nuclear Power Plant: BORSSELE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENPL_POLAND-1 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENPL_POLAND-2 [Nuclear Power Plant:]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENRO_CERNAVODA-1_00 [Nuclear Power Plant: CERNAVODA-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENRO_CERNAVODA-2 [Nuclear Power Plant: CERNAVODA-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENRO_CERNAVODA-3 [Nuclear Power Plant: CERNAVODA-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENRO_CERNAVODA-4 [Nuclear Power Plant: CERNAVODA-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENRO_CERNAVODA-5 [Nuclear Power Plant: CERNAVODA-5]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_FORSMARK-1_00 [Nuclear Power Plant: FORSMARK-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_FORSMARK-2_00 [Nuclear Power Plant: FORSMARK-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_FORSMARK-3_00 [Nuclear Power Plant: FORSMARK-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_OSKARSHAMN-1_00 [Nuclear Power Plant: OSKARSHAMN-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_OSKARSHAMN-2_00 [Nuclear Power Plant: OSKARSHAMN-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_OSKARSHAMN-3_00 [Nuclear Power Plant: OSKARSHAMN-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_RINGHALS-1_00 [Nuclear Power Plant: RINGHALS-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_RINGHALS-2_00 [Nuclear Power Plant: RINGHALS-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_RINGHALS-3_00 [Nuclear Power Plant: RINGHALS-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSE_RINGHALS-4_00 [Nuclear Power Plant: RINGHALS-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSI_KRSKO_00 [Nuclear Power Plant: KRSKO]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSI_KRSKO-2 [Nuclear Power Plant: KRSKO-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_BOHUNICE-3_00 [Nuclear Power Plant: BOHUNICE-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_BOHUNICE-4_00 [Nuclear Power Plant: BOHUNICE-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_MOCHOVCE-1_00 [Nuclear Power Plant: MOCHOVCE-1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_MOCHOVCE-2_00 [Nuclear Power Plant: MOCHOVCE-2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_MOCHOVCE-3 [Nuclear Power Plant: MOCHOVCE-3]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENSK_MOCHOVCE-4 [Nuclear Power Plant: MOCHOVCE-4]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_DUNGENESS-B1_00 [Nuclear Power Plant: DUNGENESS-B1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_DUNGENESS-B2_00 [Nuclear Power Plant: DUNGENESS-B2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HARTLEPOOL-A1_00 [Nuclear Power Plant: HARTLEPOOL-A1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HARTLEPOOL-A2_00 [Nuclear Power Plant: HARTLEPOOL-A2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HEYSHAM-A1_00 [Nuclear Power Plant: HEYSHAM-A1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HEYSHAM-A2_00 [Nuclear Power Plant: HEYSHAM-A2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HEYSHAM-B1_00 [Nuclear Power Plant: HEYSHAM-B1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HEYSHAM-B2_00 [Nuclear Power Plant: HEYSHAM-B2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HINKLEYPOINT-B1_00 [Nuclear Power Plant: HINKLEY POINT-B1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HINKLEYPOINT-B2_00 [Nuclear Power Plant: HINKLEY POINT-B2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HINKLEYPOINT-C1 [Nuclear Power Plant: HINKLEY POINT-C1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HINKLEYPOINT-C2 [Nuclear Power Plant: HINKLEY POINT-C2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HUNTERSTON-B1_00 [Nuclear Power Plant: HUNTERSTON-B1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_HUNTERSTON-B2_00 [Nuclear Power Plant: HUNTERSTON-B2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_MOORSIDE [Nuclear Power Plant: MOORSIDE]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_OLDBURY-A1_00 [Nuclear Power Plant: OLDBURY-A1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_OLDBURY-B [Nuclear Power Plant: OLDBURY-B]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_SIZEWELL-Bv_00 [Nuclear Power Plant: SIZEWELL-B]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_SIZEWELL-C1 [Nuclear Power Plant: SIZEWELL-C1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_SIZEWELL-C2 [Nuclear Power Plant: SIZEWELL-C2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_TORNESS1_00 [Nuclear Power Plant: TORNESS 1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_TORNESS2_00 [Nuclear Power Plant: TORNESS 2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_WYLFA1_00 [Nuclear Power Plant: WYLFA 1]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_WYLFA2_00 [Nuclear Power Plant: WYLFA 2]":"Nuclear|Other (Not Elsewhere Specified)",
    "ENUK_WYLFA-B [Nuclear Power Plant: WYLFA-B]":"Nuclear|Other (Not Elsewhere Specified)",
    "EUSTCOLsup01 [Supercritical Pulverised Coal lignite]":"Coal|Other (Not Elsewhere Specified)",
    "EUIGCOHCCSpre20 [Integrated gasification combined cycle + CCS Seq pre combustion]":"Gas|Combined Cycle|w/ CCS",
    "EUICDST01 [Internal Combustion Engine Diesel]":"Oil|Other (Not Elsewhere Specified)",
    "EUPVSOLS101 [Solar PV Residential c-Si, inclined]":"Solar|PV|Other (Not Elsewhere Specified)",
    "EUGEOEGS01 [Geothermal EGS with ORC]":"Geothermal|Other (Not Elsewhere Specified)",
    "EUOCETID01 [Tidal energy stream]":"Hydro|Other (Not Elsewhere Specified)",
    "EUHYDLAKEMC01 [Lake medium scale cheap hydroelectricity 1-10 MW]":"Hydro|Other (Not Elsewhere Specified)",
    "EUICBGS01 [Biomass Anaerobic Digestion]":"Biomass|Other (Not Elsewhere Specified)",
    "EUSTIISGASCS101 [EPLT: Steam Turbine.CO2Seq.IISGAS]":"Gas|Combined Cycle|w/ CCS",
    "EUHYDLAKEMC01 [Lake medium scale cheap hydroelectricity 1-10 MW]":"Hydro|Other (Not Elsewhere Specified)",
    "EUOCETID02 [Tidal energy range]":"Hydro|Other (Not Elsewhere Specified)",
    "EUPCCOHCCSoxy20 [Super-critical Pulverised Coal + CCS Seq Oxyfuel]":"Coal|w/ CCS",
    "EUSTCOLsup01 [Supercritical Pulverised Coal lignite]":"Coal|Other (Not Elsewhere Specified)",
    "P_ESTHYDPS101 [Pumped Hydro ELC Storage: DayNite (accompanying tech to represent power)]": "Hydro|Other (Not Elsewhere Specified)"
    }

times_to_recc_region_mapping = {
    "AT":"Austria",
    "BE":"Belgium",
    "BG":"Bulgaria",
    #"CH":"Switzerland",
    "CY":"Cyprus",
    "CZ":"Czech Republic",
    "DE":"Germany",
    "DK":"Denmark",
    "EE":"Estonia",
    "EL":"Greece",
    "ES":"Spain",
    "FI":"Finland",
    "FR":"France",
    "HR":"Croatia",
    "HU":"Hungary",
    "IE":"Ireland",
    #"IS":"Iceland",
    "IT":"Italy",
    "LT":"Lithuania",
    "LU":"Luxembourg",
    "LV":"Latvia",
    "Mt":"Malta", # not included in TIMES results
    "NL":"Netherlands",
    #"NO":"Norway",
    "PL":"Poland",
    "PT":"Portugal",
    "RO":"Romania",
    "SE":"Sweden",
    "SI":"Slovenia",
    "SK":"Slovakia",
    "UK":"UK"
}

## Create `RECC_2015_stock` by mapping TIMES `processes` to RECC technologies (18)
# RECC technologies list (length 18)
RECC_technologies = [
    "Biomass|w/ CCS",
    "Biomass|Other (Not Elsewhere Specified)",
    "Hydro|Other (Not Elsewhere Specified)",
    "Solar|CSP",
    "Solar|PV|Other (Not Elsewhere Specified)",
    "Wind|Onshore|Other (Not Elsewhere Specified)",
    "Wind|Offshore|Other (Not Elsewhere Specified)",
    "Geothermal|Other (Not Elsewhere Specified)",
    "Nuclear|Other (Not Elsewhere Specified)",
    #"Oil|Light oil combined cycle",
    "Oil|Other (Not Elsewhere Specified)",
    "Gas|Combined Cycle|w/ CCS",
    "Gas|Combined Cycle|Other (Not Elsewhere Specified)",
    #"Coal|Integrated gasification combined cycle",
    #"Coal|Advanced plant w/ CCS",
    "Coal|w/ CCS",
    #"Coal|w/o CCS",
    "Coal|Other (Not Elsewhere Specified)"
]

# Initialize RECC_2015_stock: Scenario x Region x RECC_technology x AgeCohort
Nc_stock = stock_2015_srpc_ex_nuc.shape[3]
RECC_2015_stock = np.zeros((len(scenarios), len(regions), len(RECC_technologies), Nc_stock), dtype=float)

# Map processes -> RECC technologies and sum if multiple processes map to same RECC tech
dropped_processes = []
for i_p, proc in enumerate(processes):
    proc_str = str(proc).strip()
    # find mapping key (exact or case-insensitive)
    map_key = None
    if proc_str in times_to_recc_technology_mapping:
        map_key = proc_str
    else:
        for k in times_to_recc_technology_mapping.keys():
            if k.strip().lower() == proc_str.lower():
                map_key = k
                break
    if map_key is None:
        dropped_processes.append(proc_str)
        continue
    recc_name = times_to_recc_technology_mapping[map_key]
    if recc_name not in RECC_technologies:
        dropped_processes.append(proc_str)
        continue
    idx_recc = RECC_technologies.index(recc_name)
    # sum over processes mapping to same RECC technology
    RECC_2015_stock[:, :, idx_recc, :] += stock_2015_srpc_ex_nuc[:, :, i_p, :]

n_dropped = len(dropped_processes)
print(f"RECC_2015_stock created: shape={RECC_2015_stock.shape}; dropped processes={n_dropped}")
if n_dropped > 0:
    print("Dropped processes (examples up to 10):", dropped_processes[:10])


## Map inflows and outflows to RECC technologies
# Initialize RECC_inflows: Scenario x Region x RECC_technology x Nt
RECC_inflows = np.zeros((len(scenarios), len(regions), len(RECC_technologies), Nt), dtype=float)
# Initialize RECC_outflows: Scenario x Region x RECC_technology x Nt x Nc
RECC_outflows = np.zeros((len(scenarios), len(regions), len(RECC_technologies), Nt, Nc), dtype=float)

# Track dropped processes separately for inflow/outflow mapping
dropped_processes_inout = []
for i_p, proc in enumerate(processes):
    proc_str = str(proc).strip()
    map_key = None
    if proc_str in times_to_recc_technology_mapping:
        map_key = proc_str
    else:
        for k in times_to_recc_technology_mapping.keys():
            if k.strip().lower() == proc_str.lower():
                map_key = k
                break
    if map_key is None:
        dropped_processes_inout.append(proc_str)
        continue
    recc_name = times_to_recc_technology_mapping[map_key]
    if recc_name not in RECC_technologies:
        dropped_processes_inout.append(proc_str)
        continue
    idx_recc = RECC_technologies.index(recc_name)
    # Sum inflows (time series)
    try:
        RECC_inflows[:, :, idx_recc, :] += inflow_2016_2060_srpt_all[:, :, i_p, :]
    except Exception:
        # handle potential shape mismatch
        pass
    # Sum outflows (age-cohorted)
    try:
        RECC_outflows[:, :, idx_recc, :, :] += outflow_2016_2060_srptc_all[:, :, i_p, :, :]
    except Exception:
        # handle potential shape mismatch
        pass

n_dropped_inout = len(dropped_processes_inout)
print(f"RECC_inflows shape={RECC_inflows.shape}; RECC_outflows shape={RECC_outflows.shape}; dropped processes={n_dropped_inout}")
if n_dropped_inout > 0:
    print("Dropped processes for inflow/outflow mapping (examples up to 10):", dropped_processes_inout[:10])


# Now map TIMES regions to RECC regions and aggregate/drop unmatched regions
# Build mapping of old region index -> RECC region name (if available)
oldidx_to_recc = {}
dropped_regions = []
for i_r, rg in enumerate(regions):
    rg_str = str(rg).strip()
    map_key = None
    if rg_str in times_to_recc_region_mapping:
        map_key = rg_str
    else:
        for k in times_to_recc_region_mapping.keys():
            if k.strip().lower() == rg_str.lower():
                map_key = k
                break
    if map_key is None:
        dropped_regions.append(rg_str)
        continue
    recc_region = times_to_recc_region_mapping[map_key]
    oldidx_to_recc[i_r] = recc_region

# Build ordered list of RECC regions present in the TIMES `regions` list
RECC_regions = []
for i_r in range(len(regions)):
    if i_r in oldidx_to_recc:
        name = oldidx_to_recc[i_r]
        if name not in RECC_regions:
            RECC_regions.append(name)

n_recc_regions = len(RECC_regions)

# Aggregate RECC_2015_stock across mapped regions
RECC_2015_stock_mapped = np.zeros((len(scenarios), n_recc_regions, len(RECC_technologies), Nc_stock), dtype=float)
for old_r, recc_name in oldidx_to_recc.items():
    new_r = RECC_regions.index(recc_name)
    RECC_2015_stock_mapped[:, new_r, :, :] += RECC_2015_stock[:, old_r, :, :]

# Aggregate RECC_inflows (Scenario x Region x Tech x Nt)
RECC_inflows_mapped = np.zeros((len(scenarios), n_recc_regions, len(RECC_technologies), Nt), dtype=float)
for old_r, recc_name in oldidx_to_recc.items():
    new_r = RECC_regions.index(recc_name)
    RECC_inflows_mapped[:, new_r, :, :] += RECC_inflows[:, old_r, :, :]

# Aggregate RECC_outflows (Scenario x Region x Tech x Nt x Nc)
RECC_outflows_mapped = np.zeros((len(scenarios), n_recc_regions, len(RECC_technologies), Nt, Nc_stock), dtype=float)
for old_r, recc_name in oldidx_to_recc.items():
    new_r = RECC_regions.index(recc_name)
    RECC_outflows_mapped[:, new_r, :, :, :] += RECC_outflows[:, old_r, :, :, :]

# Overwrite variables with mapped versions
RECC_2015_stock = RECC_2015_stock_mapped
RECC_inflows = RECC_inflows_mapped
RECC_outflows = RECC_outflows_mapped

print(f"Mapped RECC arrays to {n_recc_regions} RECC regions; dropped TIMES regions={len(dropped_regions)}")
if len(dropped_regions) > 0:
    print("Dropped TIMES regions (examples up to 10):", dropped_regions[:10])

# --- Filter scenarios for inflows/outflows using `scenario_mapping_df` ---
# Keep only TIMES scenarios that appear in the first column of `scenario_mapping_df`.
try:
    mapping_names = scenario_mapping_df.iloc[:,0].astype(str).str.strip().tolist()
except Exception:
    mapping_names = []

# build lookup of existing TIMES scenarios (lowercase stripped)
sc_lookup = {str(s).strip().lower(): i for i, s in enumerate(scenarios)}
keep_indices = []
seen = set()
for name in mapping_names:
    key = str(name).strip().lower()
    if key in sc_lookup and sc_lookup[key] not in seen:
        keep_indices.append(sc_lookup[key])
        seen.add(sc_lookup[key])

if len(keep_indices) == 0:
    print("Warning: no TIMES scenarios from scenario_mapping.xlsx matched; RECC_inflows/outflows will be empty.")
    # make empty arrays with zero scenarios
    RECC_inflows = np.zeros((0, RECC_inflows.shape[1], RECC_inflows.shape[2], RECC_inflows.shape[3]), dtype=float)
    RECC_outflows = np.zeros((0, RECC_outflows.shape[1], RECC_outflows.shape[2], RECC_outflows.shape[3], RECC_outflows.shape[4]), dtype=float)
    RECC_scenarios = []
else:
    # preserve order defined in scenario_mapping.xlsx
    RECC_inflows = RECC_inflows[keep_indices, :, :, :]
    RECC_outflows = RECC_outflows[keep_indices, :, :, :, :]
    RECC_scenarios = [scenarios[i] for i in keep_indices]
    print(f"Filtered RECC_inflows/outflows to {len(RECC_scenarios)} scenarios (from scenario_mapping.xlsx)")

# --- Filter RECC_2015_stock: keep only scenario '01_ssp2v3' ---
target_s = '01_ssp2v3'
found_idx = None
for i_s, s in enumerate(scenarios):
    if str(s).strip() == target_s:
        found_idx = i_s
        break
if found_idx is None:
    print(f"Warning: target scenario '{target_s}' not found in TIMES scenarios; RECC_2015_stock will be empty.")
    RECC_2015_stock = np.zeros((0, RECC_2015_stock.shape[1], RECC_2015_stock.shape[2], RECC_2015_stock.shape[3]), dtype=float)
    RECC_2015_scenarios = []
else:
    RECC_2015_stock = RECC_2015_stock[found_idx:found_idx+1, :, :, :]
    RECC_2015_scenarios = [scenarios[found_idx]]
    print(f"Filtered RECC_2015_stock to scenario '{target_s}' (index {found_idx})")

# Keep only age-cohorts indices 0:115 (inclusive) for RECC_2015_stock and drop 116:160
try:
    # axis 3 is age-cohort
    if RECC_2015_stock.shape[3] > 116:
        RECC_2015_stock = RECC_2015_stock[:, :, :, :116]
        print(f"Truncated RECC_2015_stock age-cohorts to 0:115 — new shape {RECC_2015_stock.shape}")
    else:
        print(f"RECC_2015_stock age-cohorts length ({RECC_2015_stock.shape[3]}) <=116; no truncation applied")
except Exception as e:
    print(f"Warning: could not truncate RECC_2015_stock age-cohorts: {e}")

# Export RECC_2015_stock to CSV: axes 0..2 -> columns A:C, age-cohorts (axis 3) as column headers from D onwards
try:
    n_age = RECC_2015_stock.shape[3]
    # derive age years starting at 1900
    age_years = list(range(1900, 1900 + n_age))
    age_cols = [str(y) for y in age_years]
    rows = []
    # scenarios, regions and technologies lists
    try:
        s_list = RECC_2015_scenarios
    except Exception:
        s_list = [str(i) for i in range(RECC_2015_stock.shape[0])]
    try:
        r_list = RECC_regions
    except Exception:
        r_list = [str(i) for i in range(RECC_2015_stock.shape[1])]
    t_list = RECC_technologies
    for i_s in range(RECC_2015_stock.shape[0]):
        sname = s_list[i_s] if i_s < len(s_list) else str(i_s)
        for i_r in range(RECC_2015_stock.shape[1]):
            rname = r_list[i_r] if i_r < len(r_list) else str(i_r)
            for i_t in range(RECC_2015_stock.shape[2]):
                tname = t_list[i_t] if i_t < len(t_list) else str(i_t)
                vals = RECC_2015_stock[i_s, i_r, i_t, :].tolist()
                rows.append([sname, rname, tname] + vals)
    cols = ['Scenario', 'Region', 'RECC_Technology'] + age_cols
    df_recc2015 = pd.DataFrame(rows, columns=cols)
    out_file = 'RECC_2015_stock.xlsx'
    df_recc2015.to_excel(out_file, sheet_name='values', index=False)
    print(f"Exported RECC_2015_stock to {out_file} with shape {df_recc2015.shape}")
except Exception as e:
    print(f"Warning: could not export RECC_2015_stock to xlsx: {e}")


# --- Reshape RECC_inflows and RECC_outflows to include SSP x RCP x CE scenario axes ---
# scenario_mapping_df: col 0=TIMES scenario, col1=SSP, col3=RCP, col4=CE
try:
    df_map = scenario_mapping_df.copy()
except Exception:
    df_map = None

if df_map is None:
    print("Warning: scenario_mapping_df not available; cannot reshape RECC_inflows/outflows to SSPxRCPxCE.")
else:
    # Build list of kept TIMES scenario names (after earlier filtering)
    try:
        RECC_scenarios
    except NameError:
        # if not defined, fall back to original scenarios kept earlier
        RECC_scenarios = [s for s in scenarios]

    sc_to_idx = {str(s).strip(): i for i, s in enumerate(RECC_scenarios)}

    # Collect mapping rows that correspond to kept TIMES scenarios
    rows = []
    for _, r in df_map.iterrows():
        times_name = str(r.iloc[0]).strip()
        if times_name in sc_to_idx:
            ssp_val = str(r.iloc[1]).strip() if pd.notna(r.iloc[1]) else ''
            rcp_val = str(r.iloc[2]).strip() if pd.notna(r.iloc[2]) else ''
            ce_val = str(r.iloc[3]).strip() if pd.notna(r.iloc[3]) else ''
            if ssp_val == '' or rcp_val == '' or ce_val == '':
                continue
            rows.append((times_name, ssp_val, rcp_val, ce_val))

    # Unique lists
    SSP_list = sorted(list({r[1] for r in rows}))
    RCP_list = sorted(list({r[2] for r in rows}))
    CE_list = sorted(list({r[3] for r in rows}))

    nSSP = len(SSP_list)
    nRCP = len(RCP_list)
    nCE = len(CE_list)

    if nSSP == 0 or nRCP == 0 or nCE == 0:
        print("Warning: could not extract SSP/RCP/CE lists from scenario_mapping_df; skipping reshape.")
    else:
        print(f"Creating SSP x RCP x CE axes: {nSSP} SSP x {nRCP} RCP x {nCE} CE")
        # Prepare new arrays
        n_regions = RECC_inflows.shape[1]
        n_tech = RECC_inflows.shape[2]
        Nt_len = RECC_inflows.shape[3]
        Nc_len = RECC_outflows.shape[4]

        RECC_inflows_new = np.zeros((nSSP, nRCP, nCE, n_regions, n_tech, Nt_len), dtype=float)
        RECC_outflows_new = np.zeros((nSSP, nRCP, nCE, n_regions, n_tech, Nt_len, Nc_len), dtype=float)

        # map each kept TIMES scenario into SSP,RCP,CE cell
        for times_name, ssp_val, rcp_val, ce_val in rows:
            s_idx = sc_to_idx[times_name]
            i_ssp = SSP_list.index(ssp_val)
            i_rcp = RCP_list.index(rcp_val)
            i_ce = CE_list.index(ce_val)
            # add (sum) inflows and outflows into the corresponding cell
            RECC_inflows_new[i_ssp, i_rcp, i_ce, :, :, :] += RECC_inflows[s_idx, :, :, :]
            RECC_outflows_new[i_ssp, i_rcp, i_ce, :, :, :, :] += RECC_outflows[s_idx, :, :, :, :]

        # Overwrite variables and provide lists
        RECC_inflows = RECC_inflows_new
        RECC_outflows = RECC_outflows_new
        RECC_SSP_list = SSP_list
        RECC_RCP_list = RCP_list
        RECC_CE_list = CE_list
        print(f"RECC_inflows reshaped to {RECC_inflows.shape}; RECC_outflows reshaped to {RECC_outflows.shape}")



# Create per-CE arrays: RECC_inflows_<CE> and RECC_outflows_<CE>
import re
RECC_inflows_by_CE = {}
RECC_outflows_by_CE = {}
if isinstance(RECC_CE_list, (list, tuple)) and len(RECC_CE_list) > 0:
    created_infl_names = []
    created_outf_names = []
    for i_ce, ce_val in enumerate(RECC_CE_list):
        ce_str = str(ce_val)
        # sanitize CE value to a safe variable suffix
        suffix = re.sub(r"\W+", "_", ce_str).strip("_")
        infl_var = f"RECC_inflows_{suffix}"
        outf_var = f"RECC_outflows_{suffix}"
        try:
            arr_in = RECC_inflows[:, :, i_ce, :, :, :].copy()
            arr_out = RECC_outflows[:, :, i_ce, :, :, :, :].copy()
        except Exception:
            # shapes may be missing if previous steps failed
            arr_in = None
            arr_out = None
        # set globals so variables are available in module namespace
        if arr_in is not None:
            globals()[infl_var] = arr_in
            RECC_inflows_by_CE[ce_str] = arr_in
            created_infl_names.append(infl_var)
        if arr_out is not None:
            globals()[outf_var] = arr_out
            RECC_outflows_by_CE[ce_str] = arr_out
            created_outf_names.append(outf_var)
    print(f"Created {len(created_infl_names)} RECC_inflows_<CE> arrays and {len(created_outf_names)} RECC_outflows_<CE> arrays")
else:
    print("No CE entries found; skipping creation of per-CE arrays.")

#rewrite outflows as dataframe with full str information across all axes (SSP, RCP, Region, Technology, Year) and age-cohorts as columns to,later exlude unecessary years with just zeros in all age-cohorts (e.g. 2015-2020) and keep only years with non-zero outflows across any age-cohort (to reduce read-in time), then export to CSV for each CE scenario
RECC_outflows_df_by_CE = {}

# Process each CE scenario
for CE_idx, CE_name in enumerate(RECC_CE_list):
    # Get the array for this CE scenario
    arr = RECC_outflows_by_CE[CE_name]  # Shape: (1, 2, 27, 18, 46, 161)
    
    # Create a list to store all rows
    rows_list = []
    years_outflow_removal = list(range(2015, 2015 + Nt))
    # Iterate through all dimensions
    for s in range(arr.shape[0]):  # SSP
        for r in range(arr.shape[1]):  # RCP
            for reg in range(arr.shape[2]):  # Regions
                for tech in range(arr.shape[3]):  # Technologies
                    for year in range(arr.shape[4]):  # Years
                        # Get the age_cohort values (161 values)
                        age_cohort_values = arr[s, r, reg, tech, year, :]
                        
                        # Only keep if NOT all zeros
                        if not np.all(age_cohort_values == 0):
                            # Create a row with index information
                            row = {
                                'SSP_Scenarios': RECC_SSP_list[s],
                                'Scenario_RCP': RECC_RCP_list[r],
                                'SSP_Regions_32': RECC_regions[reg],
                                'Sectors_industry': RECC_technologies[tech],
                                'Year': years_outflow_removal[year]
                            }
                            
                            # Add each age_cohort value as a separate column
                            for cohort_idx, cohort_value in enumerate(age_cohort_values):
                                row[cohort_idx+1900] = cohort_value
                            
                            rows_list.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows_list)
    
    # Store in dictionary
    RECC_outflows_df_by_CE[CE_name] = df
    
    print(f"CE Scenario: {CE_name}")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:10])}...")  # Show first 10 columns
    balance = abs(RECC_outflows_by_CE[CE_name][:,:,:,:,:,:].sum() - RECC_outflows_df_by_CE[CE_name].iloc[:,5:].sum().sum())
    if balance > 1e-10:  # Allow small numerical tolerance
        raise ValueError(f"Balance check failed for CE='{CE_name}': sum of array values ({RECC_outflows_by_CE[CE_name][:,:,:,:,:,:].sum()}) does not match sum of DataFrame values ({RECC_outflows_df_by_CE[CE_name].iloc[:,5:].sum().sum()})!")
    else:
        print(f" Balance check passed ✅: RECC_outflows_by_CE[{CE_name}] - RECC_outflows_df_by_CE[{CE_name}] = {round(balance,2)}, is in acceptable range")
    print()

# Now we have 7 DataFrames, one for each CE scenario
# Access them like: RECC_outflows_df_by_CE['CE_scenario_name']


# Export RECC_inflows_by_CE and RECC_outflows_by_CE to CSV files
try:
    infl_dict = RECC_inflows_by_CE
    out_dict = RECC_outflows_by_CE
except NameError:
    infl_dict = {}
    out_dict = {}

def _safe_name(s):
    import re
    return re.sub(r"\W+", "_", str(s)).strip("_")

# Years for inflow time axis (Nt) — use 2015..2060 if length matches Nt
for ce_key, arr in infl_dict.items():
    try:
        nSSP, nRCP, nReg, nTech, Nt_len = arr.shape
    except Exception:
        print(f"Skipping inflow export for CE='{ce_key}': unexpected array shape")
        continue
    years_cols = list(range(2015, 2015 + Nt_len))
    cols = ['SSP', 'RCP', 'Region', 'RECC_Technology'] + [str(y) for y in years_cols]
    rows = []
    for i_s in range(nSSP):
        s_val = RECC_SSP_list[i_s] if 'RECC_SSP_list' in globals() and i_s < len(RECC_SSP_list) else str(i_s)
        for i_r in range(nRCP):
            r_val = RECC_RCP_list[i_r] if 'RECC_RCP_list' in globals() and i_r < len(RECC_RCP_list) else str(i_r)
            for i_reg in range(nReg):
                reg_val = RECC_regions[i_reg] if i_reg < len(RECC_regions) else str(i_reg)
                for i_t in range(nTech):
                    tech_val = RECC_technologies[i_t]
                    series = arr[i_s, i_r, i_reg, i_t, :].tolist()
                    rows.append([s_val, r_val, reg_val, tech_val] + series)
    df_out = pd.DataFrame(rows, columns=cols)
    fname = f"RECC_inflows_{_safe_name(ce_key)}.xlsx"
    df_out.to_excel(fname, sheet_name="values", index=False)
    print(f"Exported inflows for CE='{ce_key}' to {fname} (rows={len(rows)}, cols={len(cols)})")

#previous version from ch wihtout removal of zero-outflow years
'''# For outflows: axis shape expected (nSSP,nRCP,nReg,nTech,Nt,Nc)
for ce_key, arr in out_dict.items():
    try:
        nSSP, nRCP, nReg, nTech, Nt_len, Nc_len = arr.shape
    except Exception:
        print(f"Skipping outflow export for CE='{ce_key}': unexpected array shape")
        continue
    age_years = list(range(1900, 1900 + Nc_len))
    cols = ['SSP', 'RCP', 'Region', 'RECC_Technology', 'Year'] + [str(y) for y in age_years]
    rows = []
    for i_s in range(nSSP):
        s_val = RECC_SSP_list[i_s] if 'RECC_SSP_list' in globals() and i_s < len(RECC_SSP_list) else str(i_s)
        for i_r in range(nRCP):
            r_val = RECC_RCP_list[i_r] if 'RECC_RCP_list' in globals() and i_r < len(RECC_RCP_list) else str(i_r)
            for i_reg in range(nReg):
                reg_val = RECC_regions[i_reg] if i_reg < len(RECC_regions) else str(i_reg)
                for i_t in range(nTech):
                    tech_val = RECC_technologies[i_t]
                    for i_time in range(Nt_len):
                        year_label = 2015 + i_time
                        age_vals = arr[i_s, i_r, i_reg, i_t, i_time, :].tolist()
                        rows.append([s_val, r_val, reg_val, tech_val, year_label] + age_vals)
    df_out = pd.DataFrame(rows, columns=cols)
    fname = f"RECC_outflows_{_safe_name(ce_key)}.csv"
    df_out.to_csv(fname, index=False)
    print(f"Exported outflows for CE='{ce_key}' to {fname} (rows={len(rows)}, cols={len(cols)})")'''

# Export for RECC_outflows_df_by_CE: axis shape expected (nSSP,nRCP,nReg,nTech,Nt,Nc)
for CE_name, df in RECC_outflows_df_by_CE.items():
    # Sanitize the CE name for use in filename
    safe_filename = re.sub(r"\W+", "_", str(CE_name)).strip("_")
    
    # Create filename
    filename = f"RECC_outflows_{safe_filename}.xlsx"
    
    # Export to Excel
    df.to_excel(filename, sheet_name ="values", index=False)
    
    print(f"Exported outflows {CE_name} as {filename} (shape: {df.shape})")