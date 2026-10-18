import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import size_kernel as kernel
from analysis_input import csv

def group_job(job):
    scope,q,members,water=job
    indexed=water.set_index(['OasisID','calendar_year']).sort_index()
    num,den=kernel.group_matrices(members,indexed); strata=kernel.strata_positions(members,scope)
    denominator=den.sum(axis=0); point=np.divide(num.sum(axis=0),denominator,out=np.full(38,np.nan),where=denominator>0)
    annual=[]
    for yi,year in enumerate(kernel.YEARS):
        draws=kernel.bootstrap_annual(num,den,strata,yi,f'ANNUAL|{scope}|{q}|{year}')
        lo,hi,valid,quality=kernel.interval(draws)
        annual.append(dict(scope=scope,global_area_quintile=q,calendar_year=int(year),nominal_oasis_count=len(members),eligible_oasis_count=int((den[:,yi]>0).sum()),nominal_fixed_area_km2=float(members.area_km2.sum()),eligible_fixed_area_km2=float(denominator[yi]),fixed_area_coverage=float(denominator[yi]/members.area_km2.sum()),area_weighted_mean=float(point[yi]),ci95_lower=lo,ci95_upper=hi,valid_replicates=valid,interval_status=quality))
    boot=kernel.bootstrap_series(num,den,strata,f'TREND|{scope}|{q}'); slopes=kernel.row_slopes(boot)
    lo,hi,valid,quality=kernel.interval(slopes)
    trend=dict(scope=scope,global_area_quintile=q,eligible_year_count=int(np.isfinite(point).sum()),centered_year_slope=kernel.slope_1d(point),ci95_lower=lo,ci95_upper=hi,valid_replicates=valid,interval_status=quality)
    return annual,trend,point,boot,os.getpid()

def main():
    p=argparse.ArgumentParser();p.add_argument('--water',required=True);p.add_argument('--area',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    area=csv(a.area).rename(columns={'oasis_id':'OasisID'});area.area_km2=pd.to_numeric(area.area_km2)
    ranked=area.sort_values(['area_km2','OasisID'],kind='stable').reset_index(drop=True)
    ranked['global_area_quintile']='Q'+(1+np.arange(len(ranked))*5//len(ranked)).astype(str)
    water=csv(a.water);water.calendar_year=pd.to_numeric(water.calendar_year).astype(int)
    water['value']=pd.to_numeric(water.water_support_ratio if 'water_support_ratio' in water else water.final_available_water_ratio,errors='coerce');water['eligible']=np.isfinite(water.value)
    if len(area)!=3437 or area.OasisID.nunique()!=3437 or set(area.OasisID)!=set(water.OasisID) or not np.isfinite(area.area_km2).all() or (area.area_km2<=0).any():
        raise ValueError('Matching 3437-object fixed-area membership required')
    if len(water)!=130606 or water.duplicated(['OasisID','calendar_year']).any() or set(water.calendar_year)!=set(kernel.YEARS) or not water.eligible.all() or ((water.value<0)|(water.value>1)).any():
        raise ValueError('Complete 3437 by 38 baseline proportions required')
    if set(area.continent)!=set(kernel.SCOPES)-{'GLOBAL'}:
        raise ValueError('The five continent labels must match the fixed analysis scopes')
    jobs=[]
    for scope in kernel.SCOPES:
        for q in kernel.QUINTILES:
            members=kernel.group_ids(ranked,scope,q); part=water[water.OasisID.isin(set(members.OasisID))].copy();jobs.append((scope,q,members,part))
    annual=[];trends=[];series={};boots={};pids=set()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for job,result in zip(jobs,pool.map(group_job,jobs)):
            aa,tt,ss,bb,pid=result;annual.extend(aa);trends.append(tt);series[(job[0],job[1])]=ss;boots[(job[0],job[1])]=bb;pids.add(pid)
    contrasts=[];paired=[]
    for scope in kernel.SCOPES:
        difference=series[(scope,'Q5')]-series[(scope,'Q1')]
        bs=boots[(scope,'Q5')]-boots[(scope,'Q1')]
        lo,hi,valid,quality=kernel.interval(kernel.row_slopes(bs))
        contrasts.append(dict(scope=scope,paired_year_count=int(np.isfinite(difference).sum()),Q5_minus_Q1_trend_contrast=kernel.slope_1d(difference),ci95_lower=lo,ci95_upper=hi,valid_replicates=valid,interval_status=quality))
        paired.extend(dict(scope=scope,calendar_year=int(year),Q5_minus_Q1=float(value)) for year,value in zip(kernel.YEARS,difference))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('SIZE_GRADIENT_ANNUAL',annual),('SIZE_GRADIENT_TRENDS',trends),('SIZE_GRADIENT_ENDPOINT_CONTRASTS',contrasts),('SIZE_GRADIENT_PAIRED_ANNUAL',paired)]:pd.DataFrame(rows).to_csv(out/(name.lower()+'.csv'),index=False)


if __name__=='__main__':
    main()
