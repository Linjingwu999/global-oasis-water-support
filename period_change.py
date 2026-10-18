import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_input import annual_input

def bootstrap_chunk(payload):
    start, stop, ids, area, delta, equal_delta, group_codes, group_count = payload
    draws = np.empty((stop-start, group_count*2))
    for pos, rep in enumerate(range(start, stop)):
        token=f'20260821|M1_FIXED_AREA_DELTA_A_1987_2024|{rep}'.encode()
        rng=np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(token).digest()[:16],'big')))
        take=rng.integers(0,len(ids),len(ids))
        count=np.bincount(take,minlength=len(ids))
        total=[]
        for codes in group_codes:
            total.extend(np.bincount(codes,weights=count*delta,minlength=int(codes.max())+1))
        for codes in group_codes:
            total.extend(np.bincount(codes,weights=count*equal_delta,minlength=int(codes.max())+1))
        draws[pos]=total
    return start, draws, os.getpid()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True)
    ap.add_argument('--membership',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    f,m=annual_input(args.inputs,args.membership)
    if len(f)!=130606 or f.OasisID.nunique()!=3437 or set(f.calendar_year)!=set(range(1987,2025)):
        raise ValueError('Annual universe must be 3437 by 38')
    m=m.sort_values('source_oasis_id').reset_index(drop=True); ids=m.source_oasis_id.tolist()
    values=f.pivot(index='source_oasis_id',columns='calendar_year',values='analysis_value').reindex(index=ids,columns=range(1987,2025)).to_numpy(float)
    area=m.fixed_2020_area_km2.to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(area).all() or (area<=0).any():
        raise ValueError('Complete annual proportions and positive finite fixed areas required')
    p0=np.nansum(values[:,:19],axis=1)/19; p1=np.nansum(values[:,19:],axis=1)/19
    delta=area*(p1-p0)
    dimensions=[('GLOBAL',np.full(len(m),'GLOBAL')),('continent5',m.continent5.to_numpy())]
    labels=[]; codes=[]; denominator=[]; points=[]; period0=[]; period1=[]; ns=[]
    for dim, vals in dimensions:
        levels=sorted(set(vals)); index={v:i for i,v in enumerate(levels)}
        code=np.array([index[x] for x in vals]); codes.append(code)
        den=np.bincount(code,weights=area,minlength=len(levels)); denominator.extend(den)
        points.extend(np.bincount(code,weights=delta,minlength=len(levels)))
        period0.extend(np.bincount(code,weights=area*p0,minlength=len(levels)))
        period1.extend(np.bincount(code,weights=area*p1,minlength=len(levels)))
        ns.extend(np.bincount(code,minlength=len(levels))); labels.extend((dim,v) for v in levels)
    denominator=np.asarray(denominator); point=np.asarray(points)/denominator
    draws=np.empty((2000,len(labels)*2)); pids=set()
    divisors=np.r_[denominator,np.asarray(ns)]
    payloads=[(a,min(a+100,2000),ids,area,delta,p1-p0,codes,len(labels)) for a in range(0,2000,100)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for start,result,pid in pool.map(bootstrap_chunk,payloads):
            draws[start:start+len(result)]=result/divisors; pids.add(pid)
    ci=np.quantile(draws,[.025,.975],axis=0)
    rows=[dict(grouping_dimension=dim,group_value=grp,oasis_count=int(ns[i]),fixed_area_km2=float(denominator[i]),
               p0_supported_record_index=float(period0[i]/denominator[i]),p1_supported_record_index=float(period1[i]/denominator[i]),
               within_change=float(point[i]),range_lower=float(ci[0,i]),range_upper=float(ci[1,i]),valid_replicates=2000,
               estimand='fixed_group_area_supported_record_index',uncertainty_method='fixed_original_group_denominator_reweighting')
          for i,(dim,grp) in enumerate(labels)]
    pd.DataFrame(rows).to_csv(out/'regional_area_change.csv',index=False)
    equal_rows=[]
    for i,(dimension,group) in enumerate(labels):
        code=codes[['GLOBAL','continent5'].index(dimension)]
        levels=sorted(set(dimensions[['GLOBAL','continent5'].index(dimension)][1]))
        mask=code==levels.index(group)
        equal_rows.append(dict(grouping_dimension=dimension,group_value=group,oasis_count=int(ns[i]),
                               p0_equal_oasis_mean=float(p0[mask].mean()),p1_equal_oasis_mean=float(p1[mask].mean()),
                               change=float((p1-p0)[mask].mean()),range_lower=float(ci[0,len(labels)+i]),range_upper=float(ci[1,len(labels)+i]),
                               valid_replicates=2000,estimand='equal_oasis_annual_analysis_mean'))
    pd.DataFrame(equal_rows).to_csv(out/'regional_equal_oasis_change.csv',index=False)

    total_area=area.sum()
    contributions=[dict(group_value=grp,area_share=float(denominator[i]/total_area),contribution_pp=float(100*point[i]*denominator[i]/total_area)) for i,(dim,grp) in enumerate(labels) if dim=='continent5']
    pd.DataFrame(contributions).to_csv(out/'regional_contributions.csv',index=False)


if __name__=='__main__':
    main()
