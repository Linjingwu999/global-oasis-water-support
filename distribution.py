import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_input import csv

def slope(values,years):
    values=np.asarray(values,float); good=np.isfinite(values); n=good.sum(axis=-1)
    x=np.asarray(years,float)-np.mean(years); sx=np.sum(good*x,axis=-1); sy=np.sum(np.where(good,values,0),axis=-1)
    numerator=np.sum(np.where(good,values*x,0),axis=-1)-np.divide(sx*sy,n,out=np.full_like(sy,np.nan),where=n>0)
    denominator=np.sum(good*x*x,axis=-1)-np.divide(sx*sx,n,out=np.full_like(sy,np.nan),where=n>0)
    return np.divide(numerator,denominator,out=np.full_like(sy,np.nan),where=(n>=2)&(denominator>0))

def gini(v):
    v=np.sort(v[np.isfinite(v)])
    if not len(v):return np.nan
    total=v.sum()
    if total==0:return 0.
    return float(np.sum((2*np.arange(1,len(v)+1)-len(v)-1)*v)/(len(v)*total))

def weighted_quantile(v,weights,q):
    ok=np.isfinite(v); x=v[ok]; w=weights[:,ok]; order=np.argsort(x,kind='stable'); x=x[order]; w=w[:,order]
    if not len(x):return np.full(len(weights),np.nan)
    cumulative=np.cumsum(w,axis=1); total=cumulative[:,-1]; h=(total-1)*q; lower=np.floor(h).astype(int); upper=np.ceil(h).astype(int)
    li=np.argmax(cumulative>lower[:,None],axis=1); ui=np.argmax(cumulative>upper[:,None],axis=1)
    answer=x[li]+(h-lower)*(x[ui]-x[li]); answer[total<=0]=np.nan; return answer

def intervals(estimate,draws):
    finite=draws[np.isfinite(draws)]
    if len(finite)<2:return dict(ci_low=np.nan,ci_high=np.nan,valid_repetitions=len(finite),p_value=np.nan,inference_status='INSUFFICIENT_FINITE_REPLICATES')
    lo,hi=np.quantile(finite,[.025,.975]); sd=finite.std(ddof=1)
    # Null-centred bootstrap tail probability; finite points survive low valid counts.
    p=(1+np.sum(np.abs(finite-finite.mean())>=abs(estimate)))/(len(finite)+1) if np.isfinite(estimate) and sd>0 else np.nan
    return dict(ci_low=float(lo),ci_high=float(hi),valid_repetitions=len(finite),p_value=p,
                inference_status='BOOTSTRAP_AVAILABLE' if np.isfinite(p) else 'POINT_ONLY_DEGENERATE_BOOTSTRAP')

def adjust_family(frame,columns,method='BH'):
    frame['q_value']=np.nan; frame['effective_m']=0; frame['adjustment_method']=method
    families=frame.groupby(columns,sort=False).groups.values() if columns else [frame.index]
    for index in families:
        p=pd.to_numeric(frame.loc[index,'p_value'],errors='coerce')
        valid=p.index[np.isfinite(p)]; m=len(valid)
        frame.loc[index,'effective_m']=m
        if not m:continue
        order=p.loc[valid].sort_values(kind='stable').index
        ranked=p.loc[order].to_numpy(float)
        if method=='Holm': q=np.maximum.accumulate(ranked*(m-np.arange(m)))
        else: q=np.minimum.accumulate((ranked*m/np.arange(1,m+1))[::-1])[::-1]
        frame.loc[order,'q_value']=np.minimum(q,1.)
    return frame

def group_job(job):
    dimension,group,analysis,original,areas,years=job; n=len(areas)
    seed=int.from_bytes(hashlib.sha256(f'20260810|{dimension}|{group}'.encode()).digest()[:8],'little')
    rng=np.random.Generator(np.random.PCG64(seed)); weights=rng.multinomial(n,np.full(n,1/n),size=2000).astype(float)
    annual=[]; trends=[]; series={}; draws={}; distributions=[]
    for spec,data in [('ANALYSIS',analysis),('ORIGINAL_DIAGNOSTIC',original)]:
        valid=np.isfinite(data); count=valid.sum(axis=0)
        mean=np.divide(np.nansum(data,axis=0),count,out=np.full(len(years),np.nan),where=count>0)
        denominator=weights@valid.astype(float); numerator=weights@np.nan_to_num(data,nan=0.)
        bootstrap_mean=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        series[spec]=mean; draws[spec]=slope(bootstrap_mean,years)*100
        for i,year in enumerate(years):
            v=data[:,i]; finite=v[np.isfinite(v)]; q=np.quantile(finite,[.1,.5,.9]) if len(finite) else [np.nan]*3
            annual.append(dict(grouping_dimension=dimension,group_value=group,sample_spec=spec,calendar_year=int(year),n_nominal=n,n_numeric=int(count[i]),coverage=float((valid[:,i]*areas).sum()/areas.sum()),mean=mean[i],p10=q[0],median=q[1],p90=q[2],gini=gini(v),status='ESTIMATED' if len(finite) else 'NO_FINITE_VALUE'))
        estimate=float(slope(mean,years)*100)
        trends.append(dict(grouping_dimension=dimension,group_value=group,sample_spec=spec,metric='mean_slope',estimate_pp_per_year=estimate,n_finite_years=int(np.isfinite(mean).sum()),**intervals(estimate,draws[spec])))
    paired=draws['ANALYSIS']-draws['ORIGINAL_DIAGNOSTIC']; point=float((slope(series['ANALYSIS'],years)-slope(series['ORIGINAL_DIAGNOSTIC'],years))*100)
    sensitivity=dict(grouping_dimension=dimension,group_value=group,comparison='analysis_minus_original_diagnostic',estimate_pp_per_year=point,**intervals(point,paired))
    if dimension=='GLOBAL':
        # Entire OasisID bootstrap weights are reused across every calendar year.
        metrics={name:np.full((2000,len(years)),np.nan) for name in ['p10','median','p90','p90_minus_p10','gini']}
        point_values={name:[] for name in metrics}
        for j in range(len(years)):
            v=analysis[:,j]; ok=np.isfinite(v); x=v[ok]; order=np.argsort(x,kind='stable'); x=x[order]; w=weights[:,ok][:,order]
            totalw=w.sum(axis=1); totalx=w@x; cumulative=np.cumsum(w,axis=1)
            numerator=(w*(2*cumulative-w-totalw[:,None])*x).sum(axis=1)
            bg=np.divide(numerator,totalw*totalx,out=np.full(2000,np.nan),where=(totalw>0)&(totalx>0)); bg[(totalw>0)&(totalx==0)]=0
            metrics['gini'][:,j]=bg
            for name,q in [('p10',.1),('median',.5),('p90',.9)]:metrics[name][:,j]=weighted_quantile(v,weights,q)
            metrics['p90_minus_p10'][:,j]=metrics['p90'][:,j]-metrics['p10'][:,j]
            q=np.quantile(x,[.1,.5,.9]) if len(x) else [np.nan]*3
            for name,value in zip(['p10','median','p90','p90_minus_p10','gini'],[q[0],q[1],q[2],q[2]-q[0],gini(v)]):point_values[name].append(value)
        for metric,values in point_values.items():
            point=float(slope(values,years)*100); b=slope(metrics[metric],years)*100
            distributions.append(dict(grouping_dimension='GLOBAL',group_value='GLOBAL',metric=metric+'_slope',estimate_pp_per_year=point,**intervals(point,b)))
    return annual,trends,sensitivity,distributions,draws['ANALYSIS'],float(slope(series['ANALYSIS'],years)*100),os.getpid()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--water',required=True)
    p.add_argument('--area',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    w=csv(a.water);w.calendar_year=pd.to_numeric(w.calendar_year).astype(int)
    field='water_support_ratio' if 'water_support_ratio' in w else 'final_available_water_ratio'
    w['value']=pd.to_numeric(w[field],errors='coerce')
    ar=csv(a.area).set_index('oasis_id');ids=sorted(set(w.OasisID));years=np.arange(1984,2022)
    if not ar.index.is_unique or set(ar.index)!=set(ids):
        raise ValueError('One matching area row per baseline object required')
    values=w.pivot(index='OasisID',columns='calendar_year',values='value').reindex(index=ids,columns=years).to_numpy(float)
    cls=w.pivot(index='OasisID',columns='calendar_year',values='repair_class').reindex(index=ids,columns=years).to_numpy()
    original=np.where(cls=='ORIGINAL_VALID',values,np.nan)
    areas=pd.to_numeric(ar.loc[ids,'area_km2']).to_numpy()
    if len(w)!=130606 or len(ids)!=3437 or set(w.calendar_year)!=set(years) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Complete 3437 by 38 baseline proportions required')
    if not np.isfinite(areas).all() or (areas<=0).any():
        raise ValueError('Positive finite fixed areas required')
    annual,trends,sensitivity,distribution,_,_,_=group_job(('GLOBAL','GLOBAL',values,original,areas,years))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(annual).to_csv(out/'annual_distributions.csv',index=False)
    for name,rows,columns in [('mean_trends',trends,['grouping_dimension','sample_spec']),('source_sensitivity',[sensitivity],['grouping_dimension']),('global_distribution_trends',distribution,[])]:
        adjust_family(pd.DataFrame(rows),columns).to_csv(out/(name+'.csv'),index=False)

if __name__=='__main__':
    main()
