from pathlib import Path
import numpy as np
import pandas as pd

def csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def annual_input(paths, membership):
    frames = [csv(Path(p)) for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['OasisID', 'calendar_year']).any():
        raise ValueError('Duplicate annual identities')
    preferred = ['analysis_value', 'water_support_ratio', 'water_ratio', 'final_available_water_ratio',
                 'source_final_available_water_ratio']
    value = pd.Series(np.nan, index=frame.index, dtype=float)
    used = pd.Series('', index=frame.index, dtype=str)
    for field in preferred:
        if field not in frame:
            continue
        candidate = pd.to_numeric(frame[field], errors='coerce')
        take = value.isna() & np.isfinite(candidate)
        value.loc[take] = candidate.loc[take]
        used.loc[take] = field
    if ((value.dropna() < 0) | (value.dropna() > 1)).any():
        raise ValueError('Water-support ratio outside its domain')
    frame['analysis_value'] = value
    frame['analysis_source_field'] = used
    frame['calendar_year'] = pd.to_numeric(frame['calendar_year']).astype(int)
    meta = csv(membership)
    meta['fixed_2020_area_km2'] = pd.to_numeric(meta['fixed_2020_area_km2'])
    identities = set(frame.OasisID)
    if identities == set(meta.source_oasis_id):
        identity_field = 'source_oasis_id'
    elif identities == set(meta.corrected_oasis_id):
        identity_field = 'corrected_oasis_id'
    else:
        raise ValueError('Annual namespace does not match source or corrected membership')
    frame = frame.merge(meta, left_on='OasisID', right_on=identity_field,
                        how='left', validate='many_to_one')
    if frame['corrected_oasis_id'].isna().any():
        raise ValueError('Unmapped annual identity')
    return frame, meta
