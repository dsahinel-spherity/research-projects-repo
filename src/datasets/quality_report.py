"""Transparent checks on prepared records; parser issues remain a separate source-level report."""
import numpy as np
import pandas as pd


def build_quality_report(frame, volume='Volume_ob'):
    if volume not in ['Volume_ob','Volume_ub','Volume_price']:
        raise ValueError('Choose a supported volume definition.')
    d = frame.reset_index(drop=True).copy()
    fields = ['TreeID','LogKey','ProductKey','LogLength','ButtDiameter_ob','TopDiameter_ob',volume,
              'CuttingDuration','FellingLatitude','FellingLongitude','FellingTime','DropOffLatitude','DropOffLongitude','DropOffTime']
    completeness = pd.DataFrame([{'Field': f, 'Missing': int(d[f].isna().sum()) if f in d else len(d), 'Rows': len(d)} for f in fields])
    completeness['AvailablePercent'] = 100*(completeness.Rows-completeness.Missing)/len(d) if len(d) else np.nan
    issues = []
    def add(mask, check, level, detail):
        for i in d.index[mask.fillna(False)]:
            issues.append({'Row': int(i), 'TreeID': d.loc[i].get('TreeID'), 'LogKey': d.loc[i].get('LogKey'), 'Check': check, 'Level': level, 'Detail': detail})
    add(d[['TreeID','LogKey']].isna().any(axis=1), 'Missing identity', 'Error', 'TreeID and LogKey are needed to identify a record.')
    add(d.duplicated(['TreeID','LogKey'], keep=False), 'Duplicate identity', 'Error', 'Repeated TreeID and LogKey; review source before counting logs.')
    for f in ['LogLength','ButtDiameter_ob','TopDiameter_ob','ButtDiameter_ub','TopDiameter_ub','CuttingDuration',volume]:
        x = pd.to_numeric(d.get(f, pd.Series(np.nan,index=d.index)), errors='coerce')
        add(x.isna() | ~np.isfinite(x), 'Missing/nonfinite '+f, 'Gap', 'Unavailable measurement; not automatically an erroneous log.')
        add(x.lt(0), 'Negative '+f, 'Error', 'Negative physical measurement or duration.')
        if f == 'CuttingDuration':
            add(x.eq(0), 'Zero cutting duration', 'Review', 'Zero recorded duration is excluded from the cutting benchmark; check timestamp resolution or source events.')
        if f == 'LogLength' or 'Diameter' in f:
            add(x.eq(0), 'Zero '+f, 'Review', 'Zero dimension needs source review.')
    for prefix in ['Felling','DropOff']:
        for axis,limit in [('Latitude',90),('Longitude',180)]:
            f=prefix+axis
            x=pd.to_numeric(d.get(f,pd.Series(np.nan,index=d.index)),errors='coerce')
            add(x.notna() & (~np.isfinite(x) | x.abs().gt(limit)), 'Invalid '+f, 'Error', 'Coordinate outside WGS84 bounds.')
    issues=pd.DataFrame(issues,columns=['Row','TreeID','LogKey','Check','Level','Detail'])
    return {'completeness':completeness,'issues':issues,
            'summary':{'rows':len(d),'rows_with_issues':int(issues.Row.nunique()),'issue_count':len(issues)}}
