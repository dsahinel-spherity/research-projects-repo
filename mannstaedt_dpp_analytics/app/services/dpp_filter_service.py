"""Scope GMH and individual DPPs by event ownership, not session membership."""
from urllib.parse import unquote, urlsplit

import pandas as pd

CUSTOM_EVENTS = {'dpp-section-view', 'dpp-document-open'}


def canonical_dpp(value):
    if value is None or pd.isna(value):
        return ''
    value = str(value).strip()
    for _ in range(2):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    if value.startswith(('http://', 'https://')):
        value = urlsplit(value).path
    return value.split('?', 1)[0].split('#', 1)[0].strip('/')


def filter_selected_dpp(events, sessions, selected_dpp, organization):
    return filter_gmh_events(events, sessions, organization, selected_dpp)


def filter_gmh_events(events, sessions, organization, selected_dpp=None):
    """Keep owned custom events and attributable pageviews in either UI scope."""
    if events.empty:
        return events.copy(), sessions.iloc[:0].copy()
    data = events.copy()
    names = data.get('eventName', pd.Series('', index=data.index))
    ids = data.get('dpp_id', pd.Series('', index=data.index)).map(canonical_dpp)
    paths = data.get('urlPath', pd.Series('', index=data.index)).map(canonical_dpp)
    orgs = data.get('organization', pd.Series('', index=data.index)).fillna('')
    target = canonical_dpp(selected_dpp)
    custom = names.isin(CUSTOM_EVENTS)
    matching = custom & orgs.eq(organization) & ids.ne('')
    if selected_dpp is not None:
        matching &= ids.eq(target)
    target_ids = set(ids[matching])
    selected_sessions = data.loc[matching, 'sessionId'].dropna()

    # Paths must be attributable to this DPP; shared/root routes are not evidence.
    target_paths = set(paths[matching]) - {''}
    conflicting_paths = set(paths[custom & ~matching]) - {''}
    target_paths -= conflicting_paths
    target_paths.update(target_ids - conflicting_paths)
    if 'eventType' in data:
        pageviews = data.eventType.eq(1)
    else:
        pageviews = names.fillna('').eq('')
    matching_pageviews = (pageviews & paths.isin(target_paths)
                          & data.sessionId.isin(selected_sessions)
                          & (orgs.eq('') | orgs.eq(organization))
                          & (ids.eq('') | ids.isin(target_ids)))

    # Preserve boundaries from the complete session before hiding other DPPs.
    data['_scope_time'] = pd.to_datetime(data.createdAt, errors='coerce', utc=True)
    data = data.sort_values(['sessionId', '_scope_time'], kind='stable')
    gaps = data.groupby('sessionId', dropna=False)['_scope_time'].diff()
    data['_journey_segment'] = gaps.gt(pd.Timedelta(minutes=10)).groupby(data.sessionId, dropna=False).cumsum()
    scoped = data.loc[(matching | matching_pageviews).reindex(data.index)].drop(columns='_scope_time').copy()
    scoped_sessions = sessions[sessions.id.isin(selected_sessions)].copy() if 'id' in sessions else sessions.iloc[:0].copy()
    return scoped, scoped_sessions
