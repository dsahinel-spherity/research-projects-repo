import unittest
from urllib.parse import quote

import pandas as pd

from app.services.dpp_filter_service import filter_gmh_events, filter_selected_dpp
from app.services.analytics_service import real_company_kpis, section_performance, document_performance, dpp_performance
from app.services.journey_service import build_user_journeys

DID = 'did:web:api.vera.spherity.dev:did-registry:bundnummer-5634892233'
ORG = 'gmh-gruppe-d034406b'


def event(key, minute, dpp, name=None, section=None, path=None, organization=ORG):
    return dict(id=key, sessionId='shared', createdAt=pd.Timestamp('2026-10-01', tz='UTC')+pd.Timedelta(minutes=minute),
                dpp_id=dpp if name else None, eventName=name, eventType=2 if name else 1,
                section=section, document='Other DPP certificate' if name=='dpp-document-open' else None,
                organization=organization if name else None, urlPath=path or '/'+dpp)


class DppFilterTests(unittest.TestCase):
    def test_all_gmh_excludes_seven_foreign_document_opens(self):
        rows = [event('entry',0,DID), event('own',1,DID,'dpp-section-view','Product')]
        rows += [event(f'foreign-{i}', i+2, 'foreign-dpp', 'dpp-document-open', organization='other-org') for i in range(7)]
        rows += [event('unowned',9,'unknown','dpp-document-open',organization=None)]
        scoped, sessions = filter_gmh_events(pd.DataFrame(rows),pd.DataFrame([{'id':'shared'}]),ORG)
        self.assertEqual(set(scoped.id), {'entry','own'})
        self.assertEqual(real_company_kpis({},scoped,sessions)['document_opens'],0)
        self.assertTrue(document_performance(scoped).empty)
        self.assertEqual(dpp_performance(scoped).document_opens.sum(),0)
        self.assertFalse(build_user_journeys(scoped,sessions).opened_document.any())

    def test_future_owned_documents_count_in_both_scopes(self):
        rows = [event('own',1,DID,'dpp-section-view','Product'), event('document',2,DID,'dpp-document-open')]
        raw = pd.DataFrame(rows)
        all_events, sessions = filter_gmh_events(raw,pd.DataFrame(),ORG)
        selected, _ = filter_selected_dpp(raw,pd.DataFrame(),DID,ORG)
        for scoped in [all_events, selected]:
            self.assertEqual(real_company_kpis({},scoped,sessions)['document_opens'],1)
            self.assertEqual(document_performance(scoped).opens.sum(),1)
            self.assertTrue(build_user_journeys(scoped,sessions).opened_document.all())

    def test_no_documents_or_sections_from_other_dpps(self):
        rows = [event('entry',0,DID),event('own',1,DID,'dpp-section-view','Product'),
                event('other-view',2,'other'),event('other-section',3,'other','dpp-section-view','Other section'),
                event('other-doc',4,'other','dpp-document-open')]
        scoped,sessions=filter_selected_dpp(pd.DataFrame(rows),pd.DataFrame([{'id':'shared'}]),DID,ORG)
        self.assertEqual(set(scoped.id),{'entry','own'})
        self.assertEqual(real_company_kpis({},scoped,sessions)['document_opens'],0)
        self.assertEqual(section_performance(scoped).section.tolist(),['Product'])
        journeys=build_user_journeys(scoped,sessions)
        self.assertFalse(journeys.opened_document.any())
        self.assertEqual(journeys.iloc[0].document_open_count,0)

    def test_encoding_does_not_change_identity_or_split_continuous_session(self):
        rows=[event('a',0,DID,'dpp-section-view','Product'),
              event('b',6,'other','dpp-section-view','Other'),
              event('c',12,quote(DID,safe=''),'dpp-section-view','Product')]
        scoped,sessions=filter_selected_dpp(pd.DataFrame(rows),pd.DataFrame(),DID,ORG)
        self.assertEqual(set(scoped.id),{'a','c'})
        self.assertEqual(len(build_user_journeys(scoped,sessions)),1)

    def test_real_inactivity_still_splits(self):
        rows=[event('a',0,DID,'dpp-section-view','Product'),event('b',11,DID,'dpp-section-view','Product')]
        scoped,sessions=filter_selected_dpp(pd.DataFrame(rows),pd.DataFrame(),DID,ORG)
        self.assertEqual(len(build_user_journeys(scoped,sessions)),2)

    def test_wrong_organization_and_ambiguous_pageview_excluded(self):
        rows=[event('entry',0,DID,path='/shared'),event('own',1,DID,'dpp-section-view','Product',path='/shared'),
              event('other',2,'other','dpp-document-open',path='/shared'),
              event('foreign',3,DID,'dpp-document-open',organization='other-org')]
        scoped,_=filter_selected_dpp(pd.DataFrame(rows),pd.DataFrame(),DID,ORG)
        self.assertEqual(scoped.id.tolist(),['own'])


if __name__=='__main__':
    unittest.main()
