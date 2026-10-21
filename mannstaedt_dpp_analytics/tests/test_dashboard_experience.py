"""Offline regression checks for the chart-first dashboard and synthetic boundary."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.services.lab_service import QUESTIONS, run_experiment


def real_fixture():
    events=[]
    for i in range(12):
        for minute, name, section, document in [(0, None, None, None),(1,'dpp-section-view',f'Section {i}',None),(2,'dpp-document-open',None,'Certificate')]:
            events.append(dict(id=f'{i}-{minute}', sessionId=str(i),createdAt=pd.Timestamp('2026-10-01',tz='UTC')+pd.Timedelta(minutes=minute),eventName=name,eventType=1 if name is None else 2,urlPath='/dpp',urlQuery='source=qr' if i%2 else '',organization='gmh-gruppe-d034406b' if name else None,dpp_id='gmh-dpp' if name else None,section=section,document=document,source='qr' if i%2 else 'web'))
    return {'events':pd.DataFrame(events),'sessions':pd.DataFrame([dict(id=str(i),device='mobile',country='DE',browser='Chrome',os='Android') for i in range(12)]),'stats':{}}


class DashboardExperienceTests(unittest.TestCase):
    def test_business_pages_render_charts_with_optional_tables(self):
        with patch('app.services.data_service.load_real_analytics',return_value=real_fixture()):
            app=AppTest.from_file('app/dashboard.py',default_timeout=30).run()
            for page in ['Overview','Engagement','User Journey & Traceability']:
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual(len(app.exception),0, [str(e) for e in app.exception])
                self.assertGreater(len(app.get('arrow_vega_lite_chart')),0)
                self.assertGreater(len(app.expander),0)

    def test_synthetic_pages_survive_umami_failure(self):
        import streamlit as st
        st.cache_data.clear()
        with patch('app.services.data_service.load_real_analytics',side_effect=RuntimeError('Offline')) as loader:
            app=AppTest.from_file('app/dashboard.py',default_timeout=30).run()
            calls = loader.call_count
            app.sidebar.radio[0].set_value('Analytics Lab').run()
            self.assertEqual(loader.call_count, calls)
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('SYNTHETIC DATA' in w.value for w in app.warning))
            app.sidebar.radio[0].set_value('Dataset Explorer').run()
            self.assertEqual(len(app.exception),0)

    def test_guided_reports_render_and_persist(self):
        with patch('app.services.data_service.load_real_analytics',return_value=real_fixture()):
            app=AppTest.from_file('app/dashboard.py',default_timeout=30).run()
            app.sidebar.radio[0].set_value('Analytics Lab').run()
            self.assertFalse(any(e.label.startswith('Cleaning and filtering') for e in app.expander))
            for question in QUESTIONS:
                next(s for s in app.selectbox if s.label=='What would you like to understand?').set_value(question)
                next(b for b in app.button if b.label=='Run synthetic experiment').click().run()
                self.assertEqual(len(app.exception),0,[str(e) for e in app.exception])
                self.assertTrue(any(s.value==question for s in app.subheader))
                self.assertTrue(any(e.label.startswith('Cleaning and filtering') for e in app.expander))
                self.assertGreater(len(app.get('arrow_vega_lite_chart')),0)
            app.run()
            self.assertEqual(len(app.exception),0)

    def test_empty_real_data_is_handled(self):
        with patch('app.services.data_service.load_real_analytics',return_value={'events':pd.DataFrame(),'sessions':pd.DataFrame(),'stats':{}}):
            import streamlit as st
            st.cache_data.clear()
            app=AppTest.from_file('app/dashboard.py',default_timeout=30).run()
            for page in ['Overview','Engagement','User Journey & Traceability']:
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual(len(app.exception),0)

    def test_advanced_classifiers_support_nonbinary_labels(self):
        from app.services.engine_service import run_engine
        for labels in [[2, 3], [0, 1, 2]]:
            frame=pd.DataFrame({'x':list(range(90)), 'target':[labels[i%len(labels)] for i in range(90)]})
            for algorithm in ['logistic-regression','random-forest-classification']:
                output=run_engine(frame,algorithm,'predict',included_columns=['x'],target_columns=['target'],config={'n_estimators':10})
                self.assertEqual(output['result']['metric_average'],'weighted')
                self.assertTrue(np.isfinite(output['result']['f1']))

    def test_synthetic_targets_and_leakage(self):
        for question in QUESTIONS:
            result=run_experiment(question, model_name='Linear / logistic baseline challenger')
            report = result['preparation']
            self.assertEqual(report['original_rows'] - report['duplicates_removed'] - report['missing_target_rows_removed'], report['retained_rows'])
            self.assertEqual(report['fit_rows'] + report['evaluation_rows'], report['retained_rows'])
            self.assertEqual(report['inputs'], result['features'])
            self.assertEqual(report['missing_values_filled'], sum(r['Missing values filled'] for r in report['replacements']))
            self.assertGreater(report['missing_values_filled'], 0)
            if result['kind'] != 'anomaly':
                self.assertNotIn(result['target'], result['features'])
                for forbidden in ['document_open_count','documents_opened','deep_engagement','journey_score']:
                    self.assertNotIn(forbidden,result['features'])
                self.assertTrue(all(np.isfinite(v) for v in result['metrics'].values()))
                self.assertGreater(result['test_rows'],0)
                self.assertEqual(len(result['predictions']),result['test_rows'])
            else:
                self.assertGreater(result['flagged'],0)


if __name__=='__main__':
    unittest.main()
