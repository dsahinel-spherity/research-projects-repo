"""DPP links follow recorded paths without changing journey boundaries."""
import unittest
from unittest.mock import patch

import pandas as pd

from app.services.journey_service import build_user_journeys, recorded_dpp_location


class JourneyLinkTests(unittest.TestCase):
    def location(self, path, **extra):
        frame = pd.DataFrame([dict(dpp_id='gmh-dpp', url_path=path, **extra)])
        with patch('app.services.journey_service.DPP_VIEWER_BASE_URL', 'https://dpp.vera.spherity.dev'):
            return recorded_dpp_location(frame, 'gmh-dpp', {})

    def test_recorded_path_preserves_encoding_and_omits_tracking_query(self):
        path = '/did:web:api.vera.spherity.dev:did-registry:gmh%3Aexample'
        self.assertEqual(self.location(path + '?source=qr'), ('https://dpp.vera.spherity.dev' + path, path))

    def test_absolute_url_and_host_metadata(self):
        self.assertEqual(self.location('https://viewer.example/dpp?id=1')[0], 'https://viewer.example/dpp')
        self.assertEqual(self.location('/dpp', hostname='recorded.example')[0], 'https://recorded.example/dpp')

    def test_missing_or_unsafe_path_is_not_a_link(self):
        for path in ['', 'javascript:alert(1)', '//other.example/dpp']:
            self.assertEqual(self.location(path), ('', ''))

    def test_multiple_dpps_preserve_single_journey_and_link_first_dpp(self):
        events = pd.DataFrame([
            dict(id='entry', sessionId='s', createdAt='2026-10-01T10:00:00Z', eventName=None, urlPath='/'),
            dict(id='a', sessionId='s', createdAt='2026-10-01T10:01:00Z', eventName='dpp-section-view', dpp_id='gmh-a', urlPath='/did:gmh-a', section='Overview'),
            dict(id='b', sessionId='s', createdAt='2026-10-01T10:02:00Z', eventName='dpp-section-view', dpp_id='gmh-b', urlPath='/did:gmh-b', section='Documents'),
        ])
        with patch('app.services.journey_service.DPP_VIEWER_BASE_URL', 'https://dpp.vera.spherity.dev'):
            journeys = build_user_journeys(events, pd.DataFrame())
        self.assertEqual(len(journeys), 1)
        self.assertEqual(journeys.iloc[0].dpp_count, 2)
        self.assertEqual(journeys.iloc[0].dpp_id, 'gmh-a')
        self.assertEqual(journeys.iloc[0].dpp_url, 'https://dpp.vera.spherity.dev/did:gmh-a')


if __name__ == '__main__':
    unittest.main()
