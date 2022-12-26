import unittest
import pandas as pd
from feature_eng import load_dataset, build_features, filter_input_date

def run_test(tcls):
    
    suite = unittest.TestLoader().loadTestsFromTestCase(tcls)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
@run_test
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cooking_metrics = load_dataset(filename="source/cooking_metrics.csv", parse_dates=['timestamp'])
        cls.batch_registry = load_dataset(filename="source/batch_registry.csv")
        cls.faulty_intervals = load_dataset(filename="source/faulty_intervals.csv", parse_dates=['start_time', 'end_time'])
    
    def test_user_input_filter(self):
        
        start_date = "2020-11-01T00:23:34"
        end_date = "2020-11-01T01:23:33"
        
        res = filter_input_date(self.cooking_metrics, start_date, end_date)
        self.assertTrue(res.empty, 'wrong filter')
    
    def test_build_features_m1_a1(self):
        
        # Arrange
        expectedRes = pd.DataFrame([{
            "timestamp": "2020-11-01T01:00:00",
            "machine_id": "m1",
            "arepa_type": "a1",
            "metric_1": 0.1282,
            "metric_2": 0.8135,
            }])
        
        expectedRes.index = expectedRes['timestamp']
        expectedRes.drop('timestamp', axis=1, inplace=True)
        MACHINE_ID = "m1"
        AREPA_TYPE = "a1"
        start_date = "2020-11-01T00:23:34"
        end_date = "2020-11-01T02:23:33"
        
        #Act
        filtered = filter_input_date(self.cooking_metrics, start_date, end_date)
        res = build_features(filtered, self.batch_registry, self.faulty_intervals, MACHINE_ID, AREPA_TYPE)
        
        #Assert
        self.assertTrue(res.equals(expectedRes), 'error in build features')