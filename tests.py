import unittest
from algorithms import prediction, getRowFromData
import pandas as pd


class TestApplication(unittest.TestCase):
    def test_negative_predication_one_team_with_bad_name(self):
        with self.assertRaises(Exception):
            res = prediction(team1='Chelsea', team2='InvalidTeam',
                             match_date="2015-09-09", algorithm='RFC')

    def test_positive_prediction_teams_with_good_names(self):
        res = prediction(team1='Chelsea', team2='Liverpool',
                         match_date='2015-05-10', algorithm='RFC')
        self.assertEqual(
            res, ('Chelsea wins', 'Draw'), "should return ('Chelsea wins', 'Draw')")

    def test_positive_getRowFromData_return_type(self):
        res = getRowFromData(team1='Chelsea', team2='Liverpool',
                             match_date='2015-05-10')
        res = res[0]
        self.assertEqual(type(res), pd.core.frame.DataFrame,
                         "should return 'pd.core.series.Series'")


# Run the tests
if __name__ == '__main__':
    unittest.main()
