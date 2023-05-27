import unittest
from algorithms import prediction, getRowFromData
import pandas as pd


class TestApplication(unittest.TestCase):
    def test_negative_predication_one_team_bad_name(self):
        """
        test case: try to predict two teams game score but one team does not exist.
        input: team1='Chelsea', team2='InvalidTeam', match_date='2015-05-10', algorithm='RFC'
        output: should throw exception
        """
        with self.assertRaises(Exception):
            res = prediction(team1='Chelsea', team2='InvalidTeam',
                             match_date='2015-05-10', algorithm='RFC')

    def test_positive_prediction_teams_with_good_names(self):
        """
        test case: try to predict two teams game score with valid.
        input: team1='Chelsea', team2='Liverpool', match_date='2015-05-10', algorithm='RFC'
        output: should return a tuple with ('Chelsea wins', 'Draw')
        """
        res = prediction(team1='Chelsea', team2='Liverpool',
                         match_date='2015-05-10', algorithm='RFC')
        self.assertEqual(
            res, ('Chelsea wins', 'Draw'), "should return ('Chelsea wins', 'Draw')")

    def test_negative_getRowFromData_bad_date(self):
        """
        test case: try to extract row from dataset with bad input.
        input: team1='Chelsea', team2='Liverpool', match_date='2015-05-01'
        output: should return (None, None)
        """
        res = getRowFromData(team1='Chelsea', team2='Liverpool',
                             match_date='2015-05-01')
        res = res[0]
        self.assertEqual(res, None, "should return None")

    def test_positive_getRowFromData_return_type(self):
        """
        test case: try to extract row teams game score with valid input.
        input: team1='Leicester City', team2='Chelsea', match_date='2015-04-29'
        output: should return a variable of type pd.core.frame.DataFrame
        """
        res = getRowFromData(team1='Leicester City', team2='Chelsea',
                             match_date='2015-04-29')
        res = res[0]
        self.assertEqual(type(res), pd.core.frame.DataFrame,
                         "should return 'pd.core.frame.DataFrame'")


# Run the tests
if __name__ == '__main__':
    unittest.main()
