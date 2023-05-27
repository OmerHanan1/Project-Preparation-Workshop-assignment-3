import unittest
from algorithms import prediction, getRowFromData


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

    def test_positive_prediction_result_inaccurate(self):
        """
        test case: try to predict two teams game score with valid.
        input: team1='Chelsea', team2='Liverpool', match_date='2015-05-10', algorithm='RFC'
        output: should return a tuple with ('Chelsea wins', 'Draw')
        """
        res = prediction(team1='Chelsea', team2='Liverpool',
                         match_date='2015-05-10', algorithm='RFC')
        self.assertEqual(
            res, ('Chelsea wins', 'Draw'), "should return ('Chelsea wins', 'Draw')")

    def test_negative_prediction_bad_date(self):
        """
        test case: try to extract row from dataset with bad input.
        input: team1='Chelsea', team2='Liverpool', match_date='2015-05-01', algorithm='MLP'
        output: should return (None, None)
        """
        with self.assertRaises(Exception):
            res = prediction(team1='Chelsea', team2='Liverpool',
                             match_date='2015-05-01', algorithm='MLP')

    def test_positive_prediction_result_accurate(self):
        """
        test case: try to extract row teams game score with valid input.
        input: team1='Chelsea', team2='Manchester United', match_date='2015-04-18', algorithm='DTC' 
        output: should return a variable of type pd.core.frame.DataFrame
        """

        res = prediction(team1='Chelsea', team2='Manchester United',
                         match_date='2015-04-18', algorithm='DTC')
        self.assertEqual(
            res, ('Chelsea wins', 'Chelsea wins'), "should return ('Chelsea wins', 'Chelsea wins')")


# Run the tests
if __name__ == '__main__':
    unittest.main()
