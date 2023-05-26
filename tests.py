import unittest
from algorithms import prediction


class TestApplication(unittest.TestCase):
    def test_negative_predication_one_team_with_bad_name(self):
        res = prediction(team1='Chelsea', team2='InvalidTeam',
                         match_date="2015-09-09", algorithm='RFC')
        self.assertEqual(
            res, None, "should return None when one team is invalid")

    def test_positive_prediction_teams_with_good_names(self):
        res = prediction(team1='Chelsea', team2='Liverpool',
                         match_date='2015-05-10', algorithm='RFC')
        self.assertEqual(
            res, ('Chelsea', 'Draw'), "should return ('Chelsea', 'Draw')")

    def test_positive_calculate_one_team_with_unavailable_name(self):
        pass


# Run the tests
if __name__ == '__main__':
    unittest.main()
