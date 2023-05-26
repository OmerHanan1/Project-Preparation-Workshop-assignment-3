import unittest
from main import calculate


class TestApplication(unittest.TestCase):
    def test_negative_calculate_one_team_with_unavailable_name(self):
        res = calculate(team1='Chelsea', team2='InvalidTeam',
                        match_date="2015-09-09", algorithm='RFC')
        print(res)
        self.assertEqual(
            res, None, "should return None when one team is invalid")

    def test_positive_calculate_one_team_with_unavailable_name(self):
        res = calculate(team1='Chelsea', team2='Liverpool',
                        match_date='2015-06-07', algorithm='RFC')
        print(res)
        self.assertEqual(
            res, None, "should return None when one team is invalid")

    def test_positive_2(self):
        pass


# Run the tests
if __name__ == '__main__':
    unittest.main()
