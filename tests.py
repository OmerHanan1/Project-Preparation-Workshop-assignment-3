import unittest
from main import calculate


class TestApplication:
    def test_negative_one_team_with_unavailable_name(self):
        res = calculate(team1='Chelsea', team2='InvalidTeam', algorithm='RFC')
        assertEquals(res, None, "should return None when one team is invalid")

    def test_positive_1(self):
        pass

    def test_positive_2(self):
        pass


# Run the tests
if __name__ == '__main__':
    unittest.main()
