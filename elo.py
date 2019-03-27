import math

k_factor = 32


def update_elo(member_one, member_two, score_one, score_two):
    expectation_one, expectation_two = expected_scores(
        rating_one=member_one.elo,
        rating_two=member_two.elo
    )

    member_one.elo = member_one.elo + k_factor * (score_one - expectation_one)
    member_two.elo = member_two.elo + k_factor * (score_two - expectation_two)


def expected_scores(rating_one, rating_two):
    score_one = 1 / (1 + math.pow(10, (rating_two - rating_one) / 400))
    score_two = 1 / (1 + math.pow(10, (rating_one - rating_two) / 400))

    return score_one, score_two
