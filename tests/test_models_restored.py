import numpy as np
from sports_edge.models.tennis import EloState,matchup_probability
from sports_edge.models.baseball import at_least_one_hit_probability,poisson_game_win_probability
from sports_edge.models.nfl import normal_over_probability
from sports_edge.models.parlay import independent_joint,correlated_joint_monte_carlo

def test_elo_symmetry():
    assert abs(EloState.win_prob(1500,1500)-.5)<1e-12
    assert matchup_probability(1700,1500)>matchup_probability(1500,1700)

def test_hit_probability_monotonic():
    assert at_least_one_hit_probability(.3,4)>at_least_one_hit_probability(.2,4)

def test_baseball_home_better():
    assert poisson_game_win_probability(5.0,3.5,20000)>.5

def test_prop_over():
    assert normal_over_probability(100,10,90)>.5

def test_parlay_independent():
    assert abs(independent_joint([.5,.5,.5])-.125)<1e-12

def test_corr_joint_positive_corr_increases_all_success_for_balanced_legs():
    p=correlated_joint_monte_carlo([.6,.6],[[1,.4],[.4,1]],30000,2)
    assert p>.36
