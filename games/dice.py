from numpy import random, sum


def roll_a_die(n):
    """this function returns a value 1-n, add 1 so the number start from 1"""
    return random.randint(n) + 1