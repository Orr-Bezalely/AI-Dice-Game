from functools import partial
from scipy.stats import multinomial

import numpy as np
import itertools


class DiceGame:
    def __init__(self, dice=3, sides=6, *, values=None, bias=None, penalty=1):
        self._dice = dice
        self._sides = sides
        self._penalty = penalty
        if values is None:
            self._values = np.arange(1, self._sides + 1)
        else:
            if len(values) != sides:
                raise ValueError("Length of values must equal sides")
            self._values = np.array(values)

        if bias is None:
            self._bias = np.ones(self._sides)/self._sides
        else:
            self._bias = np.array(bias)

        if len(self._values) != len(self._bias):
            raise ValueError("Dice values and biases must be equal length")

        self._flip = {a: b for a, b in zip(self._values, self._values[::-1])}

        self.actions = []
        for i in range(0, self._dice + 1):
            self.actions.extend(itertools.combinations(range(0, self._dice), i))

        self.states = [a for a in itertools.combinations_with_replacement(self._values, self._dice)]

        self.final_scores = {state: self.final_score(state) for state in self.states}

        self.reset()

    def reset(self):
        self._game_over = False
        self.score = self._penalty
        self._current_dice = np.zeros(self._dice, dtype=np.int)
        _, dice, _ = self.roll()
        return dice

    def final_score(self, dice):
        uniques, counts = np.unique(dice, return_counts=True)
        uniques[counts > 1] = np.array([self._flip[x] for x in uniques[counts > 1]])
        return np.sum(uniques[counts == 1]) + np.sum(uniques[counts > 1] * counts[counts > 1])

    def flip_duplicates(self):
        uniques, counts = np.unique(self._current_dice, return_counts=True)
        if np.any(counts > 1):
            self._current_dice[np.isin(self._current_dice, uniques[counts > 1])] = \
                [self._flip[x] for x in self._current_dice[np.isin(self._current_dice, uniques[counts > 1])]]
        self._current_dice.sort()

    def roll(self, hold=()):
        if hold not in self.actions:
            raise ValueError("hold must be a valid tuple of dice indices")

        if self._game_over:
            return 0

        count = len(hold)
        if count == self._dice:
            self.flip_duplicates()
            self.score += np.sum(self._current_dice)
            return np.sum(self._current_dice), self.get_dice_state(), True
        else:
            mask = np.ones(self._dice, dtype=np.bool)
            hold = np.array(hold, dtype=np.int)
            mask[hold] = False
            self._current_dice[mask] = np.random.choice(self._values, self._dice - count,
                                                        p=self._bias, replace=True)
            self._current_dice.sort()

            self.score -= self._penalty
            return -1*self._penalty, self.get_dice_state(), False

    def get_dice_state(self):
        return tuple(self._current_dice)

    def get_next_states(self, action, dice_state):
        """
        Get all possible results of taking an action from a given state.

        :param action: the action taken
        :param dice_state: the current dice
        :return: state, game_over, reward, probabilities
                 state:
                    a list containing each possible resulting state as a tuple,
                    or a list containing None if it is game_over, to indicate
                    the terminal state
                 game_over:
                    a Boolean indicating if all dice were held
                 reward:
                    the reward for this action, equal to the final value of the
                    dice if game_over, otherwise equal to -1 * penalty
                 probabilities:
                    a list of size equal to state containing the probability of
                    each state occurring from this action
        """
        if action not in self.actions:
            raise ValueError("action must be a valid tuple of dice indices")
        if dice_state not in self.states:
            raise ValueError("state must be a valid tuple of dice values")

        count = len(action)
        if count == self._dice:
            return [None], True, self.final_score(dice_state), np.array([1])
        else:
            # first, build a mask (array of True/False) to indicate which values are held
            mask = np.zeros(self._dice, dtype=np.bool)
            hold = np.array(action, dtype=np.int)
            mask[hold] = True

            # get all possible combinations of values for the non-held dice
            other_vals = np.array(list(itertools.combinations_with_replacement(self._values,
                                                                               self._dice - count)),
                                  dtype=np.int)

            # in v1, dice only went from 1 to n
            # now dice can have any values, but values don't matter for probability, so get same data with 0 to n-1
            other_index = np.array(list(itertools.combinations_with_replacement(range(self._sides),
                                                                                self._dice - count)),
                                   dtype=np.int)

            # other_index will look like this, a numpy array of combinations
            #   [[0, 0], [0, 1], ..., [5, 5]]
            # need to calculate the probability of each one, so will query a multinomial distribution
            # if dice show (1, 3) then the correct query format is index based: [1, 0, 1, 0, 0, 0]
            queries = np.apply_along_axis(partial(np.bincount, minlength=self._sides), 1, other_index)
            probabilities = multinomial.pmf(queries, self._dice - count, self._bias)

            other_vals = np.insert(other_vals, np.zeros(count, dtype=np.int),
                                   np.asarray(dice_state, dtype=np.int)[mask], axis=1)

            other_vals.sort(axis=1)

            other_vals = [tuple(x) for x in other_vals]

            return other_vals, False, -1*self._penalty, probabilities


    # def getActionValue(self, action, cur_state, gamma):
    #     states, game_over, reward, probabilities = game.get_next_states(action, cur_state)  # Get the next states using the given function
    #     if game_over:
    #         current_action_value = reward
    #     else:
    #         current_action_value = sum([probability * (reward + (gamma * state_dict[next_state])) for next_state, probability in
    #                                     zip(states, probabilities)])  # Gets the current action's value
    #     return current_action_value
    #
    # def initialise_policy(self):
    #     gamma, converge_val = 1, 0.0000001
    #     state_dict = {state: 0 for state in game.states}
    #     next_dict = {state: 0 for state in game.states}  # Temporary dictionary to be able to keep stateDict until the end of this loop
    #     done = False
    #     while not done:
    #         for cur_state in state_dict:  # Loops through every state in the game
    #             next_dict[cur_state] = max([self.getActionValue(action, cur_state, gamma) for action in game.actions])
    #         done = max([abs(state_dict[state] - next_dict[state]) for state in game.states]) < converge_val
    #         state_dict = next_dict
    #
    #
    #     action_dict = {state: None for state in game.states}
    #     for cur_state in state_dict:  # Loops through every state in the game
    #         max_action_value = float("-inf")  # set the max_action_value to -infinity so any value will have a higher value than this
    #         best_action = None
    #         for action in game.actions:  # Loop through every action that can be done
    #             current_action_value = self.getActionValue(action, cur_state, gamma)
    #             if current_action_value > max_action_value:
    #                 best_action, max_action_value = action, current_action_value
    #         action_dict[cur_state] = best_action
    #     print(action_dict)
    #     return action_dict

    # def init_policy(self):
    #     gamma, theta = 1, 0.000001
    #     valueDict, policy = {state:0 for state in game.states}, {state:None for state in game.states}
    #     while True:
    #         delta = 0
    #         for cur_state in valueDict:
    #             max_action_val = float("-inf")
    #             best_action = None
    #             for action in game.actions:
    #                 states, game_over, reward, probabilities = game.get_next_states(action, cur_state)  # Get the next states using the given function
    #                 if game_over:
    #                     current_action_value = reward
    #                 else:
    #                     current_action_value = sum([probability * (reward + (gamma * state_dict[next_state])) for next_state, probability in zip(states, probabilities)])  # Gets the current action's value
    #                 if current_action_value > max_action_val:
    #                     best_action, max_action_val = action, current_action_value
    #             policy[cur_state] = best_action
    #             delta = max(delta, abs(valueDict[cur_state] - max_action_val))
    #         if delta < theta:
    #             break
    #     print(valueDict)
    #     print(policy)
    #     return policy
#
#
from abc import ABC, abstractmethod
import numpy as np


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    # if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    # if(verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        # if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        # if(verbose and not game_over): print(f"Dice: \t\t{state}")

    # if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score

# class MyAgent(DiceGameAgent):
#     def __init__(self, game):
#         """
#         if your code does any pre-processing on the game, you can do it here
#
#         e.g. you could do the value iteration algorithm here once, store the policy,
#         and then use it in the play method
#
#         you can always access the game with self.game
#         """
#         # this calls the superclass constructor (does self.game = game)
#         super().__init__(game)
#
#         # YOUR CODE HERE
#         self.policy = self.initialise_policy()
#
#     def initialise_policy(self):
#         """
#         Calculates the optimal policy using value iteration.
#         :return: Dictionary representing the optimal policy with keys representing the possible states and values representing the action that should be taken in the given state.
#         """
#         # Initialise the values of gamma (value between 0 and 1 representing discount rate) and theta (representing the value below which the policy is considered to have converged)
#         gamma, theta = 1, 0.000001
#         # Initialising the state_dict and current_iter_dict to initial value of 0 for each state, and the best action to None in the policy dictionary
#         state_dict, current_iter_dict, policy = {state: 0 for state in game.states}, {state: 0 for state in game.states}, {state: None for state in game.states}
#         while True:
#             # Sets delta (representing the current iteration's value which the policy has converged to)
#             delta = 0
#             # Loops through each state and gets the best action and its predicted value based on previous iterations
#             for cur_state in state_dict:
#                 # Sets the max_action_val, best_action to default values which will get overwritten in the first iteration
#                 max_action_val, best_action = float("-inf"), None
#                 # Loops through every action, and gets its action value, updating max_action_val and best_action if it has a higher predicted value.
#                 for action in game.actions:
#                     # Gets all possible results of taking the action 'action' from state 'cur_state'.
#                     states, game_over, reward, probabilities = game.get_next_states(action, cur_state)
#                     #Checks if the game is over, and if so then just sets the current_action_value to the reward
#                     if game_over:
#                         current_action_value = reward
#                     # Calculates the expected current_action_value by summing the probability of getting each next_state * (reward + (gamma (being the discount rate) * value of state given
#                     else:
#                         current_action_value = sum([probability * (reward + (gamma * state_dict[next_state])) for next_state, probability in
#                                                     zip(states, probabilities)])
#                     # Updates the best action and its value if the current_action_value is bigger than the maximum one recorded until now.
#                     if current_action_value > max_action_val:
#                         best_action, max_action_val = action, current_action_value
#                 # Updates delta throughout each state
#                 delta = max(delta, abs(state_dict[cur_state] - max_action_val))
#                 # Updates the policy and the current_iter_dict dictionaries with the values calculated above
#                 policy[cur_state], current_iter_dict[cur_state] = best_action, max_action_val
#             state_dict = current_iter_dict
#             # Checks whether the current iteration's delta value has converged below the theta value set at the beginning, if so then the process is stopped and the policy is returned
#             if delta < theta:
#                 return policy
#
#
#     def play(self, state):
#         """
#         given a state, return the chosen action for this state
#         at minimum you must support the basic rules: three six-sided fair dice
#
#         if you want to support more rules, use the values inside self.game, e.g.
#             the input state will be one of self.game.states
#             you must return one of self.game.actions
#
#         read the code in dicegame.py to learn more
#         """
#         return self.policy[state]

#
#
#
#
#
# class MyAgent(DiceGameAgent):
#     def __init__(self, game):
#         """
#         if your code does any pre-processing on the game, you can do it here
#
#         e.g. you could do the value iteration algorithm here once, store the policy,
#         and then use it in the play method
#
#         you can always access the game with self.game
#         """
#         # this calls the superclass constructor (does self.game = game)
#         super().__init__(game)
#
#         # Initialising the state_dict to initial value of 0 for each state, and the best action to None in the policy dictionary
#         self.state_dict, self.policy = {state: 0 for state in game.states}, {state: None for state in game.states}
#         # Initialise the values of gamma (value between 0 and 1 representing discount rate) and theta (representing the value below which the policy is considered to have converged)
#         self.gamma, self.theta = 1, 0.000001
#         self.initialise_policy()
#
#     def current_action_value(self, action, state):
#         """
#         Calculates the expected value the 'state' will yield by doing 'action'
#         :param action: tuple representing the action taken (i.e. which dice to hold)
#         :param state: tuple representing the number rolled on each die
#         :return: float representing the expected value by doing action on state
#         """
#         # Gets all possible results of taking the action 'action' from state 'state'.
#         states, game_over, reward, probabilities = game.get_next_states(action, state)
#         # Checks if the game is over, and if so then it returns the reward
#         if game_over:
#             return reward
#         # returns the expected current_action_value by summing the probability of getting each next_state * (reward + (gamma (being the discount rate) * value of state given
#         return sum([probability * (reward + (self.gamma * self.state_dict[next_state])) for next_state, probability in zip(states, probabilities)])
#
#     def best_action_and_val(self, state):
#         """
#         Calculates the optimal action and its value and returns them
#         :param state: tuple representing the number rolled on each die
#         :return: tuple representing the best action (i.e. which dice to hold) and a float representing its value
#         """
#         # Sets the max_action_val, best_action to default values which will get overwritten in the first iteration
#         best_action, max_action_val = "hi", float("-inf")
#         # Loops through every action, and gets its action value, updating max_action_val and best_action if it has a higher predicted value.
#         for action in game.actions:
#             current_action_value = self.current_action_value(action, state)
#             # Updates the best action and its value if the current_action_value is bigger than the maximum one recorded until now.
#             if current_action_value > max_action_val:
#                 best_action, max_action_val = action, current_action_value
#         return best_action, max_action_val
#
#     def initialise_policy(self):
#         """
#         Calculates the optimal policy using value iteration.
#         :return: Dictionary representing the optimal policy with keys representing the possible states
#         and values representing the action that should be taken in the given state.
#         """
#         # Initialising the current_iter_dict to initial value of 0 for each state.
#         # this dictionary exists as we need to take the values from the previous iteration and not the current iteration
#         current_iter_dict = {state: 0 for state in game.states}
#         while True:
#             # Sets delta (representing the current iteration's value which the policy has converged to)
#             delta = 0
#             # Iterates through each state and gets the best action and its predicted value based on previous iterations
#             for cur_state in self.state_dict:
#                 # Gets the best action and its predicted value based on previous iterations
#                 best_action, max_action_val = self.best_action_and_val(cur_state)
#                 # Updates delta throughout each state
#                 delta = max(delta, abs(self.state_dict[cur_state] - max_action_val))
#                 # Updates the policy and the current_iter_dict dictionaries with the values calculated above
#                 self.policy[cur_state], current_iter_dict[cur_state] = best_action, max_action_val
#             # Updates the states_dict at the end of the iteration
#             self.state_dict = current_iter_dict.copy()
#             # Checks whether the current iteration's delta value has converged below the theta value set at the beginning, if so then the process is stopped
#             if delta < self.theta:
#                 return
#
#     def play(self, state):
#         """
#         given a state, return the chosen action for this state
#         at minimum you must support the basic rules: three six-sided fair dice
#
#         if you want to support more rules, use the values inside self.game, e.g.
#             the input state will be one of self.game.states
#             you must return one of self.game.actions
#
#         read the code in dicegame.py to learn more
#         """
#         # YOUR CODE HERE
#         return self.policy[state]

#
# class MyAgent(DiceGameAgent):
#     def __init__(self, game):
#         """
#         if your code does any pre-processing on the game, you can do it here
#
#         e.g. you could do the value iteration algorithm here once, store the policy,
#         and then use it in the play method
#
#         you can always access the game with self.game
#         """
#         # this calls the superclass constructor (does self.game = game)
#         super().__init__(game)
#
#         # Initialising the state_dict to initial value of 0 for each state,
#         # and the best action to None in the policy dictionary
#         self.state_dict, self.policy = {state: 0 for state in game.states}, {state: None for state in game.states}
#         # Initialises a dictionary of dictionaries which stores the result returned
#         # from game.get_next_states given a state and action
#         self.next_state_dict = {state: {action: game.get_next_states(action, state)
#                                         for action in game.actions} for state in game.states}
#         # Initialise the values of gamma (value between 0 and 1 representing discount rate)
#         # and theta (representing the value below which the policy is considered to have converged)
#         self.gamma, self.theta = 1, 0.01
#         self.initialise_policy()
#         #print(self.policy == {(1, 1, 1): (0, 1, 2), (1, 1, 2): (0, 1), (1, 1, 3): (0, 1), (1, 1, 4): (0, 1, 2), (1, 1, 5): (0, 1, 2), (1, 1, 6): (0, 1, 2), (1, 2, 2): (1, 2), (1, 2, 3): (0,), (1, 2, 4): (0,), (1, 2, 5): (0,), (1, 2, 6): (0,), (1, 3, 3): (0,), (1, 3, 4): (0,), (1, 3, 5): (0,), (1, 3, 6): (0,), (1, 4, 4): (0,), (1, 4, 5): (0,), (1, 4, 6): (0,), (1, 5, 5): (0,), (1, 5, 6): (0,), (1, 6, 6): (0,), (2, 2, 2): (0, 1, 2), (2, 2, 3): (0, 1), (2, 2, 4): (0, 1, 2), (2, 2, 5): (0, 1, 2), (2, 2, 6): (0, 1, 2), (2, 3, 3): (0,), (2, 3, 4): (0,), (2, 3, 5): (0,), (2, 3, 6): (0,), (2, 4, 4): (0,), (2, 4, 5): (0,), (2, 4, 6): (0,), (2, 5, 5): (0,), (2, 5, 6): (0, 1, 2), (2, 6, 6): (0,), (3, 3, 3): (), (3, 3, 4): (), (3, 3, 5): (0, 1, 2), (3, 3, 6): (0, 1, 2), (3, 4, 4): (), (3, 4, 5): (), (3, 4, 6): (0, 1, 2), (3, 5, 5): (), (3, 5, 6): (0, 1, 2), (3, 6, 6): (), (4, 4, 4): (), (4, 4, 5): (), (4, 4, 6): (), (4, 5, 5): (), (4, 5, 6): (0, 1, 2), (4, 6, 6): (), (5, 5, 5): (), (5, 5, 6): (), (5, 6, 6): (), (6, 6, 6): ()})
#         #print(self.policy)
#
#     def current_action_value(self, action, state):
#         """
#         Calculates the expected value the 'state' will yield by doing 'action'
#         :param action: tuple representing the action taken (i.e. which dice to hold)
#         :param state: tuple representing the number rolled on each die
#         :return: float representing the expected value by doing action on state
#         """
#         # Gets all possible results of taking the action 'action' from state 'state'.
#         states, game_over, reward, probabilities = self.next_state_dict[state][action]
#         # Checks if the game is over, and if so then it returns the reward
#         if game_over:
#             return reward
#         # returns the expected current_action_value by summing the probability of getting each
#         # next_state * (reward + (gamma (being the discount rate) * value of state given))
#         expected_action_value = 0
#         for next_state, probability in zip(states, probabilities):
#             expected_action_value += probability * (reward + (self.gamma * self.state_dict[next_state]))
#         return expected_action_value
#
#     def best_action_and_val(self, state):
#         """
#         Calculates the optimal action and its value and returns them
#         :param state: tuple representing the number rolled on each die
#         :return: tuple representing the best action (i.e. which dice to hold) and a float representing its value
#         """
#         # Sets the max_action_val, best_action to default values which will get overwritten in the first iteration
#         best_action, max_action_val = None, float("-inf")
#         # Loops through every action, and gets its action value, updating max_action_val and best_action
#         # if it has a higher predicted value.
#         for action in game.actions:
#             current_action_value = self.current_action_value(action, state)
#             # Updates the best action and its value if the current_action_value is bigger than
#             # the maximum one recorded until now.
#             if current_action_value > max_action_val:
#                 best_action, max_action_val = action, current_action_value
#         return best_action, max_action_val
#
#     def initialise_policy(self):
#         """
#         Calculates the optimal policy using value iteration.
#         :return: Dictionary representing the optimal policy with keys representing the possible states
#         and values representing the action that should be taken in the given state.
#         """
#         # Initialising the current_iter_dict to initial value of 0 for each state.
#         # this dictionary exists as we need to take the values from the previous iteration and not the current iteration
#         current_iter_dict = {state: 0 for state in game.states}
#         while True:
#             # Sets delta (representing the current iteration's biggest change value) to 0
#             delta = 0
#             # Iterates through each state and gets the best action and its predicted value based on previous iterations
#             for cur_state in self.state_dict:
#                 # Gets the best action and its predicted value based on previous iterations
#                 best_action, max_action_val = self.best_action_and_val(cur_state)
#                 # Updates delta throughout each state
#                 delta = max(delta, abs(self.state_dict[cur_state] - max_action_val))
#                 # Updates the policy and the current_iter_dict dictionaries with the values calculated above
#                 self.policy[cur_state], current_iter_dict[cur_state] = best_action, max_action_val
#             # Updates the states_dict at the end of the iteration
#             self.state_dict = current_iter_dict.copy()
#             # Checks whether the current iteration's delta value has converged below the theta value set at the beginning,
#             # if so then the process is stopped
#             if delta <= self.theta:
#                 return
#
#     def play(self, state):
#         """
#         given a state, return the chosen action for this state
#         at minimum you must support the basic rules: three six-sided fair dice
#
#         if you want to support more rules, use the values inside self.game, e.g.
#             the input state will be one of self.game.states
#             you must return one of self.game.actions
#
#         read the code in dicegame.py to learn more
#         """
#         # YOUR CODE HERE
#         return self.policy[state]




class MyAgent(DiceGameAgent):
    def __init__(self, game, theta, gamma):
        """
        if your code does any pre-processing on the game, you can do it here

        e.g. you could do the value iteration algorithm here once, store the policy,
        and then use it in the play method

        you can always access the game with self.game
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # Initialising the state_dict to initial value of 0 for each state,
        # and the best action to None in the policy dictionary
        self.state_dict, self.policy = {state: 0 for state in game.states}, {state: None for state in game.states}
        # Initialises a dictionary of dictionaries which stores the result returned
        # from game.get_next_states given a state and action
        self.next_state_dict = {}
        for state in game.states:
            action_dict = {}
            for action in game.actions:
                states, game_over, reward, probabilities = game.get_next_states(action, state)
                action_dict[action] = list(zip(states,probabilities)), game_over, reward
            self.next_state_dict[state] = action_dict
        # Initialise the values of gamma (value between 0 and 1 representing discount rate)
        # and theta (representing the value below which the policy is considered to have converged)
        self.gamma, self.theta = gamma, theta
        self.initialise_policy()

    def bellman_equation(self, action, state):
        """
        Calculates the expected value the 'state' will yield by doing 'action'
        :param action: tuple representing the action taken (i.e. which dice to hold)
        :param state: tuple representing the number rolled on each die
        :return: float representing the expected value by doing action on state
        """
        # Gets the transition model, whether the game is over and the reward of taking the action 'action' from state 'state'.
        transition_model, game_over, reward = self.next_state_dict[state][action]
        # Checks if the game is over, and if so then it returns the reward
        if game_over:
            return reward
        # returns the expected current_action_value by summing the probability of getting each
        # next_state * (reward + (gamma (being the discount rate) * value of state given))
        expected_action_value = 0
        for next_state, probability in transition_model:
            expected_action_value += probability * (reward + (self.gamma * self.state_dict[next_state]))
        return expected_action_value

    def best_action_and_val(self, state):
        """
        Calculates the optimal action and its value and returns them
        :param state: tuple representing the number rolled on each die
        :return: tuple representing the best action (i.e. which dice to hold) and a float representing its value
        """
        # Sets the max_action_val, best_action to default values which will get overwritten in the first iteration
        best_action, max_action_val = None, float("-inf")
        # Loops through every action, and gets its action value, updating max_action_val and best_action
        # if it has a higher predicted value.
        for action in game.actions:
            current_action_value = self.bellman_equation(action, state)
            # Updates the best action and its value if the current_action_value is bigger than
            # the maximum one recorded until now.
            if current_action_value > max_action_val:
                best_action, max_action_val = action, current_action_value
        return best_action, max_action_val

    def initialise_policy(self):
        """
        Calculates the optimal policy using value iteration.
        :return: Dictionary representing the optimal policy with keys representing the possible states
        and values representing the action that should be taken in the given state.
        """
        # Initialising the current_iter_dict to initial value of 0 for each state.
        # this dictionary exists as we need to take the values from the previous iteration and not the current iteration
        current_iter_dict = {state: 0 for state in game.states}
        while True:
            # Sets delta (representing the current iteration's biggest change value) to 0
            delta = 0
            # Iterates through each state and gets the best action and its predicted value based on previous iterations
            for cur_state in self.state_dict:
                # Gets the best action and its predicted value based on previous iterations
                best_action, max_action_val = self.best_action_and_val(cur_state)
                # Updates delta throughout each state
                delta = max(delta, abs(self.state_dict[cur_state] - max_action_val))
                # Updates the policy and the current_iter_dict dictionaries with the values calculated above
                self.policy[cur_state], current_iter_dict[cur_state] = best_action, max_action_val
            # Updates the states_dict at the end of the iteration
            self.state_dict = current_iter_dict.copy()
            # Checks whether the current iteration's delta value has converged below the theta value set at the beginning,
            # if so then the process is stopped
            if delta < self.theta:
                return

    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice

        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions

        read the code in dicegame.py to learn more
        """
        # YOUR CODE HERE
        return self.policy[state]

def main():
    print("Let's play the game!")
    game = DiceGame()
    while True:
        dice = game.reset()
        print(f"Your dice are {dice}")
        print(f"Your score is {game.score}")
        while True:
            try:
                print("Type which dice you want to hold separated by spaces indexed from 0, blank to reroll all")
                print("Hold all dice to stick and get your final score")
                holds = input(">")
                if holds == "":
                    holds = tuple()
                else:
                    holds = tuple(map(int, holds.split(" ")))
                reward, dice, game_over = game.roll(holds)
                if game_over:
                    print(f"Your final dice are {dice}")
                    print(f"Your final score is {game.score}")
                    break
                else:
                    print(f"Your dice are {dice}")
                    print(f"Your score is {game.score}")
            except KeyboardInterrupt:
                return
            except:
                continue
        print("Play again? y/n")
        again = input(">")
        if again != "y":
            break


if __name__ == "__main__":
    main()
    # import time
    #
    # game = DiceGame()
    # n, vals, seeds, start_seed = 100000, 11, 5, 5
    # for k in range(vals):
    #     gamma = 0.9 + 0.01 * k
    #     time_total, score_total = 0, 0
    #     for j in range(start_seed, start_seed+seeds):
    #         total_score = 0
    #         total_time = 0
    #         np.random.seed(j)
    #         game.reset()
    #         start_time = time.process_time()
    #         test_agent = MyAgent(game, 0.001, gamma)
    #         total_time += time.process_time() - start_time
    #
    #         for i in range(n):
    #             start_time = time.process_time()
    #             score = play_game_with_agent(test_agent, game)
    #             total_time += time.process_time() - start_time
    #             total_score += score
    #
    #         time_total += total_time
    #         score_total += total_score
    #         print()
    #         print(f"Average score: {total_score / n}")
    #         print(f"Total time: {total_time:.4f} seconds")
    #     print(f"For gamma={gamma} the total score per game is {score_total / seeds / n} and average time per game is {time_total / seeds}")