# DICE GAME README FILE

### DICE GAME RULES
The game consists of 3 6-sided fair dice.  
The aim is to get the highest score by the end of the game.  
The game starts by rolling the 3 dice, and the player can choose which dice they want to hold (if any) and which dice
should be re-rolled (if any).  
If they re-roll at least 1 die, then a penalty is subtracted from their score (which by default is set to 1),
and the game continues.  
If they hold all the dice, then firstly, any number which appears more than once is flipped (i.e. 1 to 6, 2 to 5, 3 to 4)
and then the sum of the dice (being the final score - after the flip occurs) is added to the total score, which is then determined as the total score.


### POSSIBLE MODIFICATIONS
There are multiple ways to modify the game:
1. Change the number of dice: To do this, type in the instantiation of DiceGame, `DiceGame(dice=[number of dice])`  
    where `[number of dice]` is an integer representing the number of dice that should be used in the game.  
    The default for this argument is 3 dice.
2. Change the number of sides per dice: To do this, type in the instantiation of DiceGame, `DiceGame(sides=[number of sides])`  
    where `[number of sides]` is an integer representing the number of sides that each die should have in the game.  
    The default for this argument is 6 sides.
3. Change the values of each dice: To do this, type in the instantiation of DiceGame, `DiceGame(values=[list of values])`  
    where `[list of values]` is an increasing list of non-negative integers representing the values of dice that should be used in the game.  
    The default for this argument is a list of values increasing from 1 to the number of sides + 1 in increments of 1.  
    NOTE: The list's length must be equal to the number of sides  
    NOTE 2: The values should be a list of non-negative integers in increasing order
4. Change the bias of each dice: To do this, type in the instantiation of DiceGame, `DiceGame(bias=[list of bias values])`  
    where `[list of bias values]` is list representing the biases of each side on the dice that should be used in the game.  
    The default for this argument is a list of 1/number of sides (i.e. a fair die).  
    NOTE: The list's length must be equal to the number of sides  
    NOTE 2: Each element in the list must be a number between 0 and 1 (inclusive) such that the sum of the elements is 1.
5. Change the penalty given for a re-roll: To do this, type in the instantiation of DiceGame, `DiceGame(penalty=[penalty])`  
    where `[penalty]` is a float representing the number that should be subtracted from the total score for every re-roll in the game.  
    The default for this argument is 1.  
    NOTE: If the penalty is set to be positive, the game will never end with an optimal player, as they have no intention to hold  
    (they can get infinite points if they keep re-rolling forever). Thus, the environment should be set with a non-negative penalty.

These can also be combined. For example:  
`game = DiceGame(dice=2, sides=5, values=[1, 2, 6, 7, 10], bias=[0.1, 0.2, 0.1, 0.3, 0.3], penalty=3)` will create a game with the following:
1. You roll 4 dice roll 4 dice
2. Each die has 5 sides
3. Each die's sides are labelled 1, 2, 6, 7 and 10
4. Probability of rolling a 1 or 6 is 0.1, probability of rolling 2 is 0.2, and the probability of rolling 7 or 10 is 0.3
5. Each re-roll costs 3 points


### OVERVIEW OF THE PROGRAM
The aim of the program is to write an agent that produces an optimal policy (discussed later) efficiently (i.e. reduce the computation time as much as possible).  
To do this, I will first explain how to model the game as a Markov Decision Process.


### MARKOV DECISION PROCESS
A Markov Decision Process is a mathematical model representing an environment which follows the Markov Property.  
In other words, the next state only depends on your current state and your current action.  
Note that this is the case in our dice game as the probabilities of future states do not depend on our score.  
A Markov Decision Process contains:
1. A finite set of states - In our game, this is game.states.  
    game.states is a finite set of states, where each state is a tuple of values representing a roll of the dice
2. A finite set of actions - In out game, this is game.actions.  
    game.actions is a finite set of actions, where each action is a tuple of values representing which dice to hold.
3. Transition probability - In our game, this is unpacked in the bellman_equation function.  
    This is a list containing tuples, where each tuple is (new_state,probability) given the current_state and action.
4. Reward - In our game, this is unpacked in the bellman_equation function.  
    This is minus the penalty if the game is not over, and the sum of the dice (after flipping non-unique dice) if the game is over.


### TERMINOLOGY
1. Policy PI is a map between the states and actions.  
    In other words, given a state, the policy determines which action should be taken.  
    A solution to a Markov Decision Process is in the form of a policy.
2. (Q^PI)(s,a) is the action-value function, which is a map between (state,action) pairs and values (numbers).  
    In other words, the action-value function returns the expected value of using action 'a' in state 's' under policy PI.
3. (V^PI)(s) is the state-value function, which is a map between the states and values (numbers).  
    It quantify what is the expected return of being in a state (following policy PI).  
    In other words, the state-value function returns the expected value of being in state 's' under policy PI.
4. Bellman's equation is a recursive equation representing the relationship between the value of a state and the value of its successor state.  
    Bellman's equation computes the value of a state by considering all the successor states and their likelihoods.  
    Bellman's equation uses a value gamma, which will later be determined.
5. Optimal policy is a policy that its state-value function has the highest value for each state


### CHOICE OF ALGORITHM
As stated above, the main priority is making the agent produce an optimal policy.  
Value iteration ensures this, and thus is a suitable choice of algorithm (using suitable choices of gamma and theta).  
It is worth noting that there are other equally good algorithms for the purpose of this game (such as policy iteration).

The main idea in value iteration is to use successive approximations in order to improve the state-value function.  
More specifically, we start by initialising the state-value function to arbitrary values.  
We then update the state-value function for each state using the action-value function, keeping track of the maximum difference in the given iteration as delta.  
We do this by taking the maximum value of all actions that can be taken from the given state.  
To calculate the action's value given a state, we use Bellman's equation.  
We do the above until the values converge, which is once delta is smaller than theta (which is a value to be determined).


### LAW OF LARGE NUMBERS
The Law of Large Numbers states that the more trials you perform of the same experiment, the more likely it is that the average of the results will be close to the expected value.  
Applying this, when testing our agent, we will need to do many trials to get a good approximation of its expected value when testing certain parameters.


### CHOICE OF GAMMA
As stated before, the main aim of the program is to write an agent that can play this simple dice game in an optimal way.  
In other words, we need to achieve the optimal score (on average).  
The score we get by the end of the game is a sum of all the scores generated along the way. More mathematically speaking:

<img src="gammaCalculation.png" width="500">

As can be seen, theoretically speaking, setting gamma to 1 should yield the highest score.  
Let's put that to the test:

All of the following tests will be done on the default settings (3 6-sided fair dice with faces 1-6 and penalty of 1)  
The result displayed will be for 100,000 games in each seed over 10 seeds (np.random.seed 0 to 9 if you wish to replicate results)  
in order to get more reliable results (Law of Large Numbers).  
We will use theta = 0.001 as a control variable:

| Gamma Value | Average Score | Average Time for 100,000 games (seconds) |
|:------------|:--------------|:-----------------------------------------|
| 0           | 10.496774     | 11.6813                                  |
| 0.1         | 10.496774     | 11.7500                                  |
| 0.2         | 10.496774     | 11.4797                                  |
| 0.3         | 10.496774     | 12.2781                                  |
| 0.4         | 10.628091     | 12.3578                                  |
| 0.5         | 10.730125     | 11.9109                                  |
| 0.6         | 10.891708     | 12.8297                                  |
| 0.7         | 11.253107     | 13.1203                                  |
| 0.8         | 11.718260     | 13.6078                                  |
| 0.9         | 12.349887     | 16.5047                                  |
| 1.0         | 13.351021     | 27.0953                                  |

Notice that the average score increases as gamma is set closer to 1.

| Gamma Value | Average Score | Average Time for 100,000 games (seconds) |
|:------------|:--------------|:-----------------------------------------|
| 0.90        | 12.349887     | 15.5359                                  |
| 0.91        | 12.643302     | 16.5063                                  |
| 0.92        | 12.732459     | 17.0516                                  |
| 0.93        | 12.820099     | 17.8938                                  |
| 0.94        | 12.826585     | 17.8109                                  |
| 0.95        | 12.974979     | 18.7813                                  |
| 0.96        | 12.974979     | 18.7531                                  |
| 0.97        | 13.141527     | 21.1516                                  |
| 0.98        | 13.308201     | 23.4156                                  |
| 0.99        | 12.310509     | 24.3969                                  |
| 1.00        | 13.351021     | 25.7078                                  |

Thus, we can determine that in order to get optimal score, we should set gamma to 1.  
Note that the reason the games are shorter for lower gammas, is because the policy opts to hold dice,  
and hence each game on average takes less turns to finish.


### CHOICE OF THETA
As stated before, another aim of the program is to write an agent that can calculate the optimal policy in a short amount of time (as speed is a factor in this coursework).  
This is where the choice of theta can come in handy. We do not want to make theta too big, as then the value iteration might break before it has converged, thus yielding a different
policy (which will not play optimally), but we also do not want to make theta too small, as then the value iteration will keep going even after it has basically converged.  
In other words, we need to set theta such to be the biggest value possible that still keeps the same policy.

It is worth noting that theoretically speaking, theta may be dependent on the penalty and the values.  
Consider the following argument:  
Why is theta = 10 not a good choice of theta?  
It is because that the difference between the values of the faces of the dice and the penalty is below 10.  
Hence, if we set theta to 10, we are breaking out of the while loop before the values converged.  
We can scale this example down (for example `values = [0.1,...,0.6], penalty = 0.1`) to show that any fixed theta is not valid for certain values and penalty.  
However, as the list of values must be a list of integers, then there is a certain small enough theta which allows the values to converge for no matter which values/penalty.  
We will try to find this theta in the following tests.

To check what the optimal policy is, I will first set theta to 0 (and change the line to ```delta <= theta``` just so it will break when theta is 0).  
I will then increase theta and record where the policy changes.  
All of the following tests will be done on the default settings (3 6-sided fair dice with faces 1-6 and penalty of 1).  
The result displayed will be for 100,000 games in each seed over 10 seeds (np.random.seed 0 to 9 if you wish to replicate results) in order
to get more reliable results (Law of Large Numbers).  
We will use gamma = 1 as a control variable:

| Theta value | Average score | Time taken for 100,000 games(seconds) | Same Policy as Theta=0 |
|:------------|:--------------|:--------------------------------------|:-----------------------|
| 0.0000001   | 13.351021     | 25.2141                               | True                   |
| 0.000001    | 13.351021     | 25.0172                               | True                   |
| 0.00001     | 13.351021     | 24.9031                               | True                   |
| 0.0001      | 13.351021     | 24.8359                               | True                   |
| 0.001       | 13.351021     | 24.7609                               | True                   |
| 0.01        | 13.350721     | 24.5297                               | False                  |
| 0.1         | 13.350721     | 24.4953                               | False                  |
| 1           | 13.159282     | 20.1344                               | False                  |

As can be seen, the place where the policy changes the first time is between theta = 0.001 and theta = 0.01.  
Thus, we can determine that in order to reduce the time taken while maintaining optimal score, we should set theta to 0.001.


### OPTIMISING THE CODE
One of the requirements for the program is for the agent to be efficient (i.e. the speed of the code).  
Here, I will discuss how I made my code more efficient.

The biggest change between the implementations is storing what game.get_next_states returns.  
Originally, I called game.get_next_states multiple times for each state,action pair.  
This is obviously inefficient, especially considering that function is costly calculation-wise.  
To avoid this, I made a dictionary of dictionaries.  
The inner dictionaries contain actions as keys and the return value of game.get_next_states as its values for a given state.  
The outer dictionary contain states as its keys and the dictionaries described above as its values.  
This means that we now only call game.get_next_states once per state,action pair.

Another thing which makes my code efficient, is that I store the policy as a dictionary rather than a list.  
A dictionary's average case to look up a value is O(1) whereas a list's average case to look up a value is O(n)  
Thus, this minor change makes a big difference when playing many games using the same environment.

