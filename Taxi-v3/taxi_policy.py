
'''

        Self Driving Cab


'''



import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make('Taxi-v3')
random_policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA



'''

POLICY ITERATION

'''


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.env.nS)
    while True:
        # TODO: Implement!
        delta = 0  # delta = change in value of state from one iteration to next

        for state in range(env.env.nS):  # for all states
            val = 0  # initiate value as 0

            for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                for prob, next_state, reward, done in env.env.P[state][
                    action]:  # transition probabilities,state,rewards of each action
                    val += act_prob * prob * (reward + discount_factor * V[next_state])  # eqn to calculate
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val
        if delta < theta:  # break if the change in value is less than the threshold (theta)
            break
    return np.array(V)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA

    while True:
        # Implement this!
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  # eval current policy
        policy_stable = True  # Check if policy did improve (Set it as True first)
        for state in range(env.env.nS):  # for each states
            chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
            act_values = one_step_lookahead(state, curr_pol_val)  # use one step lookahead to find action values
            best_act = np.argmax(act_values)  # find best action
            if chosen_act != best_act:
                policy_stable = False  # Greedily find best action
            policy[state] = np.eye(env.env.nA)[best_act]  # update
        if policy_stable:
            return policy, curr_pol_val

    return policy, np.zeros(env.env.nS)


pol_iter_policy = policy_iteration(env,policy_eval,discount_factor=0.99)
pol_iter_policy[0]



def count(policy):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        counter += 1
    return counter



pol_count = count(pol_iter_policy[0])
pol_counts = [count(pol_iter_policy[0]) for i in range(10000)]
print("An agent using a policy which has been improved using policy-iterated takes about an average of " + str(int(np.mean(pol_counts)))
      + " steps to successfully complete its mission.")
sns.distplot(pol_counts)


def view_policy(policy):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[0][curr_state]))
        curr_state = state
        counter += 1
        env.env.s = curr_state
        env.render()


view_policy(pol_iter_policy)