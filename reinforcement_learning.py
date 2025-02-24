import numpy as np
import gymnasium as gym

class Agent:
    """An Agent class being able to interact with a gym environment"""
    def __init__(self, env, pol=None, Q=None):
        """The basic init function"""
        self.env = env
        if not isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            raise AssertionError("Action space is not discrete")
        if not isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
            raise AssertionError("Observation space is not discrete")
        if pol is None:
            self.pol = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        else:
            self.pol = pol
        if Q is None:
            self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        else:
            self.Q = Q
        self.memory = []
        self.replay_memory = []
        self.t = 0

    def reset_pol(self, pol=None):
        """Restore the policity to the default uniform one or fix it to a specified one"""
        if pol is None:
            self.pol = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        else:
            self.pol = pol

    def update_state_pol(self, state):
        """Perform a policy improvement step for a given state"""
        self.pol[state,] = 0
        self.pol[state, np.argmax(self.Q[state,])] = 1

    def update_pol(self):
        """Perform a policy improvement step for all states"""
        for state in range(self.env.observation_space.n):
           self.update_state_pol(state)

    def select_action(self, state, epsilon=0):
        """Select the next action as defined by a state and the current policy possibly modified with an epsilon smoothing"""
        return(np.random.choice(self.env.action_space.n, p=self.pol [state,] * (1 - epsilon) + epsilon / self.env.action_space.n))

    def play_select_action(self, state):
        """Select the action during play"""
        return(self.select_action(state))

    def learn_select_action(self, state):
        """Select the action during the learning"""
        return(self.select_action(state))

    def learn_init(self):
        """Initialize the learning algorithm"""
        pass

    def learn_update_after_action(self):
        """Update the parameters after an action"""
        pass

    def learn_update_after_episode(self):
        """Update the parameters after an episode"""
        pass

    def learn_final_update(self):
        """Update the parameters at the end of learning"""
        pass

    def save_memory(self, state, action, reward, replay):
        """Save the current state in a memory, optionnaly storing also the result of render()"""
        self.t = self.t + 1
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward
        })
        if replay:
            self.replay_memory.append({
                'frame': self.env.render(),
                'state': state,
                'action': action,
                'reward': reward
        })
        

    def replay(self, sleep_duration=.001):
        """Replay the last saved render history"""
        for t, frame in enumerate(self.replay_memory):
            display.clear_output(wait=True)
            plt.axis('off')
            plt.imshow(frame['frame'])
            display.display(plt.gcf())   
            print(f"Timestep: {t}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(sleep_duration)
        display.clear_output(wait=True)
        
    def play_or_learn(self, nb_episodes, nb_max_actions=None,replay=False, learn=False):
        """The core function describing the interaction with the environment"""
        G = 0
        nb_actions = 0
        self.t = -1
        nbactions = 0
        self.replay_record = []
        self.memory = []
        if learn:
            self.learn_init()
        for i in range(nb_episodes):
            if  not (nb_max_actions is None):
                if  nb_actions > nb_max_actions:
                    done = True
            display.clear_output()
            print(i + 1," / ", nb_episodes)
            state, info = self.env.reset()
            done = False
            reward = 0
            if learn:
                action = self.learn_select_action(state)
            else:
                action = self.play_select_action(state)
            nb_actions += 1
            done = False
            while not done:
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                G += reward
                if done:
                    action = "Done"
                else:
                    if learn:
                        action = self.learn_select_action(state)
                    else:
                        action = self.play_select_action(state)
                    nb_actions += 1
                self.save_memory(state, action, reward, replay)
                if learn:
                    self.learn_update_after_action()
                if  not (nb_max_actions is None):
                    if (nb_actions > nb_max_actions):
                        done = True
            if learn:
                self.learn_update_after_episode()
        if learn:
            self.learn_final_update()    
        return(G, nb_actions)

    def play(self, nb_episode, nb_max_actions=None, replay=False):
        """An alias function to play"""
        return(self.play_or_learn(nb_episode, nb_max_actions, replay, learn=False))
    
    def learn(self, nb_episode, nb_max_actions=None, replay=False):
        """An alias function to learn"""
        return(self.play_or_learn(nb_episode, nb_max_actions, replay, learn=True))
       
       
class SarsaAgent(Agent):
    def __init__(self, env, pol=None, Q=None, gamma=.9, stepsize=.1, epsilon_init=1, epsilon_min=.01, epsilon_decay=.99):
        super().__init__(env, pol, Q)
        self.gamma = gamma
        self.stepsize = stepsize
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def learn_select_action(self, state):
        return self.select_action(state, self.epsilon)

    def learn_init(self):
        self.epsilon = self.epsilon_init

    def learn_update_after_action(self):
        state = self.memory[self.t - 1]['state']
        action = self.memory[self.t - 1]['action']
        new_state = self.memory[self.t]['state']
        new_reward = self.memory[self.t]['reward']
        new_action = self.memory[self.t]['action']
        
        if new_action == 'Done':
            new_action = 0  # Ensure this is a valid action index if needed

        if action != 'Done':
            self.Q[state, action] += self.stepsize * (
                new_reward + self.gamma * self.Q[new_state, new_action] - self.Q[state, action]
            )
        self.update_state_pol(state)

    def learn_update_after_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn_final_update(self):
        self.update_pol() 