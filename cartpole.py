"""
open terminal
activate container
    $ source activate tensorflow
navigate to file
    $ cd Desktop
run file
    $  python cartpole.py

"""




import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


LR = 1e-3
env = gym.make("CartPole-v0")
# env = gym.make("AirRaid-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000




def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()






def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score+=reward

            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)


    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data




def neural_network_model(input_size, layers_node_list):
    model = 0
    network = input_data(shape=[None, input_size, 1], name='input')

    # network = fully_connected(network, 64, activation='relu')
    # network = dropout(network, 0.8)
    drop_percent = 0.6

    for layer in layers_node_list:

        this_drop_percent = min(1.0, drop_percent)
        drop_percent += 0.1

        network = fully_connected(network, layer, activation='relu')
        network = dropout(network, this_drop_percent)


    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, layers_node_list):
    model=False

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), layers_node_list = layers_node_list)

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=1000, show_metric=True, run_id='openai_learning')
    return model




def test_model(model):

    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            # env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    model_score = sum(scores)/len(scores)

    print('Average Score:', model_score)
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)

    return model_score




def build_model():

    training_data = initial_population()
    # model.save("asdfasaa.model")
    # model.load("asdfasaa.model")

    random_nodes_one = (2**random.randrange(10,12))
    layers_node_list = [128,256,512,1024,random_nodes_one,128]
    model = train_model(training_data, layers_node_list)
    indivudual_health = test_model(model)
    print(indivudual_health)

build_model()
