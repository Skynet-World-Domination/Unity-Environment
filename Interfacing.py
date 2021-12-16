import torch
import random
import os
import gc
import numpy as np
from typing import Dict, NamedTuple, List
from mlagents_envs.environment import UnityEnvironment, ActionTuple, BaseEnv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
modules = list(model.children())[:-1]
resnet50 = torch.nn.Sequential(*modules)
resnet50.to(device)
#torch.cuda.empty_cache()

def compute_resnet(image):
    feat = torch.from_numpy(np.moveaxis(image, -1, 1))
    with torch.no_grad():
        feat = resnet50(feat.to(device))
    feat = np.squeeze(feat, axis=2)
    feat = np.squeeze(feat, axis=2)
    return feat.cpu()


class BeautifulNetwork(torch.nn.Module):
    def __init__(self):
        """
        Creates a neural network
        """
        super(BeautifulNetwork, self).__init__()

        self.siamese = torch.nn.Linear(2048, 512)
        self.fusion = torch.nn.Linear(1024, 512)
        self.lastFC = torch.nn.Linear(512, 512)
        self.policy = torch.nn.Linear(512, 3)
        self.value = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, visual_obs: torch.tensor, target: torch.tensor):
        # Siamese layers
        outObs = self.relu(self.siamese(visual_obs))
        outTar = self.relu(self.siamese(target))
        # Fusion output of siamese
        fusedOutputs = torch.cat((outObs, outTar), 1)
        # Fusion layer
        outEmbbeding = self.relu(self.fusion(fusedOutputs))
        # Fully connected of scene specific
        outLastFC = self.relu(self.lastFC(outEmbbeding))
        # Policy layer
        policy = (self.policy(outLastFC))
        # Value layer
        value = self.value(outLastFC)

        return policy


class Experience(NamedTuple):
    """
  An experience contains the data of one Agent transition.
  - Observation
  - Action
  - Reward
  - Done flag
  - Next Observation
  """

    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray


# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]


class Trainer:
    @staticmethod
    def generate_trajectories(
            env: BaseEnv, b_net: BeautifulNetwork, buffer_size: int, epsilon: float
    ):
        """
    Given a Unity Environment and a Q-Network, this method will generate a
    buffer of Experiences obtained by running the Environment with the Policy
    derived from the Q-Network.
    :param BaseEnv: The UnityEnvironment used.
    :param q_net: The Q-Network used to collect the data.
    :param buffer_size: The minimum size of the buffer this method will return.
    :param epsilon: Will add a random normal variable with standard deviation.
    epsilon to the value heads of the Q-Network to encourage exploration.
    :returns: a Tuple containing the created buffer and the average cumulative
    the Agents obtained.
    """
        # Create an empty Buffer
        buffer: Buffer = []
        # Reset the environment
        env.reset()
        # Read and store the Behavior Name of the Environment
        behavior_name = list(env.behavior_specs)[0]
        ##    print(behavior_name)
        # Read and store the Behavior Specs of the Environment
        spec = env.behavior_specs[behavior_name]

        # Create a Mapping from AgentId to Trajectories. This will help us create
        # trajectories for each Agents
        dict_trajectories_from_agent: Dict[int, Trajectory] = {}
        # Create a Mapping from AgentId to the last observation of the Agent
        dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to the last action of the Agent
        dict_last_action_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to cumulative reward (Only for reporting)
        dict_cumulative_reward_from_agent: Dict[int, float] = {}
        # Create a list to store the cumulative rewards obtained so far
        cumulative_rewards: List[float] = []

        while len(buffer) < buffer_size:  # While not enough data in the buffer
            # Get the Decision Steps and Terminal Steps of the Agents
            ##      print("Buffer size : " + str(len(buffer)))
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # For all Agents with a Terminal Step:
            for agent_id_terminated in terminal_steps:
                #print("Agent " + str(agent_id_terminated) + " in terminal step")
                # Create its last experience (is last because the Agent terminated)
                last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(),
                    next_obs=[compute_resnet(np.expand_dims(terminal_steps[agent_id_decisions].obs[0], axis=0)),
                              compute_resnet(np.expand_dims(terminal_steps[agent_id_decisions].obs[1], axis=0))],
                )

                # Clear its last observation and action (Since the trajectory is over)
                dict_last_obs_from_agent.pop(agent_id_terminated)
                dict_last_action_from_agent.pop(agent_id_terminated)
                # Report the cumulative reward
                cumulative_reward = (
                        dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                        + terminal_steps[agent_id_terminated].reward
                )
                cumulative_rewards.append(cumulative_reward)
                # Add the Trajectory and the last experience to the buffer
                buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
                buffer.append(last_experience)
                #print("Cummulativ ,")
                #print(cumulative_reward)
                print("Size of the buffer now : " + str(len(buffer)))

            # For all Agents with a Decision Step:
            for agent_id_decisions in decision_steps:
                # If the Agent does not have a Trajectory, create an empty one
                if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = 0

                # If the Agent requesting a decision has a "last observation"
                if agent_id_decisions in dict_last_obs_from_agent:
                    # Create an Experience from the last observation and the Decision Step
                    exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=[compute_resnet(np.expand_dims(decision_steps[agent_id_decisions].obs[0], axis=0)),
                                  compute_resnet(np.expand_dims(decision_steps[agent_id_decisions].obs[1], axis=0))],
                    )
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                    #print(decision_steps[agent_id_decisions].reward)
                # Store the observation as the new "last observation" as resnet features
                dict_last_obs_from_agent[agent_id_decisions] = (
                    [compute_resnet(np.expand_dims(decision_steps[agent_id_decisions].obs[0], axis=0)),
                     compute_resnet(np.expand_dims(decision_steps[agent_id_decisions].obs[1], axis=0))]
                )

            # Generate an action for all the Agents that requested a decision
            # Compute resnet features for observations
            visual_obs = compute_resnet(decision_steps.obs[0])
            target = compute_resnet(decision_steps.obs[1])

            # Compute the values for each action given the observation
            actions_values = (
                b_net(visual_obs.to(device), target.to(device)).cpu().detach().numpy()
            )
##            print("--------------------")
##            print(actions_values)
            # Add some noise with epsilon to the values
            actions_values += epsilon * (
                np.random.randn(actions_values.shape[0], actions_values.shape[1])
            ).astype(np.float32)
##            print(actions_values)

            # Pick the best action using argmax
            actions = np.argmax(actions_values, axis=1)
            actions = np.asarray([actions])
            actions.resize((len(decision_steps), 1))
            #print(actions)

            # Store the action that was picked, it will be put in the trajectory later
            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index]
                
            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            env.set_actions(behavior_name, action_tuple)
            # Perform a step in the simulation
            env.step()
        #print("mean")
        #print(np.mean(cumulative_rewards))
        return buffer, np.mean(cumulative_rewards)

    @staticmethod
    def update_b_net(
            b_net: BeautifulNetwork,
            optimizer: torch.optim,
            buffer: Buffer,
            action_size: int,
            num_tsteps: int
    ):
        """
    Performs an update of the Q-Network using the provided optimizer and buffer
    """

        lossvalues = 0
        lossmeanepoch = 0
        
        BATCH_SIZE = 5
        NUM_EPOCH = 3
        GAMMA = 0.9
        batch_size = min(len(buffer), BATCH_SIZE)
        random.shuffle(buffer)
        # Split the buffer into batches
        batches = [
            buffer[batch_size * start: batch_size * (start + 1)]
            for start in range(int(len(buffer) / batch_size))
        ]
        for epoch_num in range(NUM_EPOCH):
            print("EPOCH N° : " + str(epoch_num))
            countbatch = 0
            lossvalues = 0
            for batch in batches:
                #print("Batch n° : " + str(countbatch))
                countbatch += 1
                # Create the Tensors that will be fed in the network
                obs_curr = torch.from_numpy(np.squeeze(np.stack([ex.obs[0] for ex in batch])))
                obs_tar = torch.from_numpy(np.squeeze(np.stack([ex.obs[1] for ex in batch])))
  
                reward = torch.from_numpy(
                    np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
                )
                done = torch.from_numpy(
                    np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1)
                )
                action = torch.from_numpy(np.stack([ex.action for ex in batch]))

                next_obs_curr = torch.from_numpy(np.squeeze(np.stack([ex.next_obs[0] for ex in batch])))
                next_obs_tar = torch.from_numpy(np.squeeze(np.stack([ex.next_obs[1] for ex in batch])))
     
                # Use the Bellman equation to update the Network
                target = (
                        reward
                        + (1.0 - done)
                        * GAMMA
                        * torch.max(b_net(next_obs_curr.to(device), next_obs_tar.to(device)).cpu().detach(), dim=1,
                                    keepdim=True).values
                )
                mask = torch.zeros((len(batch), action_size))
                mask.scatter_(1, action, 1)
                prediction = torch.sum(bnet(obs_curr.to(device), obs_tar.to(device)).cpu() * mask, dim=1, keepdim=True)
                criterion = torch.nn.MSELoss()
                loss = criterion(prediction, target)

                # Perform the backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
                lossvalues += loss.item()

            lossmeanepoch = lossvalues/countbatch
            print("Loss : " + str(lossmeanepoch))
            
        writer.add_scalar('Loss/training_steps', lossmeanepoch, num_tsteps)
        #print(lossmeanepoch)


##        print("Back prop of a batch done")


'''
#Check model architecture
BModel = BeautifulNetwork()
summary(BModel, input_size=[(1,8192),(1,8192)])
'''
try:
    env.close()
except:
    pass

print("Press play in unity to load the environment...")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
print("Environment created.")

bnet = BeautifulNetwork()
bnet.to(device)

experiences: Buffer = []
optim = torch.optim.RMSprop(bnet.parameters(), lr=0.0007) #0.0001

cumulative_rewards: List[float] = []

# The number of training steps that will be performed
NUM_TRAINING_STEPS = int(os.getenv('QLEARNING_NUM_TRAINING_STEPS', 10)) #50
# # The number of experiences to collect per training step
NUM_NEW_EXP = int(os.getenv('QLEARNING_NUM_NEW_EXP', 200)) #2000
# The maximum size of the Buffer
BUFFER_SIZE = int(os.getenv('QLEARNING_BUFFER_SIZE', 200)) #2000

for n in range(NUM_TRAINING_STEPS):
    print("Collecting data for updating...")
    new_exp, _ = Trainer.generate_trajectories(env, bnet, NUM_NEW_EXP, epsilon=0.1)
    random.shuffle(experiences)
    experiences.extend(new_exp)
    if len(experiences) > BUFFER_SIZE:
        experiences = experiences[len(experiences)-BUFFER_SIZE:]
    Trainer.update_b_net(bnet, optim, experiences, 3, n+1)
    print("Collecting data for monitoring reward...")
    _, rewards = Trainer.generate_trajectories(env, bnet, 100, epsilon=0) #1000
    cumulative_rewards.append(rewards)
    print("Training step ", n + 1, "\treward ", rewards)
    writer.add_scalar('Reward/training_steps', rewards, n+1)
    writer.flush()

env.close()

# Show the training graph
plt.plot(range(NUM_TRAINING_STEPS), cumulative_rewards)

