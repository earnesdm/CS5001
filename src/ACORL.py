import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# simple convolutional neural network used for both the q-function and the imitation policy
class ConvNet(nn.Module):
    def __init__(self, frames, num_actions):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # fully connected head for q-net
        self.q_fc1 = nn.Linear(3136, 512)
        self.q_fc2 = nn.Linear(512, num_actions)

        # fully connected head for imitation learning policy
        self.i_fc1 = nn.Linear(3136, 512)
        self.i_fc2 = nn.Linear(512, num_actions)

    def forward(self, state):
        # pass through convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # pass through fully connected q-net head
        q = F.relu(self.q_fc1(x.reshape(-1, 3136)))
        q = self.q_fc2(q)

        # pass through fully connected policy head
        i = F.relu(self.q_fc1(x.reshape(-1, 3136)))
        i = self.q_fc2(i)

        return q, i


class ACORL:
    def __init__(self, state_dim, action_dim, threshold=0.3, gamma=0.99, target_update=8e3):
        # define the two-headed network for calculating q-values and cloning the behavioral policy
        self.net = ConvNet(state_dim[0], action_dim)

        # define the target network for double DQN
        self.target_net = copy.deepcopy(self.net)

        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0000625, eps=0.00015)

        # initialize the probability threshold for considering actions
        self.threshold = threshold

        # initialize the discount
        self.gamma = gamma

        # store training sets
        self.n = 0

        # store the frequency of target network updates
        self.target_update = target_update

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)

            # compute q-values and the cloned logits
            q, i = self.net(state)

            # convert logits to probabilities
            probs_i = torch.softmax(i, dim=1)

            # mask to set actions with probability below the threshold to -1e8
            mask = torch.ones_like(probs_i) + (probs_i / torch.max(probs_i, dim=1, keepdim=True)[0] <= self.threshold) * (-1e8)

            # return the actions
            return int(torch.argmax(q * mask, dim=1))

    def train_step(self, replay_buffer, both=True):
        # sample (s, a, r, s') from the replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # compute the target step
        # TODO: add double DQN for overfitting
        with torch.no_grad():
            # compute q-values and the cloned logits
            q, i = self.target_net(next_state)

            # convert logits to probabilities
            probs_i = torch.softmax(i, dim=1)

            # mask to set actions with probability below the threshold to -1e8
            mask = torch.ones_like(probs_i) + (probs_i/torch.max(probs_i, dim=1, keepdim=True)[0] <= self.threshold) * (-1e8)

            # compute the target values
            target = reward + done * self.gamma * torch.gather(q, 1, torch.argmax(q * mask, dim=1).unsqueeze(1))

        # calculate current q-value
        q, i = self.net(state)
        q = torch.gather(q, 1, action)

        # calculate the loss
        bellman_loss = F.mse_loss(q, target)
        imitation_loss = F.nll_loss(torch.softmax(i, dim=1), action.squeeze(1)) + 1e-3 * torch.mean(torch.square(i))

        if both:
            total_loss = bellman_loss + imitation_loss
        else:
            total_loss = imitation_loss

        # perform gradient descent
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # update the number of training steps
        self.n += 1

        # update the target network
        if self.n % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())




