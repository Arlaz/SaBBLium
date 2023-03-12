# coding=utf-8
#
# Copyright © Facebook, Inc. and its affiliates.
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def _index(tensor_3d, tensor_2d):
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


def cumulated_reward(reward, done):
    T, B = done.size()
    done = done.detach().clone()

    v_done, index_done = done.float().max(0)
    assert v_done.eq(
        1.0
    ).all(), "[agents.rl.functional.cumulated_reward] Computing cumulated reward over unfinished trajectories."
    arange = torch.arange(T, device=done.device).unsqueeze(-1).repeat(1, B)
    index_done = index_done.unsqueeze(0).repeat(T, 1)

    mask = arange.le(index_done)
    reward = (reward * mask.float()).sum(0)
    return reward.mean().item()


def temporal_difference(critic, reward, must_bootstrap, discount_factor):
    target = discount_factor * critic[1:].detach() * must_bootstrap.float() + reward[1:]
    td = target - critic[:-1]
    to_add = torch.zeros(1, td.size()[1]).to(td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def doubleqlearning_temporal_difference(
    q, action, q_target, reward, must_bootstrap, discount_factor
):
    action_max = q.max(-1)[1]
    q_target_max = _index(q_target, action_max).detach()[1:]

    mb = must_bootstrap.float()
    target = reward[1:] + discount_factor * q_target_max * mb

    q = _index(q, action)[:-1]
    td = target - q
    to_add = torch.zeros(1, td.size()[1], device=td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def gae(critic, reward, must_bootstrap, discount_factor, gae_coef):
    mb = must_bootstrap.float()
    td = reward[1:] + discount_factor * mb * critic[1:].detach() - critic[:-1]
    # handling td0 case
    if gae_coef == 0.0:
        return td

    td_shape = td.shape[0]
    gae_val = td[-1]
    gaes = [gae_val]
    for t in range(td_shape - 2, -1, -1):
        gae_val = td[t] + discount_factor * gae_coef * mb[:-1][t] * gae_val
        gaes.append(gae_val)
    gaes = list([g.unsqueeze(0) for g in reversed(gaes)])
    gaes = torch.cat(gaes, dim=0)
    return gaes


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):
    batch_size = reward.size()[1]

    # Find the first occurrence of done for each element in the batch
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()
    # Shorten trajectories for faster computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = g**2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "policy_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


# Compute the temporal difference loss from a dataset to update a critic
def compute_critic_loss(cfg, reward, must_bootstrap, q_values, action):
    """_summary_

    Args:
        cfg (_type_): _description_
        reward (torch.Tensor): A (T × B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (T × B) tensor containing 0 if the episode is completed at time $t$
        q_values (torch.Tensor): a (T × B × A) tensor containing Q values
        action (torch.LongTensor): a (T) long tensor containing the chosen action

    Returns:
        torch.Scalar: The DQN loss and the temporal difference
    """

    # We check if we have transitions or not
    if len(reward.size()) == 2:
        # We compute the max of Q-values over all actions
        max_q = q_values.max(-1)[0].detach()
        target = reward[:-1] + cfg.algorithm.discount_factor * max_q * must_bootstrap.int()
        # To get Q(s,a), we use torch.gather along the 3rd dimension (the action)
        act = action[0].unsqueeze(-1)
        qvals = q_values[0].gather(dim=1, index=act).squeeze()
        # Compute the temporal difference
        td = target - qvals
    else:
        # We compute the max of Q-values over all actions
        max_q = q_values.max(2)[0].detach()
        # To get the max of Q(s_{t+1}, a), we take max_q[1:]
        # The same about must_bootstrap.
        target = reward[:-1] + cfg.algorithm.discount_factor * max_q[1:] * must_bootstrap[1:].int()
        # To get Q(s,a), we use torch.gather along the 3rd dimension (the action)
        act = action.unsqueeze(-1)
        qvals = q_values.gather(dim=2, index=act).squeeze(-1)
        # Compute the temporal difference (use must_boostrap as to mask out finished episodes)
        td = (target - qvals[:-1]) * must_bootstrap[:-1].int()

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss, td


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
