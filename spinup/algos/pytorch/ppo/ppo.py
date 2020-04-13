import math
import os
from collections import deque
from copy import deepcopy
import random
from numba.typed import List
import numpy as np
import torch
from numba import njit
from torch.nn import init
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.custom_envs import import_custom_envs
from spinup.utils.logx import EpochLogger, get_date_str, PYTORCH_SAVE_DIR
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, max_size, gamma=0.99, lam=0.95,
                 num_agents=1, shift_advs_pct=0.0):
        print(f'Max buffer size {max_size} - # agents {num_agents}')
        assert max_size % num_agents == 0
        n_size = max_size // num_agents

        self.obs_buf = np.zeros(core.combined_shape(num_agents, core.combined_shape(n_size, obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(num_agents, core.combined_shape(n_size, act_dim)), dtype=np.float32)
        self.adv_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.rew_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.ret_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.val_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.logp_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.num_agents = num_agents
        self.ptr = np.zeros((num_agents,), dtype=np.int)
        self.path_start_idx = np.zeros((num_agents,), dtype=np.int)
        self.size = 0
        self.max_size = max_size
        self.n_size = n_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shift_advs_pct = shift_advs_pct

    def epoch_ended(self, step_num):
        return step_num == self.max_size - 1

    def store(self, obs, act, rew, val, logp, agent_index):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.size < self.max_size  # buffer has to have room so you can store
        self.obs_buf[agent_index][self.ptr[agent_index]] = obs
        self.act_buf[agent_index][self.ptr[agent_index]] = act
        self.rew_buf[agent_index][self.ptr[agent_index]] = rew
        self.val_buf[agent_index][self.ptr[agent_index]] = val
        self.logp_buf[agent_index][self.ptr[agent_index]] = logp
        self.ptr[agent_index] += 1
        self.size += 1

    def finish_path(self, agent_index, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to estimate the reward-to-go calculation via bootstrapping
        to account for timesteps beyond the arbitrary episode horizon
        (or epoch cutoff).
        """

        i = agent_index

        path_slice = slice(self.path_start_idx[i], self.ptr[i])
        rews = np.append(self.rew_buf[i][path_slice], last_val)
        vals = np.append(self.val_buf[i][path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[i][path_slice] = core.discount_cumsum(
            deltas, self.gamma * self.lam)

        if self.shift_advs_pct:
            advs = self.adv_buf[i][path_slice]
            # TODO: Tie this to Entropy
            # 90 seems to work speed up convergence whereas 99 and 99.9 slow it down
            # perhaps something to do with size of 1 standard deviation
            self.adv_buf[i][path_slice] = advs - np.percentile(advs, self.shift_advs_pct)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[i][path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx[i] = self.ptr[i]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.size == self.max_size  # buffer has to be full before you can get

        # Reset pointers so next epoch overwrites buffers
        self.size = 0
        self.ptr = np.zeros((self.num_agents,), dtype=np.int)
        self.path_start_idx = np.zeros((self.num_agents,), dtype=np.int)

        # Concatenate agents' episodes
        obs_buf = self.obs_buf.reshape(core.combined_shape(self.num_agents * self.n_size, self.obs_dim))
        act_buf = self.act_buf.reshape(core.combined_shape(self.num_agents * self.n_size, self.act_dim))
        ret_buf = self.ret_buf.flatten()
        logp_buf = self.logp_buf.flatten()
        adv_buf = self.adv_buf.flatten()

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf,
                    adv=adv_buf, logp=logp_buf)

        # TODO: See if we are copying below if we run into memory issues
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def record_episode(self, ep_len, ep_ret, step_num):
        # Implementation only needed if culling currently
        pass

    def prepare_for_update(self):
        # Implementation only needed if culling currently
        pass


class PPOBufferCulled(PPOBuffer):
    def __init__(self, obs_dim, act_dim, local_steps_per_epoch, gamma, lam,
                 num_agents, shift_advs_pct, cull_ratio):
        super().__init__(obs_dim, act_dim, local_steps_per_epoch, gamma, lam,
                         num_agents, shift_advs_pct)
        self.cull_ratio = cull_ratio
        if cull_ratio != 0:
            self._reset()

    def _reset(self):
        self._staging = []
        self.episode_returns = List()  # TODO: Make this muli-agent
        self.episode_lengths = List()  # TODO: Make this muli-agent
        self.good_episode_indexes = None  # Index of each episode in a reverse sorted list by return # TODO: Make this muli-agent
        self.total_good_steps = 0  # TODO: Make this muli-agent
        self.complete_episode_steps = 0  # TODO: Make this muli-agent
        self.episode_number = 0  # TODO: Make this muli-agent
        self.last_val = 0  # TODO: Muli-agent

    def finish_path(self, agent_index, last_val=0, stage=True):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to estimate the reward-to-go calculation via bootstrapping
        to account for timesteps beyond the arbitrary episode horizon
        (or epoch cutoff).
        """
        if stage:
            # Record last value for final step of epoch (since we don't know
            # reward to go).
            self.last_val = last_val  # TODO: Make this multi-agent

            # We'll do this after culling - will need to change for
            # general approach though
            return
        else:
            super().finish_path(agent_index, last_val=last_val)

    def store(self, obs, act, rew, val, logp, agent_index, stage=True):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if stage:
            self._stage_step(obs, act, rew, val, logp, agent_index)
        else:
            super().store(obs, act, rew, val, logp, agent_index)

    def prepare_for_update(self):
        self._store_good_episodes()

    def epoch_ended(self, step_num):
        # TODO: Make this multi-agent
        # TODO: This needs to be made more robust by replaying from
        #  low valued(judged by return to go) states
        #  to ensure returns aren't just low because the situation is tricky.
        #  For now, I'm testing in an env where each episode is pretty much
        #  the same difficulty to POC this idea so am just culling entire
        #  episodes if they have low returns.
        new_steps = step_num - self.complete_episode_steps
        # Last (partial) episode could be bad, but we're willing to live with
        # that.
        ret = self.total_good_steps + new_steps >= self.max_size
        if ret:
            self._set_last_step_of_episode()
        return ret

    def record_episode(self, ep_len, ep_ret, step_num):
        # TODO: Make this multi-agent
        self.episode_returns.append(ep_ret)
        self.episode_lengths.append(ep_len)
        keep_ratio = 1 - self.cull_ratio
        episodes_to_keep = math.ceil(len(self.episode_returns) * keep_ratio)
        self.good_episode_indexes = \
            np.argsort(self.episode_returns)[::-1][:episodes_to_keep]
        self.total_good_steps = get_total_good_steps(
            ep_lengths=self.episode_lengths,
            ep_indexes=self.good_episode_indexes)
        self.complete_episode_steps = step_num
        self.episode_number += 1
        self._set_last_step_of_episode()

    def _stage_step(self, obs, act, rew, val, logp, agent_index):
        last_step_of_episode = False  # Will be updated if it is
        self._staging.append([obs, act, rew, val, logp, agent_index,
                              self.episode_number, last_step_of_episode])

    def _store_good_episodes(self):
        for staged in self._staging:
            (obs, act, rew, val, logp, agent_index, episode_number,
             last_step_of_episode) = staged
            if episode_number in self.good_episode_indexes:
                self.store(obs, act, rew, val, logp, agent_index, stage=False)
                if last_step_of_episode:
                    self.finish_path(agent_index, stage=False)
            elif episode_number == self.episode_number:
                # TODO: Make this multi-agent
                self.store(obs, act, rew, val, logp, agent_index, stage=False)
                if last_step_of_episode:
                    self.finish_path(agent_index, stage=False, last_val=self.last_val)
        self._reset()

    def _set_last_step_of_episode(self):
        self._staging[-1][-1] = True  # Set last_step_of_episode to True


def ppo_buffer_factory(obs_dim, act_dim, local_steps_per_epoch, gamma, lam,
                       num_agents, shift_advs_pct, cull_ratio):
    if cull_ratio == 0:
        return PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam,
                         num_agents, shift_advs_pct)
    else:
        return PPOBufferCulled(obs_dim, act_dim, local_steps_per_epoch, gamma,
                               lam, num_agents, shift_advs_pct, cull_ratio)


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, resume=None,
        reinitialize_optimizer_on_resume=True, render=False, notes='',
        env_config=None, boost_explore=0, partial_net_load=False,
        num_inputs_to_add=0, episode_cull_ratio=0, try_rollouts=0,
        steps_per_try_rollout=0, take_worst_rollout=False, shift_advs_pct=0,
        **kwargs):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        resume (str): Path to directory with simple_save model info
            you wish to resume from

        reinitialize_optimizer_on_resume: (bool) Whether to initialize
            training state in the optimizers, i.e. the individual learning
            rates for weights in Adam

        render: (bool) Whether to render the env during training. Useful for
            checking that resumption of training caused visual performance
            to carry over

        notes: (str) Experimental notes on what this run is testing

        env_config (dict): Environment configuration pass through

        boost_explore (float): Amount to increase std of actions in order to
        reinvigorate exploration.

        partial_net_load (bool): Whether to partially load the network when
        resuming. https://pytorch.org/tutorials/beginner/saving_loading_models.html#id4

        num_inputs_to_add (int): Number of new inputs to add, if resuming and
        partially loading a new network.

        episode_cull_ratio (float): Ratio of bad episodes to cull
        from epoch

        try_rollouts (int): Number of times to sample actions

        steps_per_try_rollout (int): Number of steps per attempted rollout

        take_worst_rollout (bool): Use worst rollout in training

        shift_advs_pct (float): Action should be better than this pct of actions
            to be considered advantageous.
    """
    config = deepcopy(locals())

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    import_custom_envs()

    # Instantiate environment
    env = env_fn()
    if hasattr(env.unwrapped, 'configure_env'):
        env.unwrapped.configure_env(env_config)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    num_agents = getattr(env, 'num_agents', 1)

    if hasattr(env.unwrapped, 'logger'):
        print('Logger set by environment')
        logger_kwargs['logger'] = env.unwrapped.logger

    logger = EpochLogger(**logger_kwargs)
    logger.add_key_stat('won')
    logger.add_key_stat('trip_pct')
    logger.add_key_stat('HorizonReturn')
    logger.save_config(config)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space,
                      num_inputs_to_add=num_inputs_to_add, **ac_kwargs)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Resume
    if resume is not None:
        ac, pi_optimizer, vf_optimizer = get_model_to_resume(
            resume, ac, pi_lr, vf_lr, reinitialize_optimizer_on_resume,
            actor_critic, partial_net_load, num_inputs_to_add)
        if num_inputs_to_add:
            add_inputs(ac, ac_kwargs, num_inputs_to_add)

    if boost_explore:
        boost_exploration(ac, boost_explore)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = ppo_buffer_factory(obs_dim, act_dim, local_steps_per_epoch, gamma,
                             lam, num_agents, shift_advs_pct,
                             cull_ratio=episode_cull_ratio)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Sync params across processes
    sync_params(ac)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, r, d = reset(env)

    effective_horizon = round(1 / (1 - gamma))
    effective_horizon_rewards = []
    for _ in range(num_agents):
        effective_horizon_rewards.append(deque(maxlen=effective_horizon))

    if hasattr(env, 'agent_index'):
        agent_index = env.agent_index
        agent = env.agents[agent_index]
        is_multi_agent = True
    else:
        agent_index = 0
        agent = None
        is_multi_agent = False

    def get_action_fn(_obz):
        return ac.step(torch.as_tensor(_obz, dtype=torch.float32))

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        epoch_episode = 0
        info = {}
        epoch_ended = False
        step_num = 0
        ep_len = 0
        ep_ret = 0
        while not epoch_ended:
            if try_rollouts != 0:
                # a, v, logp, next_o, r, d, info
                # a, v, logp, obs, r, done, info
                rollout = do_rollouts(
                    get_action_fn, env, o, steps_per_try_rollout, try_rollouts,
                    take_worst_rollout)
            else:
                a, v, logp = get_action_fn(o)
                # NOTE: For multi-agent, steps current agent,
                # but returns values for next agent (from its previous action)!
                # TODO: Just return multiple agents observations
                next_o, r, d, info = env.step(a)

            if render:
                env.render()

            curr_reward = env.curr_reward if is_multi_agent else r

            # save and log
            buf.store(o, a, curr_reward, v, logp, agent_index)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            if 'stats' in info and info['stats']:  # TODO: Optimize this
                logger.store(**info['stats'])

            if is_multi_agent:
                agent_index = env.agent_index
                agent = env.agents[agent_index]

                # TODO: Store vector of these for each agent when changing step API
                ep_len = agent.episode_steps
                ep_ret = agent.episode_reward
            else:
                ep_len += 1
                ep_ret += r

            calc_effective_horizon_reward(
                agent_index, effective_horizon_rewards, logger, r)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = buf.epoch_ended(step_num)
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(agent_index, v)
                if terminal:
                    buf.record_episode(ep_len=ep_len, ep_ret=ep_ret, step_num=step_num)
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if 'stats' in info and info['stats'] and info['stats']['done_only']:
                        logger.store(**info['stats']['done_only'])
                o, r, d = reset(env)
                if not is_multi_agent:
                    ep_len = 0
                    ep_ret = 0
            step_num += 1

        buf.prepare_for_update()

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('DateTime', get_date_str())
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('HorizonReturn', with_min_and_max=True)
        if getattr(env.unwrapped, 'is_deepdrive', False):
            logger.log_tabular('trip_pct', with_min_and_max=True)
            logger.log_tabular('collided')
            logger.log_tabular('harmful_gs')
            logger.log_tabular('timeup')
            logger.log_tabular('exited_lane')
            logger.log_tabular('circles')
            logger.log_tabular('skipped')
            logger.log_tabular('backwards')
            logger.log_tabular('won')

        if 'stats' in info and info['stats']:
            for stat, value in info['stats'].items():
                logger.log_tabular(stat, with_min_and_max=True)

        if logger.best_category or (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state(dict(env=env), pytorch_save=dict(
                ac=ac.state_dict(),
                pi_optimizer=pi_optimizer.state_dict(),
                vf_optimizer=vf_optimizer.state_dict(),
                epoch=epoch,
                stats=logger.epoch_dict,
            ), itr=None, best_category=logger.best_category)

        logger.dump_tabular()


def do_rollouts(get_action_fn, env, start_o, steps_per_try_rollout,
                try_rollouts, take_worst_rollout=False, is_eval=False):
    start_state = env.get_state()
    rollouts = []
    end_states = []
    best_ret = math.inf if take_worst_rollout else -math.inf
    best_rollout_i = None
    for rollout_i in range(try_rollouts):
        ret, rollout = do_rollout(start_o, env, get_action_fn,
                                  is_eval, steps_per_try_rollout)
        # IDEA: Rank by value instead of return
        rollouts.append(rollout)
        end_states.append(env.get_state())
        if take_worst_rollout:
            if ret < best_ret:
                best_rollout_i = rollout_i
                best_ret = ret
        elif ret > best_ret:
            best_rollout_i = rollout_i
            best_ret = ret
        if is_eval or rollout_i != try_rollouts - 1:
            # No need to reset on last rollout as we'll be loading best
            # rollout's state immediately afterwards during training.
            # In eval, we need to render out the steps, so it's not much
            # overhead to rerun the actions
            # (though will be different if not deterministic)
            env.set_state(start_state)
    assert best_rollout_i is not None
    # a, v, logp, o, r, d, info = rollouts[best_rollout_i]
    best_rollout = rollouts[best_rollout_i]
    if not is_eval:
        env.set_state(end_states[best_rollout_i])
    return best_rollout


def do_rollout(obs, env, get_action_fn, is_eval, steps_per_try_rollout):
    rollout = []
    ret = 0
    for try_step in range(steps_per_try_rollout):
        if is_eval:
            # We just get action during eval.
            a, v, logp = get_action_fn(obs)
        else:
            a, v, logp = get_action_fn(obs)
        obs, r, done, info = env.step(a)
        rollout.append((a, v, logp, obs, r, done, info))
        ret += r
        if done:
            break
    return ret, rollout


def add_inputs(ac, ac_kwargs, num_inputs_to_add):
    # Only works for MLP
    input_layers = ac.pi.mu_net[0], ac.v.v_net[0]
    for layer in input_layers:
        new_input_weights = torch.nn.Parameter(
            torch.Tensor(ac_kwargs['hidden_sizes'][0],
                         num_inputs_to_add))

        # Random init
        # Note: sqrt(5) just makes init random vs
        # dependent on activation etc...
        # https://github.com/pytorch/pytorch/issues/15314
        init.kaiming_uniform_(new_input_weights, a=math.sqrt(5))

        # Decrease initial weights of new inputs in order to slowly
        # ramp up on them.
        new_input_weights.data = torch.mul(new_input_weights.data, 1e-4)

        layer.weight = torch.nn.Parameter(
            torch.cat((layer.weight, new_input_weights), dim=1))


def boost_exploration(ac, boost_explore):
    state_dict = ac.pi.state_dict()
    for name, param in state_dict.items():
        # Don't update if this is not a weight.
        if name == 'log_std':
            # Transform the parameter as required.
            transformed_param = torch.log(torch.exp(param) * boost_explore)
            # Update the parameter.
            state_dict[name].copy_(transformed_param)


def get_model_to_resume(resume, ac, pi_lr, vf_lr,
                        reinitialize_optimizer_on_resume,
                        actor_critic,
                        partial_net_load=False,
                        num_inputs_to_add=0):
    model_path = os.path.join(resume, PYTORCH_SAVE_DIR, 'model.pt')
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, actor_critic):
        if reinitialize_optimizer_on_resume:
            raise RuntimeError('No optimizer state in this checkpoint')
        if partial_net_load or num_inputs_to_add:
            raise NotImplementedError('Partially loading non-state-dict models not implemented')
        ac = checkpoint

        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    else:
        ac.load_state_dict(checkpoint['ac'], strict=(not partial_net_load))
        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
        if reinitialize_optimizer_on_resume:
            pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
            vf_optimizer.load_state_dict(checkpoint['vf_optimizer'])
    ac.train()  # Set to train mode
    return ac, pi_optimizer, vf_optimizer


def calc_effective_horizon_reward(agent_index, effective_horizon_rewards,
                                  logger, r):
    ehr = effective_horizon_rewards[agent_index]
    ehr.append(r)
    logger.store(HorizonReturn=sum(ehr))


def reset(env):
    """
    Resets env on first call, after that resets/respawns the current agent
    :return: obs, reward, done
    """
    return env.reset(), 0, False

@njit(nogil=True)
def get_total_good_steps(ep_lengths, ep_indexes):
    total = 0
    for ep_i in ep_indexes:
        total += ep_lengths[ep_i]
    return total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)