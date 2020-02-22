import math
from collections import deque
from os.path import join

import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger, get_date_str
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, \
    mpi_statistics_scalar, num_procs
from spinup.utils.save_load_scope import save_scope


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    `num_agents` allows for episodes from multiple agents in the same
    environment to be trained via self-play. To do this we simply concatenate
    their episode data in `get`.
    """

    def __init__(self, obs_dim, act_dim, max_size, gamma=0.99, lam=0.95,
                 num_agents=1):
        print(f'Max buffer size {max_size} - # agents {num_agents}')
        assert max_size % num_agents == 0
        n_size = max_size // num_agents

        self.obs_buf = np.zeros(
            core.combined_shape(num_agents,
                                core.combined_shape(n_size, obs_dim)),
            dtype=np.float32)
        self.act_buf = np.zeros(
            core.combined_shape(num_agents,
                                core.combined_shape(n_size, act_dim)),
            dtype=np.float32)
        self.adv_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.rew_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.ret_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.val_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.logp_buf = np.zeros((num_agents, n_size), dtype=np.float32)
        self.gamma, self.lam, self.num_agents = gamma, lam, num_agents
        self.ptr = np.zeros((num_agents,), dtype=np.int)
        self.path_start_idx = np.zeros((num_agents,), dtype=np.int)
        self.size = 0
        self.max_size = max_size
        self.n_size = n_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

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
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        i = agent_index

        path_slice = slice(self.path_start_idx[i], self.ptr[i])
        rews = np.append(self.rew_buf[i][path_slice], last_val)
        vals = np.append(self.val_buf[i][path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[i][path_slice] = core.discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[i][path_slice] = core.discount_cumsum(rews, self.gamma)[
                                   :-1]

        self.path_start_idx[i] = self.ptr[i]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.size == self.max_size  # buffer has to be full before you can get
        self.size = 0
        self.ptr = np.zeros((self.num_agents,), dtype=np.int)
        self.path_start_idx = np.zeros((self.num_agents,), dtype=np.int)

        obs_buf = self.obs_buf.reshape(core.combined_shape(
            self.num_agents * self.n_size, self.obs_dim))
        act_buf = self.act_buf.reshape(core.combined_shape(
            self.num_agents * self.n_size, self.act_dim))
        ret_buf = self.ret_buf.flatten()
        logp_buf = self.logp_buf.flatten()
        adv_buf = self.adv_buf.flatten()

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        return [obs_buf, act_buf, adv_buf,
                ret_buf, logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""


# noinspection DuplicatedCode
def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
        max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, resume=None,
        reinitialize_optimizer_on_resume=True, render=False,
        **kwargs):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to PPO.

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
            on the objective anymore. (Usually small, 0.1 to 0.3.)

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
            non-trainable variables in the tensorflow graph such as Adam
            state

        render: (bool) Whether to render the env during training. Useful for
            checking that resumption of training caused visual performance
            to carry over

    """

    logger = EpochLogger(**logger_kwargs)
    logger.add_key_stat('trip_pct')
    logger.add_key_stat('HorizonReturn')
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Register custom envs
    import gym_match_input_continuous
    import deepdrive_2d

    env = env_fn()

    num_agents = env.num_agents

    if hasattr(env.unwrapped, 'gamma'):
        logger.log(f'Gamma set by environment to {env.unwrapped.gamma}.'
                   f' Overriding current value of {gamma}')
        gamma = env.unwrapped.gamma

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space,
                                               env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    sess = tf.Session(
        # config=tf.ConfigProto(log_device_placement=True)
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam,
                    env.num_agents)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph,
                       (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(
        logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(
        -logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess.run(tf.global_variables_initializer())

    # Main outputs from computation graph
    if resume is not None:
        from utils.test_policy import get_policy_model
        # Caution! We assume action space has not changed here.
        should_save_model, _ = get_policy_model(resume, sess)
        pi, logp, logp_pi, v = (should_save_model['pi'], should_save_model['logp'],
                                should_save_model['logp_pi'], should_save_model['v'])

        # It looks like the first update destroys performance when
        # using this!
        # if reinitialize_optimizer_on_resume:
        #     # HACK to reinitialize our optimizer variables\
        #     trainable_variables = tf.trainable_variables()
        #     non_trainable_variables = [v for v in tf.global_variables()
        #                                if v not in trainable_variables]
        #     sess.run(tf.variables_initializer(non_trainable_variables))

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph},
                          outputs={'pi': pi, 'v': v, 'logp_pi': logp_pi,
                                   'logp': logp})

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent],
                                          feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl,
                                              clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()

    o, r, d = reset(env)

    # TODO: Make multi-agent aware
    effective_horizon = round(1 / (1 - gamma))
    effective_horizon_rewards = []
    for _ in range(num_agents):
        effective_horizon_rewards.append(deque(maxlen=effective_horizon))

    agent_index = env.agent_index
    agent = env.agents[agent_index]

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        info = {}
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(
                get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t, agent_index)
            logger.store(VVals=v_t)

            if render:
                env.render()

            # NOTE: o,r,d,info is for the next agent (from its previous action)!
            o, r, d, info = env.step(a[0])

            if 'stats' in info and info['stats']:
                logger.store(**info['stats'])

            agent_index = env.agent_index
            agent = env.agents[agent_index]

            calc_effective_horizon_reward(
                agent_index, effective_horizon_rewards, logger, r)

            ep_len = agent.episode_steps
            ep_ret = agent.episode_reward

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                          ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={
                    x_ph: o.reshape(1, -1)})
                buf.finish_path(agent_index, last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d = reset(env)

        # Save model
        should_save_model = False
        best_model = False
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            should_save_model = True

        # Perform PPO update!
        update()

        # Reset all agents
        reset(env)

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

        if 'stats' in info and info['stats']:
            for stat, value in info['stats'].items():
                logger.log_tabular(stat, with_min_and_max=True)

        if logger.new_key_stat_record:
            best_model = True
            should_save_model = True

        if should_save_model:
            logger.save_state({'env': env}, None, is_best=best_model)

        logger.dump_tabular()

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
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)