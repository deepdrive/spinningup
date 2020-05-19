import time
import joblib
import os
import os.path as osp
import random

import numpy as np
import torch
from spinup import EpochLogger
from spinup.algos.pytorch.ppo import core
from spinup.algos.pytorch.ppo.ppo import do_rollouts
from spinup.utils.logx import restore_tf_graph, restore_tf_graph_model_only, \
    TF_MODEL_ONLY_DIR, PYTORCH_SAVE_DIR


# TODO: From CSQ orig, merge
def get_policy_model(fpath, sess, itr='last', use_model_only=True):

    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    if use_model_only:
        # We need this to get the same agent performance on resume.
        model = restore_tf_graph_model_only(
            sess, osp.join(fpath, f'{TF_MODEL_ONLY_DIR}/'))
    else:
        model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    return model, itr

def load_policy_and_env(fpath, itr='last', deterministic=False, env=None,
                        net_config=None):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, PYTORCH_SAVE_DIR)
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(
            fpath=fpath, itr=itr, deterministic=deterministic, env=env,
            net_config=net_config)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

# TODO: From CSQ orig, merge
def load_policy(fpath, itr='last', deterministic=False, use_model_only=False):

    sess = tf.Session()
    model, itr = get_policy_model(fpath, sess, itr,
                                  use_model_only=use_model_only)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False, actor_critic_cls=None,
                        env=None, net_config=None):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    actor_critic_cls = actor_critic_cls or core.MLPActorCritic  # TODO(support all algos): Don't assume ppo actor critic

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    checkpoint = torch.load(fname)
    if isinstance(checkpoint, actor_critic_cls):
        model = checkpoint
    else:
        model = actor_critic_cls(env.observation_space, env.action_space,
                                 hidden_sizes=net_config['hidden_units'],
                                 activation=net_config['activation'],
                                 deterministic=deterministic)
        model.load_state_dict(checkpoint['ac'])

    model.train()  # Set to train mode

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.step(x)
        return action

    return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True,
               try_rollouts=0,
               steps_per_try_rollout=0):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)

    logger = EpochLogger()
    o, r, done, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    rollout = []
    while n < num_episodes:
        if try_rollouts != 0:
            if not rollout:
                rollout = do_rollouts(
                    get_action, env, o, steps_per_try_rollout, try_rollouts,
                    is_eval=True, take_worst_rollout=False)
            a, v, logp, _o, _r, _done, _info = rollout.pop(0)
            o, r, done, info = env.step(a)
            assert np.array_equal(o, _o)
            assert r == _r
            assert done == _done
            step_output = o, r, done, info
        else:
            a = get_action(o)[0]
            step_output = env.step(a)

        if render:
            env.render()
            # time.sleep(1e-3)

        if hasattr(env, 'last_step_output'):
            step_output = env.last_step_output

        o, r, done, info = step_output

        ep_ret += r
        ep_len += 1

        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))