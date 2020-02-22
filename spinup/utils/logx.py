"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import json
from collections import deque
from datetime import datetime
from glob import glob
from os.path import dirname, join

import joblib
import shutil
import numpy as np
import tensorflow as tf
import os.path as osp, time, atexit, os
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar
from spinup.utils.serialization_utils import convert_json
from spinup.utils.save_load_scope import load_scope, save_scope

SIMPLE_SAVE_DIR = 'simple_save'
MODEL_ONLY_DIR = 'model_only'

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model


def restore_tf_graph_model_only(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs'
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``.
    """

    # DOESN'T WORK - Still loads Adam variables, need reinitialize hack to
    # workaround. Keeping this, however, as it explicitly defines variables
    # to save via scope, and hopefully will help in the future with variables
    # that tf CAN actually extract. Also note that reinitializing Adam
    # vars is not actually a good thing at least for the initial performance
    # on resume. So said hack is disabled.

    load_scope('model', fpath, sess)
    model_info_dir = join(dirname(dirname(fpath)), SIMPLE_SAVE_DIR)
    model_info = joblib.load(osp.join(model_info_dir, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model


def get_date_str():
    date_str = datetime.now().strftime('%Y_%m-%d_%H-%M.%S')
    return date_str


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt',
                 exp_name=None, num_snapshots_to_keep=10,
                 snapshot_save_freq_mins=30):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id()==0:
            if output_dir:
                date_str = get_date_str()
                self.output_dir = output_dir + '_' + date_str
            else:
                self.output_dir = "/tmp/experiments/%i"%int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.snapshot_save_freq_mins = snapshot_save_freq_mins
        self.num_snapshots_to_keep = num_snapshots_to_keep
        self.best_model_snapshots = deque(maxlen=num_snapshots_to_keep)
        self.timed_snapshots = deque(maxlen=num_snapshots_to_keep)

        # Special stats that allow us to save models when new levels of
        # performance are reached.
        self.key_stats = {}
        self.add_key_stat('EpRet')

        # Track whether a new high value was encountered this iteration/epoch
        self.new_key_stat_record = False

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None, is_best=False):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.

            is_best: Whether this model's performance reached a new record high,
                in which case we put it in a special folder
        """
        if proc_id()==0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'tf_saver_elements'):
                self._tf_simple_save(itr)
                self._save_model_only(itr)
                self._save_snapshots(is_best)

    def setup_tf_saver(self, sess, inputs, outputs):
        """
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the 
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {'inputs': {k:v.name for k,v in inputs.items()},
                              'outputs': {k:v.name for k,v in outputs.items()}}

    def _tf_simple_save(self, itr=None):
        """
        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore. 
        """
        if proc_id() == 0:
            assert hasattr(self, 'tf_saver_elements'), \
                "First have to setup saving with self.setup_tf_saver"
            fpath = SIMPLE_SAVE_DIR + ('%d'%itr if itr is not None else '')
            fpath = osp.join(self.output_dir, fpath)
            if osp.exists(fpath):
                # simple_save refuses to be useful if fpath already exists,
                # so just delete fpath if it's there.
                shutil.rmtree(fpath)
            tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
            joblib.dump(self.tf_saver_info, osp.join(fpath, 'model_info.pkl'))

    def _save_snapshots(self, is_best=False):
        if proc_id() != 0 or not self.num_snapshots_to_keep:
            return

        now = time.time()
        if is_best:
            save_snapshot = True
            snapshots = self.best_model_snapshots
            snapshots_dir = join(self.output_dir, 'best')
        else:
            snapshots = self.timed_snapshots
            snapshots_dir = join(self.output_dir, 'snapshots')
            if not self.timed_snapshots:
                save_snapshot = True
            else:
                prev_time, _prev_dir = self.timed_snapshots[-1]
                save_snapshot = prev_time < (now - self.snapshot_save_freq_mins * 60)

        if save_snapshot:
            if len(snapshots) == snapshots.maxlen:
                # Roll snapshots
                _old_time, old_dir = snapshots.popleft()
                shutil.rmtree(old_dir)
            new_dir = join(snapshots_dir, get_date_str())
            os.makedirs(new_dir)
            curr_simple_save_dir = join(self.output_dir, SIMPLE_SAVE_DIR)
            curr_model_only_dir = join(self.output_dir, MODEL_ONLY_DIR)
            shutil.copytree(curr_simple_save_dir, join(new_dir, SIMPLE_SAVE_DIR))
            shutil.copytree(curr_model_only_dir, join(new_dir, MODEL_ONLY_DIR))

            snapshots.append((now, new_dir))

    def _save_model_only(self, itr=None):
        # logger.save_state saves optimizer state which we don't want for
        # resuming purposes, so save model variables here separately
        save_scope('model', join(self.output_dir, f'{MODEL_ONLY_DIR}/'),
                   self.tf_saver_elements['session'])

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                print(fmt%(key, valstr))
                vals.append(val)
            print("-"*n_slashes)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join(map(str,vals))+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

    def track_key_stats(self, key, stats):
        if key in self.key_stats:
            if len(stats) == 2:
                print('Key stats need min and max values!')
            else:
                # TODO: Name the model folder after the stat that was best
                mean, std, global_min, global_max = stats
                curr_stat = self.key_stats[key]
                if mean > curr_stat['best_avg']:
                    curr_stat['best_avg'] = mean
                    self.new_key_stat_record = True
                elif global_max > curr_stat['best']:
                    curr_stat['best'] = global_max
                    self.new_key_stat_record = True

    def add_key_stat(self, key):
        assert key not in self.key_stats, f'Key {key} already added'
        self.key_stats = {key: {'best': -np.inf, 'best_avg': -np.inf}}


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not is_number(v):
                pass
                # print(f'warning - value for {k} is not a number: {v}')
            else:
                if not(k in self.epoch_dict.keys()):
                    self.epoch_dict[k] = []
                self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            if key not in self.epoch_dict:
                return False
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
            self.track_key_stats(key, stats)
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)


def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False