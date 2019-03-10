from obstacle_tower_env import ObstacleTowerEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
from baselines.bench import Monitor
from baselines import logger
import baselines.ppo2.ppo2 as ppo2 

from settings import OBS_TOWER_ENVPATH

import os

try:
    from mpi4py import MPI 
except ImportError:
    MPI = None

def make_unity_env(env_directory, num_env, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = ObstacleTowerEnv(env_directory, worker_id=rank, realtime_mode=True)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main():
    print('Executable environment', OBS_TOWER_ENVPATH)
    env = make_unity_env(OBS_TOWER_ENVPATH, 1)

    ppo2.learn(
        network='mlp',
        env=env,
        total_timesteps=1e5,
        lr=1e-3,
    )
    ppo2.save('obs_tower_chall_model.pkl')

if __name__ == '__main__':
    main()