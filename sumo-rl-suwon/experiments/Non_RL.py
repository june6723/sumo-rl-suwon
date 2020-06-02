import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn import DQNTFPolicy
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from gym import spaces
import numpy as np
from sumo_rl.environment.env_NonRL import SumoEnvironment
import traci

def policy_mapping(id):
    if id == "3210041371":
        return "3210041371"

    elif id == "452397025":
        return "452397025"

    elif id == "4708662059":
        return "4708662059"
        
    else :
        return "5870232715"

if __name__ == '__main__':
    ray.init()

    register_env("2TLS", lambda _: SumoEnvironment(net_file='/home/sonic/Desktop/sumo-rl-suwon/sumo-rl-suwon/experiments/nets/suwon/osm.net.xml',
                                                    route_file='/home/sonic/Desktop/sumo-rl-suwon/sumo-rl-suwon/experiments/nets/suwon/osm_nonrl.rou.xml',
                                                    out_csv_path='outputs/NonRL3/',
                                                    out_csv_name='Nonrl',
                                                    use_gui=True,
                                                    num_seconds=44000,
                                                    time_to_load_vehicles=43200,
                                                    max_depart_delay=0)
                    )

    trainer = DQNTrainer(env="2TLS", config={
        "multiagent": {
            "policy_graphs": {
                '3210041371': (DQNTFPolicy, spaces.Box(low=np.zeros(16), high=np.array(['inf']*16)), spaces.Discrete(2), {}),
                '452397025': (DQNTFPolicy, spaces.Box(low=np.zeros(14), high=np.array(['inf']*14)), spaces.Discrete(2), {}),
                '4708662059': (DQNTFPolicy, spaces.Box(low=np.zeros(19), high=np.array(['inf']*19)), spaces.Discrete(2), {}),
                '5870232715': (DQNTFPolicy, spaces.Box(low=np.zeros(10), high=np.array(['inf']*10)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        "lr": 0.0001,
    })
    
    while True:
        result = trainer.train()
