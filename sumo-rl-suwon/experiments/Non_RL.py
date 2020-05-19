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
    if id == "left":
        return "left"
    else:
        return "right"

if __name__ == '__main__':
    ray.init()

    register_env("2TLS", lambda _: SumoEnvironment(net_file='nets/Research/case03/intersection_NonRL.net.xml',
                                                    route_file='nets/Research/case03/test.rou.xml',
                                                    out_csv_path='outputs/grad/',
                                                    out_csv_name='nonrl',
                                                    use_gui=True,
                                                    num_seconds=22000,
                                                    time_to_load_vehicles=21600,
                                                    max_depart_delay=0)
                    )

    trainer = DQNTrainer(env="2TLS", config={
        "multiagent": {
            "policy_graphs": {
                'left': (DQNTFPolicy, spaces.Box(low=np.zeros(21), high=np.ones(21)), spaces.Discrete(2), {}),
                'right': (DQNTFPolicy, spaces.Box(low=np.zeros(21), high=np.ones(21)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        "lr": 0.0001,
    })
    
    while True:
        result = trainer.train()
