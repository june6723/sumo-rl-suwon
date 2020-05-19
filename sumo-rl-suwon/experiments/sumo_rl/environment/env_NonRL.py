import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from .traffic_signal_NonRL import TrafficSignal


class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, out_csv_path=None,out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                 time_to_load_vehicles=0, delta_time=4, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')        
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_reward = {i: 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.yellow_time = 2
        
        # addition
        self.passed_num_of_veh_in_ts = dict()
        self.pass_veh_between_intersection_start = dict()
        self.pass_veh_between_intersection_end = []
        self.wait_veh = dict()
        self.last_step_waiting_time = 0

        input = [{'gneE1_0':'gneE11_0'}, {'-gneE9_0':'gneE3_0'}, {'gneE1_0':'gneE10_0'}, {'-gneE9_0':'gneE2_0'}] # manually
        self.in_lane = []
        self.out_lane = []
        for index in input : 
            for key, value in index.items() :
                if key not in self.in_lane :    
                    self.in_lane.append(key)
                if value not in self.out_lane :
                    self.out_lane.append(value) 
                    
        """
        Default observation space is a vector R^(#greenPhases + 1 + 2 * #lanes)
        s = [current phase one-hot encoded, elapsedTime / maxGreenTime, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        # self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 1 + 2*self.lanes_per_ts), high=np.ones(self.num_green_phases + 1 + 2*self.lanes_per_ts))
        # self.action_space = spaces.Discrete(2)

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.out_csv_path = out_csv_path

        traci.close()

    # Reset before start iterations    
    def reset(self):
        if self.run != 0:
            self.save_csv(self.out_csv_path, self.out_csv_name, self.run)
            traci.close()
        self.run += 1
        self.metrics = []
        self.passed_num_of_veh_in_ts.clear()
        self.pass_veh_between_intersection_start.clear()
        self.pass_veh_between_intersection_end = []
        self.last_step_waiting_time = 0
        self.wait_veh.clear()

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000', 
                     '--random']
        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time)
            self.passed_num_of_veh_in_ts[ts] = 0
            self.last_measure[ts] = 0.0

        self.vehicles = dict()

        # Load vehicles
        for i in range(self.time_to_load_vehicles):
            self._sumo_step()
            """
            Every  time step
            """
            info = self._compute_step_info_before_rl()
            self.metrics.append(info)
            for ts in self.ts_ids:
                self.traffic_signals[ts].phase_dur_in_nonrl()
        
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getCurrentTime()/1000  # milliseconds to seconds

    def step(self, action):
        # act
        for ts in self.ts_ids :
            if self.traffic_signals[ts].time_to_act() :
                self.traffic_signals[ts].set_next_phase(action[ts])
            else :
                self.traffic_signals[ts].phase_system()
        self._sumo_step()
           
        # observe new state and reward
        observation = self._compute_observations() if self.sim_step <= self.sim_max_time else self._compute_last_observations()
        reward = self._compute_rewards() if self.sim_step <= self.sim_max_time else self._queue_last_reward()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], {}
        else:
            return observation, reward, done, {}

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        for ts in self.ts_ids:
            if self.traffic_signals[ts].time_to_act() or self.traffic_signals[ts].regular_obs() :
                observations[ts] = self.traffic_signals[ts]._compute_observation()
        return observations

    def _compute_last_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        for ts in self.ts_ids:
            observations[ts] = self.traffic_signals[ts]._compute_observation()
        return observations    

    def _compute_rewards(self):
        # return self._waiting_time_reward()
        return self._queue_reward() 
        #return self._waiting_time_reward2()
        #return self._queue_average_reward()

    def _queue_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            if self.traffic_signals[ts].time_to_act() or self.traffic_signals[ts].regular_obs() :
                rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num()))**2
        return rewards

    def _queue_last_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num()))**2
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards
            
    def _sumo_step(self):
        traci.simulationStep()  
        self._get_passed_veh_in_ts()  
        self._get_time_of_passveh_in_intersection()

    def step_stopped(self):
        veh_list = traci.vehicle.getIDList()
        step_halting_num = 0
        for veh in veh_list :
            if traci.vehicle.getWaitingTime(veh) == 1 :
                step_halting_num += 1
            self.wait_veh[veh] = traci.vehicle.getAccumulatedWaitingTime(veh)
        return step_halting_num

    def step_wait_time(self) :
        accumulated_wait_time = sum(self.wait_veh.values())
        step_wait_time = (accumulated_wait_time - self.last_step_waiting_time)
        self.last_step_waiting_time = accumulated_wait_time
        return step_wait_time

    def _compute_step_info(self):
        info = {
            'step_time': self.sim_step,
            # 'reward': self.last_reward[self.ts_ids[0]],
            'step_stopped': self.step_stopped(),
            'step_wait_time' : self.step_wait_time(),
            'accumulated_wait_time' : sum(self.wait_veh.values())
        }
        for ts in self.ts_ids :
            info.update({'{0}_passed'.format(ts) : self.passed_num_of_veh_in_ts[ts]})
        return info

    def _compute_step_info_before_rl(self):
        info = {
            'step_time': self.sim_step-1,
            # 'reward': self.last_reward[self.ts_ids[0]],
            'step_stopped': self.step_stopped(),
            'step_wait_time' : self.step_wait_time(),
            'accumulated_wait_time' : sum(self.wait_veh.values())
        }
        for ts in self.ts_ids :
            info.update({'{0}_passed'.format(ts) : self.passed_num_of_veh_in_ts[ts]})
        return info

    def _compute_phase_duration(self):
        duration = dict()
        length=[]
        for ts in self.ts_ids :
            for p in range(self.traffic_signals[ts].num_green_phases) :
                length.append(len(self.traffic_signals[ts].duration_of_green_phase[p*2]))
                
        maxlen = max(length)
        for ts in self.ts_ids :
            for p in range(self.traffic_signals[ts].num_green_phases) :
                tmp_len = len(self.traffic_signals[ts].duration_of_green_phase[p*2])
                if maxlen != tmp_len :
                    self.traffic_signals[ts].duration_of_green_phase[p*2].extend(['']*(maxlen-tmp_len))
                duration.update({'(TS : {0}  Phase : {1})'.format(ts,p*2) : self.traffic_signals[ts].duration_of_green_phase[p*2]})
        return duration

    def _compute_phase_info(self):
        info = dict()
        for ts in self.ts_ids :
            ts_object = self.traffic_signals[ts]
            for p in range(self.traffic_signals[ts].num_green_phases) :
                mean = np.mean(ts_object.duration_of_green_phase[p*2])
                var = np.var(ts_object.duration_of_green_phase[p*2])
                std = np.std(ts_object.duration_of_green_phase[p*2]) 
                info['(TS : {0}  Phase : {1})'.format(ts,p*2)] = [mean, var, std]
        return info
    
        # Get number of passed vehicle by each intersection
    def _get_passed_veh_in_ts(self):
        for ts in self.ts_ids :
            for lane in self.traffic_signals[ts].passed_veh_in_lane :    
                veh_list = traci.lane.getLastStepVehicleIDs(lane)
                for veh in veh_list :
                    if veh not in self.traffic_signals[ts].passed_veh_in_lane[lane] :
                        self.traffic_signals[ts].passed_veh_in_lane[lane].append(veh)
                        self.passed_num_of_veh_in_ts[ts] += 1

    # Get elapsed time of passed vehicle between intersection
    def _get_time_of_passveh_in_intersection(self) :
        for lane in self.in_lane :
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list :
                if veh not in self.pass_veh_between_intersection_start :
                    self.pass_veh_between_intersection_start[veh] = [self.sim_step, lane]
        for lane in self.out_lane:            
            out_veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in out_veh_list :
                if veh in self.pass_veh_between_intersection_start :
                    self.pass_veh_between_intersection_end.append({'in' : self.pass_veh_between_intersection_start[veh][0],
                                                                    'out' : self.sim_step,
                                                                    'from': self.pass_veh_between_intersection_start[veh][1],
                                                                    'to':lane,
                                                                    'elapsed_time': self.sim_step - self.pass_veh_between_intersection_start[veh][0]}
                                                                )

    def close(self):
        traci.close()

    def save_csv(self, out_csv_path, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_path + out_csv_name + '_run{}'.format(run) + '.csv', index=False)

            df2 = pd.DataFrame(self._compute_phase_info())
            df3 = pd.DataFrame(self._compute_phase_duration())
            df4 = pd.concat([df2, df3])
            df4.to_csv(out_csv_path + out_csv_name + '_Phaseduration' +'_run{}'.format(run) + '.csv', index=False)

            df5 = pd.DataFrame(self.pass_veh_between_intersection_end)
            
            df5.to_csv(out_csv_path + out_csv_name + '_between_intersection' + '_run{}'.format(run) + '.csv', index=False)
        
