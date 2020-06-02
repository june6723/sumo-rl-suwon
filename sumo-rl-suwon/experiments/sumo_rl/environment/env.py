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

from .traffic_signal import TrafficSignal


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
                 time_to_load_vehicles=0, single_agent=False):

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
        self.traffic_signals = dict()
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        
        # addition
        self.passed_num_of_veh_in_ts = dict()
        self.pass_veh_between_intersection_start = dict()
        self.pass_veh_between_intersection_end = []
        self.wait_veh = dict()
        self.last_step_waiting_time = 0

        self.flow_list = [1,2,3,4,5,13,15]
        self.travel_log = dict()
        self.travel_time_by_flow = dict()

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.out_csv_path = out_csv_path

        traci.close()

    def slice_veh_id(self, vehid) :
        if '_' in vehid:
            return 'none'
        elif vehid[7] == '0' :
            return 'carflow{0}'.format(vehid[8])
        else :
            flow_number = int(vehid[7:vehid.index('.')]) % 16
            return 'carflow{0}'.format(flow_number)

    def get_departed_veh(self) :
        for veh in traci.simulation.getDepartedIDList() :
            self.travel_log[veh] = traci.simulation.getTime()

    def get_travel_time(self) :
        for veh in traci.simulation.getArrivedIDList() :
            if veh in self.travel_log and self.slice_veh_id(veh) in self.travel_time_by_flow:
                self.travel_time_by_flow[self.slice_veh_id(veh)].append([self.travel_log[veh], traci.simulation.getTime() - self.travel_log[veh]])

    # Reset before start iterations    
    def reset(self):
        if self.run != 0:
            self.save_csv(self.out_csv_path, self.out_csv_name, self.run)
            traci.close()
        self.run += 1
        self.metrics = []
        # origin
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000', 
                     '--random']
        
        self.metrics = []
        self.passed_num_of_veh_in_ts.clear()
        self.pass_veh_between_intersection_start.clear()
        self.pass_veh_between_intersection_end = []
        self.last_step_waiting_time = 0
        self.wait_veh.clear()

        self.travel_log.clear()
        for flow in self.flow_list :
            self.travel_time_by_flow["carflow{0}".format(flow)] = []

        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts)
            self.passed_num_of_veh_in_ts[ts] = 0

        # Load vehicles
        for i in range(self.time_to_load_vehicles):
            self._sumo_step()
        
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # act
        for ts in self.ts_ids :
            if self.traffic_signals[ts].time_to_act() :
                self.traffic_signals[ts].set_next_phase(action[ts])
            else :
                self.traffic_signals[ts].phase_system()
        self._sumo_step()
           
        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        
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
            if self.traffic_signals[ts].time_to_act() or self.sim_step > self.sim_max_time :
                observations[ts] = self.traffic_signals[ts]._compute_observation()
                self.traffic_signals[ts].observation_log.append({ 'step_time': self.sim_step, 'observation' : observations[ts], 'reward ': -1 * self.traffic_signals[ts].get_lanes_queue_sum()})
        return observations

    def _compute_rewards(self):
        rewards = {}
        for ts in self.ts_ids:
            if self.traffic_signals[ts].time_to_act() or self.sim_step > self.sim_max_time :
                rewards[ts] = -1 * self.traffic_signals[ts].get_lanes_queue_sum()
        return rewards
        
    def _sumo_step(self):
        traci.simulationStep()  
        self.get_departed_veh()
        self.get_travel_time()

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
            'step_stopped': self.step_stopped(),
            'step_wait_time' : self.step_wait_time(),
            'accumulated_wait_time' : sum(self.wait_veh.values()),
        }
        return info

    def _compute_step_info_before_rl(self):
        info = {
            'step_time': self.sim_step,
            'step_stopped': self.step_stopped(),
            'step_wait_time' : self.step_wait_time(),
            'accumulated_wait_time' : sum(self.wait_veh.values())
        }
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

            for flow in self.travel_time_by_flow :
                tmp_metrics=[]
                index=0
                for _ in self.travel_time_by_flow[flow] :
                    tmp_metrics.append({'departed' : self.travel_time_by_flow[flow][index][0], 'travel_time' : self.travel_time_by_flow[flow][index][1]})
                    index += 1
                tmp_df = pd.DataFrame(tmp_metrics)
                tmp_df.to_csv(out_csv_path + out_csv_name + '_' + flow + '_run{}'.format(run) + '.csv', index=False)

            for ts in self.ts_ids :
                df_obs = pd.DataFrame(self.traffic_signals[ts].observation_log)
                df_obs.to_csv(out_csv_path + out_csv_name + '_' + ts + '_run{}'.format(run) + '.csv', index=False)



