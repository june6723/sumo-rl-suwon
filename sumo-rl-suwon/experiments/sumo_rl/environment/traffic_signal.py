import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """
    def _get_phase_info(self) :
        phases = []
        Logic = traci.trafficlight.getAllProgramLogics(self.id) # returns logics in tuple
        for phase in Logic[0].getPhases() :  # so first element of tuple
            phases.append(phase)
            if 'y' in phase.state :
                self.min_green.append(phase.minDur)
                self.max_green.append(phase.maxDur)
            else :
                self.num_green_phases += 1
                self.min_green.append(phase.minDur)
                self.max_green.append(phase.maxDur) 
        return phases

    def _get_connected_lane(self, key_lane) :
        connected_lane = [key_lane]

        add_lane=self._add_lane_info(connected_lane)
        connected_lane += add_lane
        while(add_lane) :
            add_lane=self._add_lane_info(connected_lane)
            connected_lane += add_lane
                
        return connected_lane

    def _add_lane_info(self, connected_lane) :
        add_lane = []
        all_lanes_in_net = traci.lane.getIDList()
        for lane in all_lanes_in_net :
            if not ':' in lane and lane not in self.key_lanes :
                link = traci.lane.getLinks(lane)
                for index in link :
                    if index[0] == connected_lane[-1] and self._judge_whether_internal_lane_is_in_ts(index[4]) and lane not in add_lane :
                        add_lane.append(lane)        
        return add_lane
    
    def _judge_whether_internal_lane_is_in_ts(self, internal_lane) :
        judge = True
        # for ts_id in self.env.ts_ids :
        for ts_id in traci.trafficlight.getIDList() :
            if ts_id in internal_lane :
                judge = False
        return judge

    def __init__(self, env, ts_id):
        self.id = ts_id
        self.env = env
        self.num_green_phases = 0
        self.min_green = []
        self.max_green = []
        # Bring phase information out of net file  
        self.phases = self._get_phase_info()
        self.time_on_phase = 0.0
        self.key_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.lanes = dict()
        for lane in self.key_lanes :
            self.lanes[lane] = self._get_connected_lane(lane)

        logic = traci.trafficlight.Logic("new-program", 0, 0, phases=self.phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

        # addition
        self.duration_of_green_phase = dict() # key : GreenPhase number / value : list of duration
        for p in range(self.num_green_phases) :
            self.duration_of_green_phase[p*2] = []
        self.passed_veh_in_lane = dict() # key : Outgoing lane of ts / value : list of vehIDs
        link = traci.trafficlight.getControlledLinks(self.id)
        for incoming_lane in link :
            if incoming_lane[0][1] not in self.passed_veh_in_lane :
                self.passed_veh_in_lane[incoming_lane[0][1]] = []

        self.step_count=0

        self.observation_log=[]

    def time_to_act(self) :
        if 'y' not in self.phases[self.phase].state : # if current phase is green phase
            if self.time_on_phase >= self.min_green[self.phase] : # if elapsed time of green excess minimum green time
                return True 
            else :
                return False
        else :
            return False

    def regular_obs(self):
        if 'y' not in self.phases[self.phase].state :
            return False
        else :
            if self.time_on_phase % 4 == 0 :
                return True
            else :
                return False
        
    def _compute_observation(self) :
        """
        Return the current observation for each traffic signal
        """
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]  #one-hot encoding
        elapsed = self.time_on_phase
        density = self.get_lanes_density()
        observations = phase_id + [elapsed] + density
        return observations
        
    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def phase_system(self):
        if 'y' in self.phases[self.phase].state :
            if self.time_on_phase < self.min_green[self.phase] :
                traci.trafficlight.setPhase(self.id, self.phase)
                self.time_on_phase += 1
            else :
                next_phase = (self.phase+1) if self.phase+1 < len(self.phases) else 0
                traci.trafficlight.setPhase(self.id, next_phase)
                self.time_on_phase = 1
        else :
            if self.time_on_phase < self.min_green[self.phase] : 
                traci.trafficlight.setPhase(self.id, self.phase)
                self.time_on_phase += 1
        # if self.id == 'gneJ00':
        #     print('ID : {0} Action :      phase : {1} time_on_phase/current_min_green  : {2}/{3}'
        #               .format(self.id, self.phase, self.time_on_phase, self.min_green[self.phase]))
        

    def set_next_phase(self, action):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """
    
        if action == 0 and self.time_on_phase < self.max_green[self.phase] :
            traci.trafficlight.setPhase(self.id, self.phase)
            self.time_on_phase += 1
        else:
            self.duration_of_green_phase[self.phase].append(self.time_on_phase)
            self.time_on_phase = 1
            next_phase = (self.phase+1) if self.phase+1 < len(self.phases) else 0
            traci.trafficlight.setPhase(self.id, next_phase)  # turns yellow
        # if self.id == 'gneJ00':
        #     print('ID : {0} Action : {1}    phase : {2} time_on_phase/current_min_green  : {3}/{4}'
        #               .format(self.id, action, self.phase, self.time_on_phase, self.min_green[self.phase]))

    def get_lanes_density(self):
        density = []
        for key in self.lanes :
            sum = 0
            for lane in self.lanes[key] :
                sum += traci.lane.getLastStepVehicleNumber(lane)
            density.append(sum)
        return density

    def get_lanes_queue_sum(self):
        sum = 0
        for key in self.lanes :
            for lane in self.lanes[key] :
                sum += traci.lane.getLastStepHaltingNumber(lane)
        return sum
    