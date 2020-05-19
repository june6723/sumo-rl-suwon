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
            if 'G' in phase.state or 'g' in phase.state :
                self.num_green_phases += 1
                self.min_green.append(phase.minDur)
                self.max_green.append(phase.maxDur)
            else :
                self.min_green.append(phase.minDur)
                self.max_green.append(phase.maxDur) 
        return phases

    def __init__(self, env, ts_id, delta_time, yellow_time):
        self.id = ts_id
        self.env = env
        self.num_green_phases = 0
        self.min_green = []
        self.max_green = []
        # Bring phase information out of net file  
        self.phases = self._get_phase_info()
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.edges = self._compute_edges()
        self.edges_capacity = self._compute_edges_capacity()
        self.yellow_time = yellow_time
       
        # logic = traci.trafficlight.Logic("new-program", 0, 0, phases=self.phases)
        # traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

        l = traci.lane.getIDList()
        # for lane in self.lanes :
        info = traci.lane.getLinks('-gneE6_0')
        # addition
        self.duration_of_green_phase = dict() # key : GreenPhase number / value : list of duration
        for p in range(self.num_green_phases) :
            self.duration_of_green_phase[p*2] = []
        self.passed_veh_in_lane = dict() # key : Outgoing lane of ts / value : list of vehIDs
        link = traci.trafficlight.getControlledLinks(self.id)
        for incoming_lane in link :
            if incoming_lane[0][1] not in self.passed_veh_in_lane :
                self.passed_veh_in_lane[incoming_lane[0][1]] = []

    def phase_dur_in_nonrl(self):
        if 'y' not in self.phases[self.phase].state:
            self.time_on_phase += 1
            if self.min_green[self.phase] == self.time_on_phase :
                self.duration_of_green_phase[self.phase].append(self.time_on_phase)
                self.time_on_phase = 0
                
   
    def time_to_act(self) :
        if 'g' in self.phases[self.phase].state or 'G' in self.phases[self.phase].state : # if current phase is green phase
            if self.time_on_phase >= self.min_green[self.phase] : # if elapsed time of green excess minimum green time
                return True 
            else :
                return False
        else :
            return False

    def regular_obs(self):
        if 'y' in self.phases[self.phase].state :
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
        elapsed = self.time_on_phase / self.max_green[self.phase]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observations = phase_id + [elapsed] + density + queue
        return observations
        
    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def phase_system(self):
        if 'y' in self.phases[self.phase].state :
            if self.time_on_phase < self.yellow_time :
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
            # self.duration_of_green_phase[self.phase].append(self.time_on_phase)
            self.time_on_phase = 1
            next_phase = (self.phase+1) if self.phase+1 < len(self.phases) else 0
            traci.trafficlight.setPhase(self.id, next_phase)  # turns yellow
        # if self.id == 'gneJ00':
        #     print('ID : {0} Action : {1}    phase : {2} time_on_phase/current_min_green  : {3}/{4}'
        #               .format(self.id, action, self.phase, self.time_on_phase, self.min_green[self.phase]))

    def _compute_edges(self):
        """
        return: Dict green phase to edge id
        """
        return {p : self.lanes[p*2:p*2+2] for p in range(self.num_green_phases)}  # two lanes per edge

    def _compute_edges_capacity(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return {
            p : sum([traci.lane.getLength(lane) for lane in self.edges[p]]) / vehicle_size_min_gap for p in range(self.num_green_phases)
        }

    # def get_density(self):
    #     return [sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for p in range(self.num_green_phases)]

    # def get_stopped_density(self):
    #     return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for p in range(self.num_green_phases)]

    def get_stopped_vehicles_num(self):
        return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) for p in range(self.num_green_phases)]

    def get_waiting_time(self):
        wait_time_per_road = []
        for p in range(self.num_green_phases):
            veh_list = self._get_veh_list(p)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_road.append(wait_time)
        return wait_time_per_road

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    @staticmethod
    def get_edge_id(lane):
        ''' Get edge Id from lane Id
        :param lane: id of the lane
        :return: the edge id of the lane
        '''
        return lane[:-2]
    
    def _get_veh_list(self, p):
        veh_list = []
        for lane in self.edges[p]:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @DeprecationWarning
    def keep(self):
        if self.time_on_phase >= self.max_green:
            self.change()
        else:
            self.time_on_phase += self.delta_time
            traci.trafficlight.setPhaseDuration(self.id, self.delta_time)

    @DeprecationWarning
    def change(self):
        if self.time_on_phase < self.min_green:  # min green time => do not change
            self.keep()
        else:
            self.time_on_phase = self.delta_time
            traci.trafficlight.setPhaseDuration(self.id, 0)
