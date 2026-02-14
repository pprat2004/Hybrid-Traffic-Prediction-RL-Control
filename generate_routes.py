"""
Enhanced Route Generation with Emergency Vehicles
Author: Enhanced by Claude for Emergency-Aware System
"""

import random
import numpy as np
from datetime import datetime


class EnhancedRouteGenerator:
    def __init__(self, emergency_rate=0.05, rush_hour=True):
        """
        Args:
            emergency_rate: Probability of emergency vehicle (0.05 = 5%)
            rush_hour: Enable rush hour traffic patterns
        """
        self.emergency_rate = emergency_rate
        self.rush_hour = rush_hour
        
    def generate_routes(self, output_file="input_routes.rou.xml", duration=3600):
        """Generate routes with normal and emergency vehicles"""
        random.seed(int(datetime.now().timestamp()))
        
        # Base demand rates (vehicles per second)
        base_pH = 1. / 7   # Horizontal
        base_pV = 1. / 11  # Vertical
        base_pAR = 1. / 30 # Always right
        base_pAL = 1. / 25 # Always left
        
        with open(output_file, "w") as routes:
            # Write header
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Normal Vehicle Types -->
    <vType id="normal_car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="14" color="1,1,0"/>
    <vType id="normal_truck" accel="0.6" decel="4.0" sigma="0.3" length="7.5" minGap="3" maxSpeed="11" color="0.5,0.5,0.5"/>
    <vType id="normal_bus" accel="0.7" decel="4.2" sigma="0.4" length="12" minGap="3" maxSpeed="12" color="0,0.5,1"/>
    
    <!-- Emergency Vehicle Types -->
    <vType id="ambulance" accel="2.0" decel="6.0" sigma="0" length="6" minGap="1.5" maxSpeed="20" color="1,0,0" speedFactor="1.3" speedDev="0"/>
    <vType id="firetruck" accel="1.5" decel="5.5" sigma="0" length="8" minGap="2" maxSpeed="18" color="1,0.3,0" speedFactor="1.2" speedDev="0"/>
    <vType id="police" accel="2.2" decel="6.5" sigma="0" length="5" minGap="1.5" maxSpeed="22" color="0,0,1" speedFactor="1.4" speedDev="0"/>
    
    <!-- Routes -->
    <route id="horizontal" edges="2fi 2si 1o 1fi 1si 2o 2fi"/>
    <route id="vertical" edges="3fi 3si 4o 4fi 4si 3o 3fi"/>
    <route id="always_right" edges="1fi 1si 4o 4fi 4si 2o 2fi 2si 3o 3fi 3si 1o 1fi"/>
    <route id="always_left" edges="3fi 3si 2o 2fi 2si 4o 4fi 4si 1o 1fi 1si 3o 3fi"/>
    <route id="north_south" edges="3fi 3si 4o"/>
    <route id="south_north" edges="4fi 4si 3o"/>
    <route id="east_west" edges="1fi 1si 2o"/>
    <route id="west_east" edges="2fi 2si 1o"/>

''', file=routes)
            
            vehNr = 0
            emg_count = 0
            
            for i in range(duration):
                # Rush hour dynamics (7-9 AM, 5-7 PM equivalent)
                time_factor = 1.0
                if self.rush_hour:
                    hour = (i / 3600) * 24
                    if 7 <= hour < 9 or 17 <= hour < 19:
                        time_factor = 2.5  # 2.5x more traffic
                    elif 9 <= hour < 17:
                        time_factor = 1.3
                    else:
                        time_factor = 0.7
                
                # Adjust probabilities based on time
                pH = base_pH * time_factor
                pV = base_pV * time_factor
                pAR = base_pAR * time_factor
                pAL = base_pAL * time_factor
                
                # Generate emergency vehicle?
                is_emergency = random.uniform(0, 1) < self.emergency_rate
                
                if is_emergency:
                    emg_type = random.choice(['ambulance', 'firetruck', 'police'])
                    route = random.choice(['horizontal', 'vertical', 'north_south', 
                                         'south_north', 'east_west', 'west_east'])
                    print(f'    <vehicle id="emergency_{emg_count}" type="{emg_type}" '
                          f'route="{route}" depart="{i}" departSpeed="max"/>', 
                          file=routes)
                    emg_count += 1
                    vehNr += 1
                    continue
                
                # Normal traffic generation
                vehicle_type = random.choices(
                    ['normal_car', 'normal_truck', 'normal_bus'],
                    weights=[0.75, 0.15, 0.10]
                )[0]
                
                if random.uniform(0, 1) < pH:
                    print(f'    <vehicle id="horizontal_{vehNr}" type="{vehicle_type}" '
                          f'route="horizontal" depart="{i}"/>', file=routes)
                    vehNr += 1
                    
                if random.uniform(0, 1) < pV:
                    print(f'    <vehicle id="vertical_{vehNr}" type="{vehicle_type}" '
                          f'route="vertical" depart="{i}"/>', file=routes)
                    vehNr += 1
                    
                if random.uniform(0, 1) < pAL:
                    print(f'    <vehicle id="left_{vehNr}" type="{vehicle_type}" '
                          f'route="always_left" depart="{i}"/>', file=routes)
                    vehNr += 1
                    
                if random.uniform(0, 1) < pAR:
                    print(f'    <vehicle id="right_{vehNr}" type="{vehicle_type}" '
                          f'route="always_right" depart="{i}"/>', file=routes)
                    vehNr += 1
            
            print("</routes>", file=routes)
            
        print(f"Route generation complete:")
        print(f"  Total vehicles: {vehNr}")
        print(f"  Emergency vehicles: {emg_count}")
        print(f"  Emergency rate: {emg_count/vehNr*100:.2f}%")
        
        return vehNr, emg_count


if __name__ == "__main__":
    # Test route generation
    generator = EnhancedRouteGenerator(emergency_rate=0.05, rush_hour=True)
    generator.generate_routes("input_routes.rou.xml", duration=3600)
