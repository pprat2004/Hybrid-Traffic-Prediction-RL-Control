import shutil
import os

# Copy and rename files
files_to_copy = [
    ('net_net.xml', 'net.net.xml'),
    ('input_routes_rou.xml', 'input_routes.rou.xml'),
    ('upgraded_cross3ltl.sumocfg', 'cross3ltl.sumocfg')
]

for src, dst in files_to_copy:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"✓ Copied {src} → {dst}")
    elif os.path.exists(dst):
        print(f"✓ {dst} already exists")
    else:
        print(f"⚠ {src} not found")

# Create cross3ltl.sumocfg if it doesn't exist
if not os.path.exists('cross3ltl.sumocfg'):
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="net.net.xml"/>
        <route-files value="input_routes.rou.xml"/>
    </input>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <xml-validation value="never"/>
        <duration-log.disable value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
'''
    with open('cross3ltl.sumocfg', 'w') as f:
        f.write(config_content)
    print("✓ Created cross3ltl.sumocfg")

print("\nSetup complete! Now run:")
print("python main.py --mode train --episodes 1")