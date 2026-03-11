# </
# />
# Helper python script that runs a generic .auto command file

import os
import sys
import json

cmdfile = "restart.auto"


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!UPDATE THESE WITH THE CORRECT PATHS IN YOUR SYSTEM!!!
auto_dir = "C:\\MinGW\\msys\\1.0\\home\\sanch\\auto\\07p\\python"
eq_dir = __import__("os").environ.get("AUTO_EQ_DIR", "C:\\Users\\sanch\\Desktop\\Elastic model")
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


os.chdir(auto_dir)
if auto_dir not in sys.path:
    sys.path.append(auto_dir)
    
# Open AUTO CLUI
#execfile(auto_dir + "\\auto.py")

# Run the auto commands script
import auto
os.chdir(eq_dir)
# Pre-write failure state - restart.auto will overwrite if it succeeds
with open("auto_status.json", "w") as f:
    json.dump({"converged": False, "reason": "AUTO did not complete"}, f)
auto.auto(cmdfile)
