import os, subprocess

print("Running custom_startup.py...")

# If apt-get is needed, it likely won't work in the default Azure container
# But if you want to try, you can do:
# subprocess.run(["apt-get", "update"])
# subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"])

# Remove /agents/python from PYTHONPATH
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace("/agents/python:", "")

print("PYTHONPATH is now:", os.environ["PYTHONPATH"])

# Finally, run your waitress app
subprocess.run(["python", "startup.py"])
