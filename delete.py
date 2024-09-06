import subprocess

import sys
with open('requirements.txt') as f:
    packages = f.read().splitlines()

subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y'] + packages)