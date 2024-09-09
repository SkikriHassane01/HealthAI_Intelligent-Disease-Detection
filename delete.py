import subprocess
import sys
import chardet

with open('requirements.txt', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

try:
    cleaned_content = raw_data.decode(encoding).replace('\x00', '')
except UnicodeDecodeError as e:
    print(f"Error decoding with {encoding}: {e}")
    sys.exit(1)

packages = cleaned_content.splitlines()

subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y'] + packages)
