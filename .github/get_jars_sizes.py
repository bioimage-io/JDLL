import json
import requests
import argparse

def get_file_size(url):
    response = requests.head(url, allow_redirects=True)
    if 'Content-Length' not in response.headers:
        raise Exception(f"Unable to find size of {url}")
    return int(response.headers['Content-Length'])
    
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Get file sizes and save them to a target file.")
parser.add_argument('target_file', type=str, help="Path to the target file where results will be saved.")
args = parser.parse_args()

# Assign the target file from the command-line argument
target_file = args.target_file


fname="src/main/resources/availableDLVersions.json"

with open(fname, 'r') as ff:
	engines = json.load(ff)

jar_dict = {}

for engine in engines["versions"]:
	for jar in engine["jars"]:
		if jar in jar_dict.keys:
			continue
		jar_dict[jar] = get_file_size(jar)
		
		
with open(target_file, "w") as ff:
	json.dump(jar_dict)
