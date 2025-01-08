import json
import requests

def get_file_size(url):
    response = requests.head(url, allow_redirects=True)
    if 'Content-Length' not in response.headers:
        raise Exception(f"Unable to find size of {url}")
    return int(response.headers['Content-Length'])


fname="src/main/resources/availableDLVersions.json"

with open(fname, 'r') as ff:
	engines = json.load(ff)

jar_dict = {}

for engine in engines["versions"]:
	for jar in engine["jars"]:
		if jar in jar_dict.keys:
			continue
		jar_dict[jar] = get_file_size(jar)
