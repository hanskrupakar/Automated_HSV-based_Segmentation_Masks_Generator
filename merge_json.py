import simplejson as json
import glob

main_dict = {}

read_files = glob.glob("*.json")
with open("merged_file.json", "wb") as outfile:
    for f in read_files:
        with open(f, 'r') as infile:
            print infile.read()[0]
            f_dict = json.loads(infile.read())
            #print f_dict
