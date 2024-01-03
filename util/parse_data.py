import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", default = r"none", type = str, help="file of performance data")
args = parser.parse_args()
with open(args.file, "r") as f:
    for line in f.readlines():
        if "Avg" in line:
            latency = line.rstrip().split(":")[-1].split("us")[0]
            print("{:.2f}".format(int(latency)/1000))