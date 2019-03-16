import os
import re

file_name_regexp = re.compile(r'data_(\w+)_([0-9]+)')
memory_usage_regexp = re.compile(r'_TFProfRoot \(--/(.*), --/(.*), --/(.*), --/(.*)\)')

files = os.listdir("./")
print("net_type | run_number | requested bytes | peak bytes | residual bytes | output bytes")
for file_path in files:
    matcher = file_name_regexp.match(file_path)
    if matcher is not None:
        net_type = matcher.group(1)
        run_number = matcher.group(2)
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 10:
                    mem_matcher = memory_usage_regexp.match(line)
                    if mem_matcher is None:
                        raise Exception("wrong regex")
                    requested = mem_matcher.group(1)
                    peak = mem_matcher.group(2)
                    residual = mem_matcher.group(3)
                    output = mem_matcher.group(4)
                    print(f"{net_type} {run_number} {requested} {peak} {residual} {output}")
                    break
