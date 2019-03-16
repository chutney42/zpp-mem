import os
import re

file_name_regexp = re.compile(r'data_(\w+)_([0-9]+)')
memory_usage_regexp = re.compile(r'_TFProfRoot \(--/(.*), --/(.*), --/(.*), --/(.*)\)')
labels = "requested bytes | peak bytes | residual bytes | output bytes\n"

files = os.listdir("./")
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
                    out_str = f"{requested} {peak} {residual} {output}\n"
                    with open(f"../network_metadata/{file_path}", 'a') as meta:
                        meta.write(labels)
                        meta.write(out_str)
                    break
