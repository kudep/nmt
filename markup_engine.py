import re
import os

params=dict()
buf_path = "/tmp/nmt_chit-chat/buffers/"
inf_input_file = os.path.join(buf_path, 'inference_input_buf')
inf_output_file = os.path.join(buf_path, 'inference_output_buf')
context_file = os.path.join(buf_path, 'context_buf')

def mkprotecteddir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
def write_into_file(context,path, addition = ""):
    with open(path, "wt") as f:
        for line in context:
            f.write(line+addition )

def read_from_file(path):
    with open(path, "r") as pipe:
        line = pipe.read()
    return line


def preproc(line):
    line = re.sub(r'[\s+]', ' ', line)
    line = re.sub(r'(\\n)', ' ', line)
    line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)
    line = re.sub(r"([\*\"\'\\\/\|\{\}\[\]\;\:\<\>\,\.\?\*\(\)])", r" \1 ", line)
    line = re.sub(' +', ' ', line)
    return line
