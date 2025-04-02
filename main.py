import pickle,shutil
import pandas as pd
from drugGPT.generate import GPT
import os,string
STRINGDB_PATH = f'HuRI.tsv'
TEST = True
const2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
const1 = [f'./protein/graph/',f'./protein/logits/',f'./drug/fingerprint/',f'./drug/fingerprint_gpt/']
def create_path():
    for i in const1:
        for j in const2:
            path = i+j
            if not os.path.exists(path):
                os.makedirs(path,exist_ok=True)

def zip_file():
    for i in const1:
        for j in const2:
            data = {}
            path = i + j
            if os.path.exists(path):
                filepath = path + ".pickle"
                lst = os.listdir(path)
                for k in lst:
                    pic = path + "/" + k
                    with open(pic, "rb") as f:
                        tmp = pickle.load(f)
                        name = k.split('.')[0]
                        data[name] = tmp
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
                shutil.rmtree(path)


from protein.load_protein import load_protein
from drug.load_drugs import load_drugs
from protein.ESMlogits import prot_logits
def main():
    create_path()
    load_protein(STRINGDB_PATH)
    logits = prot_logits()
    logits.get_logits()
    load = load_drugs()
    load.load_drugs()
    zip_file()

if __name__ == "__main__":
    if not TEST:main()