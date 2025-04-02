import os.path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import rdkit.Chem.rdFingerprintGenerator as rf
from decimal import Decimal
import torch,pickle
from drugGPT.generate import GPT
import numpy as np
import requests
GPT_INDEX = 0
def fingerprint(__drugname,__SMILE = None):
    drugname = __drugname.replace("DRUG_",'')
    SMILE=''
    if os.path.exists(f'./drug/fingerprint/{__drugname}.pickle'):return
    if os.path.exists(f'./drug/fingerprint_gpt/{__drugname[-1]}/{__drugname}.pickle'): return
    if __SMILE == None:
        torch.set_printoptions(profile="full")
        base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES,title/CSV'
        headers = {"content-type": "application/x-www-form-urlencoded"}
        tmp = drugname.split(':')
        if len(tmp) > 1:
            name = drugname.split(':')[1]
        else:
            name = drugname
        url = base_url.format(name)
        try:
            SMILE = requests.get(url, headers=headers).text.split('\n')[1].split(',')[1]
            SMILE = SMILE.replace("\"", "")
        except BaseException:
            return
    else:
        SMILE = __SMILE
    molecule = Chem.MolFromSmiles(SMILE)
    mfpgen = rf.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprint = torch.tensor(mfpgen.GetFingerprintAsNumPy(molecule))
    if __SMILE == None:
        with open(f'./drug/fingerprint/{__drugname}.pickle', 'wb') as f:
            pickle.dump(fingerprint,f)
    else:
        with open(f'./drug/fingerprint_gpt/{__drugname[-1]}/{__drugname}.pickle', 'wb') as f:
            pickle.dump(fingerprint,f)
    return fingerprint

class load_drugs:
    def __init__(self):
        self.GPTindex = GPT_INDEX
        self.prot2drug = {}

    def load_drugs(self,GPTonly = False):
        prot = []
        c1 = []
        c2 = []
        seqdf = pd.read_csv(f'./protein/seq.csv', sep=',').values.tolist()[1:]
        seq = {}
        for i in seqdf:
            seq[i[0]] = i[1]
            if i[0] == '': continue
            prot.append(i[0])
        df = pd.read_csv(f'./DGIdatabase.csv', sep=',').values.tolist()[1:]
        for i in df:
            if i[0] in prot:
                c1.append(i[0])
                c2.append("DRUG_" + i[2])
                if i[0] in self.prot2drug:
                    self.prot2drug[i[0]].append("DRUG_" + i[2])
                else:
                    self.prot2drug[i[0]] = ["DRUG_" + i[2]]
                if not GPTonly:
                    fingerprint("DRUG_"+i[2])
        for i in prot:
            try:
                if not i in self.prot2drug:
                    c1.append(i)
                    c2.append("DRUG_GPT" + str(self.GPTindex))
                    self.prot2drug[i] = ['DRUG_GPT' + str(self.GPTindex)]
                    self.GPTindex += 1
                    if self.GPTindex % 10 == 0:
                        print(f"GPTINDEX:{self.GPTindex}")
                    if self.GPTindex % 100 == 0:
                        data = {'target': c1, 'drug': c2}
                        df = pd.DataFrame(data)
                        df.to_csv(f"./drug/edges.csv", mode='a',header=False,index=False)
                        print('SAVED!')
                    fingerprint('DRUG_GPT' + str(self.GPTindex), GPT(seq[i]))
            except BaseException:
                continue
