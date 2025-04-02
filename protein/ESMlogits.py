import pandas as pd
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, LogitsConfig
import torch,os,pickle
from Bio.PDB import PDBParser,NeighborSearch
import networkx as nx
import torch
from Bio import SeqIO
class prot_logits:
    def __init__(self):
        self.TEST_MODE = True
        os.environ['INFRA_PROVIDER'] = '叫什么无所谓，他只是判断这个环境变量是不是存在'
        self.protein = ESMProtein()
        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-sm-open-v1").cuda()



    def __get_logits(self,protname,seq):
        seq = str(seq)
        if len(seq) > 1024: return
        if os.path.exists(f"./protein/graph/{protname[0]}/{protname}.pickle") or len(seq) > 1024: return
        self.protein.sequence = seq
        with torch.no_grad():
            protein_1 = self.model.generate(self.protein, GenerationConfig(track="structure", num_steps=8))
            protein_1.to_pdb(f"./protein/tmp.pdb")
            logits_output = self.model.logits(self.model.encode(self.protein),
                                              LogitsConfig(sequence=True, return_embeddings=True, ))
            embeddings = logits_output.embeddings
            length = embeddings.shape[1]
            embeddings = embeddings[:,1:length-1]
            with open(f"./protein/logits/{protname[0]}/{protname}.pickle", 'wb') as f:
                pickle.dump(embeddings, f)
        p = PDBParser()
        s = p.get_structure(protname,f"./protein/tmp.pdb")[0]['A']
        tmp = []
        for i in s:
            tmp.append(i['CA'])
        ns = NeighborSearch(tmp)
        ns = ns.search_all(8)
        edge_index = []
        for i in ns:
            edge_index.append((tmp.index(i[0]),tmp.index(i[1])))
        row = []
        col = []
        for i in edge_index:
            row.append(i[0])
            col.append(i[1])
        row = torch.tensor(row)
        col = torch.tensor(col)
        edge_index = torch.stack([row, col], dim=0)
        with open(f"./protein/graph/{protname[0]}/{protname}.pickle",'wb') as f:
            pickle.dump(edge_index,f)
        print(f'{f} data collected')

    def get_logits(self):
        df = pd.read_csv(f'./protein/seq.csv', sep=',').values.tolist()[1:]
        for i in df:
            try:
                self.__get_logits(i[0],i[1])
            except BaseException:
                0