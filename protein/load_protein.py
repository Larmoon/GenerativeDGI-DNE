import os.path

import pandas as pd,requests,json,pickle
from Bio import SeqIO


def __uniprot_entry(gene_names):
    lst = ''
    for i in gene_names:
        lst += i + ','
    url ="https://biodbnet-abcc.ncifcrf.gov/webServices/rest.php/biodbnetRestApi.json?method=db2db&input=ensemblgeneid&inputValues=%s&outputs=uniprotentryname"
    reply = json.loads(requests.get(url % lst,stream=True).text)
    ret = {}
    for i in range(len(gene_names)):
        if reply[str(i)]['outputs']:
            ret[reply[str(i)]['InputValue']] = reply[str(i)]['outputs']['UniProt Entry Name']
    return ret

def uniprot_entry(gene_names):
    ret = {}
    if os.path.exists(f"./protein/uniprot.pickle"):
        with open(f"./protein/uniprot.pickle",'rb') as f:
            ret = pickle.load(f)
    else:
        n = 400
        chunks = [gene_names[i:i + n] for i in range(0, len(gene_names), n)]
        for i in chunks:
            ret.update(__uniprot_entry(i))
            print(f"get uniprot entry:{len(ret)} proteins")
        print("uniprot entry finished")
        with open(f"./protein/uniprot.pickle", 'wb') as f:
            pickle.dump(ret,f)
    return ret

def get_sequence(uni):
    print("GETTING SEQUENCE")
    uniprot_list = uni
    fasta = SeqIO.parse(f"./protein/huprot.fasta", "fasta")
    ret = {}
    for i in fasta:
        for j in uniprot_list:
            if j in i.id:
                ret[j] = str(i.seq)
                uniprot_list.remove(j)
    ret_name = []
    ret_seq = []
    for i in ret:
        ret_name.append(i)
        ret_seq.append(ret[i])
    return ret_name,ret_seq

def replace(e2u,obj):
    ret = []
    for i in obj:
        try:
            ret.append(e2u[i][0].replace("_HUMAN",''))
        except BaseException:
            ret.append('')
    return ret

def load_protein(ppipath,sep='\t'):
    __node1 =[]
    __node2 = []
    __prot_list = []
    df = pd.read_csv(ppipath, sep=sep).values.tolist()[1:]
    # 打开CSV文件
    if os.path.exists(f"./protein/saved.pickle"):
        with open(f"./protein/saved.pickle", 'rb') as f:
            __node1, __node2, __prot_list = pickle.load(f)
    else:
        for row in df:
            __node1.append(row[0])
            __node2.append(row[1])
            if not row[0] in __prot_list: __prot_list.append(row[0])
            if not row[1] in __prot_list: __prot_list.append(row[1])
            leng = len(__prot_list)
            if leng % 100 == 0:
                print(f"detected protein:{leng}")
    with open(f"./protein/saved.pickle",'wb') as f:
        pickle.dump((__node1,__node2,__prot_list),f)
    e2u = uniprot_entry(__prot_list)
    node1 = replace(e2u,__node1)
    node2 = replace(e2u, __node2)
    prot_list = replace(e2u, __prot_list)
    node1f = []
    node2f = []
    for i in range(len(node1)):
        if node1[i] and node2[i]:
            node1f.append(node1[i])
            node2f.append(node2[i])
    data = {'node1':node1f,'node2':node2f}
    df = pd.DataFrame(data)
    df.to_csv(f"./protein/edges.csv", index=False)
    ret_name, ret_seq = get_sequence(prot_list)
    data = {'UniprotEntry': ret_name, 'seq': ret_seq}
    df = pd.DataFrame(data)
    df.to_csv(f"./protein/seq.csv", index=False)



