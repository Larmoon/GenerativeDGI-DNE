---
license: gpl-3.0
datasets:
- alimotahharynia/approved_drug_target
language:
- en
base_model:
- openai-community/gpt2
- liyuesen/druggpt
pipeline_tag: text-generation
library_name: transformers
tags:
- chemistry
- biology
- medical
---
# DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback

DrugGen is a GPT-2 based model specialized for generating drug-like SMILES structures based on protein sequence. The model leverages the characteristics of approved drug targets and has been trained through both supervised fine-tuning and reinforcement learning techniques to enhance its ability to generate chemically valid, safe, and effective structures.

## Model Details

-  Model Name: DrugGen
-  Training Paradigm: Supervised Fine-Tuning (SFT) + Proximal Policy Optimization (PPO)
-  Input: Protein Sequence
-  Output: SMILES Structure
-  Training Libraries: Hugging Face’s transformers and Transformer Reinforcement Learning (TRL)
-  Model Sources: liyuesen/druggpt

## How to Get Started with the Model
-  DrugGen can be used via command-line interface (CLI) or integration into Python scripts.

### Installation
#### Clone the repository and navigate to its directory
```bash
git clone https://github.com/mahsasheikh/DrugGen.git
cd DrugGen
```

#### Install dependencies
```bash
pip3 install -r requirements.txt
```

### Command-Line Interface
DrugGen provides a CLI to generate SMILES structures based on UniProt IDs, protein sequences, or both.

#### Generating SMILES Structures
```bash
python3 drugGen_generator_cli.py --uniprot_ids <UniProt_IDs> --sequences <Protein_Sequences> --num_generated <Number_of_Structures> --output_file <Output_File_Name>
```

#### Example Command
```bash
python3 drugGen_generator_cli.py --uniprot_ids P12821 P37231 --sequences "MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS" --num_generated 10 --output_file g_smiles_test.txt
```
#### Parameters
-  uniprot_ids: Space-separated UniProt IDs.
-  sequences: Space-seperated protein sequences in string format.
-  num_generated: Number of unique SMILES structures to generate.
-  output_file: Name of the output file to save the generated structures.
  

### Python Integration
```python
# Example call for inference using only sequences
from drugGen_generator import run_inference
run_inference(
    sequences=[ "MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS"],
    num_generated=10,
    output_file="output_SMILES.txt"
)

# Example call for inference using only UniProt IDs
from drugGen_generator import run_inference
run_inference(
    uniprot_ids=["P12821", "P37231"],
    num_generated=10,
    output_file="output_SMILES.txt"
)

# Example call for inference using both UniProt IDs and sequences
run_inference(
    sequences=["MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS"], 
    uniprot_ids=["P12821", "P37231"], 
    num_generated=10, 
    output_file="output_SMILES.txt"
)
```

## Training Details
### Training Data
[alimotahharynia/approved_drug_target](https://huggingface.co/datasets/alimotahharynia/approved_drug_target)
- This dataset contains approved SMILES-protein sequences pairs data. It was used to train the model for generating SMILES strings.

### Training Procedure
- **Training regime:** fp32

#### Supervised Fine-Tuning
DrugGen was initially trained using supervised fine-tuning on a curated dataset of approved drug targets.
- **Training: validation sets** (ratio of 8:2) 
- **sft_config**
  - `num_train_epochs= 5`
  - `per_device_train_batch_size= 8`
  - `per_device_eval_batch_size= 8`
  - `evaluation_strategy="steps"`
  - `save_strategy="epoch"`
  - `eval_steps=50`
  - `logging_steps=25`
  - `logging_strategy="steps"`
  - `do_eval=True`
  - `do_train=True`
  - `learning_rate=5e-4`
  - `adam_epsilon=1e-08`
  - `warmup_steps=100`
  - `eval steps=50`
  - `dataloader_drop_last=True`
  - `save_safetensors=False`
  - `max_seq_length=768`

- **AdamW optimizer**
  - `lr=5e-4`
  - `eps=1e-08`

- **scheduler**
  - get_linear_schedule_with_warmup
#### Proximal Policy Optimization

- **Rollout:** Generates a response based on an input query. Generation parameters include:

  - `do_sample=True`
  - `top_k=9`
  - `max_length=1024`
  - `top_p=0.9`
  - `bos_token_id=tokenizer.bos_token_id`
  - `eos_token_id=tokenizer.eos_token_id`
  - `pad_token_id=tokenizer.pad_token_id`
  - `num_return_sequences=10`

In each epoch, generation continued until 30 unique small molecules were generated for each target.

- **Evaluation:** A reward function include:

  - Binding affinity predictor: "Protein-Ligand Binding Affinity Prediction Using Pretrained Transformers was (PLAPT)"
  - Customized invalid structure assessor: Based on RDKit library
  - A multiplicative penalty of "0.7" when a generated SMILES matched a molecule present in the approved SMILES dataset.

- **Optimization:**

- **ppo_config**
  -  `mini_batch_size=8`
  -  `batch_size=240`
  -  `learning_rate=1.41e-5`
  -  `use_score_scaling=True`
  -  `use_score_norm=True`

Prompts with a tensor size greater than 768 were omitted, resulting in 2053 sequences (98.09% of the initial dataset).

## Citation
If you use this model in your research, please cite our paper:
```
@misc{sheikholeslami2024druggenadvancingdrugdiscovery,
      title={DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback}, 
      author={Mahsa Sheikholeslami and Navid Mazrouei and Yousof Gheisari and Afshin Fasihi and Matin Irajpour and Ali Motahharynia},
      year={2024},
      eprint={2411.14157},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2411.14157}, 
}
```