import deepchem as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def get_morgan_fingerprints(smiles_list, radius=3, n_bits=2048):
    """Generates Morgan fingerprints for a list of SMILES."""
    featuriser = dc.feat.CircularFingerprint(radius=radius, size=n_bits)
    return featuriser.featurize(smiles_list)

def tanimoto_similarity_search(smiles_list, target_smiles, radius=2):
    """Performs a Tanimoto similarity search against a target molecule."""
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    target_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(target_smiles))
    
    similarities = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = fpgen.GetFingerprint(mol)
            similarities.append(DataStructs.TanimotoSimilarity(target_fp, fp))
        else:
            similarities.append(0.0) # Append 0 if SMILES is invalid
            
    return similarities