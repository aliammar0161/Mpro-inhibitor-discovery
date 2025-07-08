import os
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit import Chem
from IPython.display import Image, SVG
from rdkit.Chem.Draw import rdMolDraw2D

def plot_umap(features, labels, title, save_path="output/umap_plot.png"):
    """Generates and saves a UMAP plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.5, style="whitegrid")
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in labels],
        alpha=0.6,
        s=20
    )
    plt.title(title, fontsize=24, pad=20)
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(save_path, dpi=300)
    print(f"UMAP plot saved to {save_path}")
    plt.close()

def mol_to_image(molecule, scaling=20, font_size=24):
    """Draw a molecule with the supplied RDKit molecule object."""
    d2d = Draw.MolDraw2DCairo(-1, -1)
    d2d.drawOptions().scalingFactor = scaling
    d2d.drawOptions().fixedFontSize = font_size
    d2d.DrawMolecule(molecule)
    d2d.FinishDrawing()
    return Image(d2d.GetDrawingText())