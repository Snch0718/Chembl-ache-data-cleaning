import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns


def safe_mol_from_smiles(smiles):
    """Safely convert SMILES string to RDKit molecule object."""
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def calculate_descriptors(mol):
    """Calculate basic molecular descriptors using RDKit."""
    if mol is None:
        return {}
    try:
        desc_dict = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'rings': Descriptors.RingCount(mol),
            'tpsa': Descriptors.TPSA(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'amide_bonds': Descriptors.NumAmideBonds(mol),
        }
        
        try:
            desc_dict['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
        except Exception:
            desc_dict['num_heteroatoms'] = 0
            
        try:
            desc_dict['num_aromatic_heterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
        except Exception:
            desc_dict['num_aromatic_heterocycles'] = 0
            
        try:
            desc_dict['num_aliphatic_heterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
        except Exception:
            desc_dict['num_aliphatic_heterocycles'] = 0
            
        return desc_dict
    except Exception as e:
        print(f"      Error calculating descriptors: {e}")
        return {}


def calculate_sa_score(mol):
    """Simplified synthetic accessibility metric."""
    if mol is None:
        return 10.0
    try:
        mw = Descriptors.MolWt(mol)
        rings = Descriptors.RingCount(mol)
        
        try:
            chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        except Exception:
            chiral_centers = 0
            
        try:
            ring_info = mol.GetRingInfo()
            complex_rings = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
        except Exception:
            complex_rings = 0
        
        sa_score = (mw / 500 * 2 + rings * 0.5 + chiral_centers * 1.5 + complex_rings * 2)
        return min(10.0, max(1.0, sa_score))
    except Exception:
        return 10.0


def generate_molecular_descriptors(df):
    """Generate molecular descriptors and SA scores for optimization."""
    print("   Converting SMILES to molecules...")
    df['molecule'] = df['canonical_smiles'].apply(safe_mol_from_smiles)
    
    successful_conversions = df['molecule'].notna().sum()
    if successful_conversions == 0:
        print("   WARNING: No molecules could be converted from SMILES!")
        return df
    
    initial_count = len(df)
    df = df[df['molecule'].notna()].copy()
    print(f"   Removed {initial_count - len(df)} failed conversions")
    
    print("   Calculating molecular descriptors...")
    descriptors_list = []
    successful_calcs = 0
    
    for idx, mol in enumerate(df['molecule']):
        if idx % 500 == 0:
            print(f"      Processed {idx}/{len(df)} molecules...")
        desc = calculate_descriptors(mol)
        if desc:
            successful_calcs += 1
        descriptors_list.append(desc)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    if descriptors_df.empty:
        print("   WARNING: No descriptors could be calculated!")
        return df
    
    print(f"   Successful descriptor calculations: {successful_calcs}/{len(df)}")
    
    df = pd.concat([df.reset_index(drop=True), descriptors_df.reset_index(drop=True)], axis=1)
    
    print("   Calculating synthetic accessibility scores...")
    df['sa_score'] = df['molecule'].apply(calculate_sa_score)
    
    print(f"   Generated {len(descriptors_df.columns)} molecular descriptors")
    print(f"   Available descriptors: {list(descriptors_df.columns)}")
    print(f"   SA score range: {df['sa_score'].min():.1f} - {df['sa_score'].max():.1f}")
    
    df = df.drop(columns=['molecule'])
    return df


def clean_chembl_dataset(df):
    """Clean and filter ChEMBL dataset for IC50 values."""
    original_size = len(df)
    df_clean = df.copy()
    
    # 0. Standardize column names (converts 'Standard Type' -> 'standard_type')
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('#', '')
    )

    # Alias common ChEMBL column variations
    column_aliases = {
        'smiles': 'canonical_smiles',
        'molecule_smiles': 'canonical_smiles',
        'standard_value_units': 'standard_units'
    }
    df_clean = df_clean.rename(columns=column_aliases)

    # 1. Remove empty columns
    empty_cols = df_clean.columns[df_clean.isna().all()].tolist()
    if empty_cols:
        print(f"\n1. Removing empty columns: {empty_cols}")
        df_clean = df_clean.drop(columns=empty_cols)
    
    # 2. Filter IC50
    print("\n2. Filtering for IC50 measurements...")
    ic50_before = len(df_clean)
    if 'standard_type' in df_clean.columns:
        df_clean = df_clean[df_clean['standard_type'] == 'IC50']
    print(f"   Kept {len(df_clean)}/{ic50_before} IC50 measurements")
    
    # 3. Remove missing critical data
    print("\n3. Removing rows with missing critical data...")
    if 'standard_value' in df_clean.columns:
        df_clean = df_clean[df_clean['standard_value'].notna()]
        print(f"   - After removing missing standard_value: {len(df_clean)}")
    
    if 'canonical_smiles' in df_clean.columns:
        df_clean = df_clean[df_clean['canonical_smiles'].notna()]
        print(f"   - After removing missing SMILES: {len(df_clean)}")
    else:
        raise KeyError("Could not find a valid SMILES column in the CSV dataset!")
    
    # 4. Validity & relation filters
    if 'data_validity_comment' in df_clean.columns:
        valid_before = len(df_clean)
        df_clean = df_clean[df_clean['data_validity_comment'].isna()]
        print(f"   - After validity filter: {len(df_clean)}/{valid_before}")
    
    if 'standard_relation' in df_clean.columns:
        relation_before = len(df_clean)
        # Strip potential surrounding quotes from ChEMBL relations (e.g. "'='")
        df_clean['standard_relation'] = df_clean['standard_relation'].astype(str).str.replace("'", "")
        df_clean = df_clean[df_clean['standard_relation'] == '=']
        print(f"   - After exact measurements filter: {len(df_clean)}/{relation_before}")
    
    # 5. Units validation
    print("\n5. Validating units and values...")
    if 'standard_units' in df_clean.columns:
        unit_counts = df_clean['standard_units'].value_counts()
        print(f"   Unit distribution: {unit_counts.to_dict()}")
        unit_before = len(df_clean)
        df_clean = df_clean[df_clean['standard_units'] == 'nM']
        print(f"   - After nM unit filter: {len(df_clean)}/{unit_before}")
    
    df_clean['standard_value'] = pd.to_numeric(df_clean['standard_value'], errors='coerce')
    value_before = len(df_clean)
    df_clean = df_clean[df_clean['standard_value'] > 0]
    print(f"   - After positive value filter: {len(df_clean)}/{value_before}")
    
    # 6. Deduplication
    print("\n6. Handling duplicate compounds...")
    duplicate_count = df_clean['canonical_smiles'].duplicated().sum()
    print(f"   Found {duplicate_count} duplicate SMILES")
    
    if duplicate_count > 0:
        aggregation_rules = {'standard_value': 'median'}
        
        if 'molecule_chembl_id' in df_clean.columns:
            aggregation_rules['molecule_chembl_id'] = 'first'
        if 'assay_chembl_id' in df_clean.columns:
            aggregation_rules['assay_chembl_id'] = 'count'
        if 'document_chembl_id' in df_clean.columns:
            aggregation_rules['document_chembl_id'] = 'nunique'
        if 'pchembl_value' in df_clean.columns:
            aggregation_rules['pchembl_value'] = 'median'
        
        df_clean = df_clean.groupby('canonical_smiles').agg(aggregation_rules).reset_index()
        
        rename_dict = {}
        if 'assay_chembl_id' in df_clean.columns:
            rename_dict['assay_chembl_id'] = 'num_assays'
        if 'document_chembl_id' in df_clean.columns:
            rename_dict['document_chembl_id'] = 'num_publications'
        if rename_dict:
            df_clean = df_clean.rename(columns=rename_dict)
    
    print(f"   After deduplication: {len(df_clean)} unique compounds")
    
    # 7. Calculate pIC50
    print("\n7. Calculating potency metrics...")
    df_clean['pIC50'] = -np.log10(df_clean['standard_value'] * 1e-9)
    df_clean['IC50_nM'] = df_clean['standard_value']
    
    print(f"   pIC50 range: {df_clean['pIC50'].min():.2f} - {df_clean['pIC50'].max():.2f}")
    print(f"   IC50 range: {df_clean['IC50_nM'].min():.2f} - {df_clean['IC50_nM'].max():.2f} nM")
    
    # 8. Molecular descriptors
    print("\n8. Generating molecular descriptors...")
    df_clean = generate_molecular_descriptors(df_clean)
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print(f"Original dataset: {original_size} rows")
    print(f"Cleaned dataset: {len(df_clean)} rows")
    print(f"Data retention: {len(df_clean)/original_size*100:.1f}%")
    
    return df_clean


def visualize_cleaned_data(df):
    """Generate plots for distribution, correlations, and synthetic accessibility."""
    print("\nGENERATING DATA VISUALIZATIONS")
    
    descriptor_cols = ['mol_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa', 'sa_score', 'rings']
    available_cols = [col for col in descriptor_cols if col in df.columns]
    
    if not available_cols:
        plt.figure(figsize=(10, 6))
        plt.hist(df['pIC50'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('pIC50')
        plt.ylabel('Frequency')
        plt.title('Distribution of pIC50 Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pIC50_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    n_plots = min(len(available_cols) + 1, 6)
    n_rows = (n_plots + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_plots == 1 else axes)
    
    axes[0].hist(df['pIC50'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('pIC50')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of pIC50 Values')
    axes[0].grid(True, alpha=0.3)
    
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i, col in enumerate(available_cols[:5]):
        ax_idx = i + 1
        if ax_idx < len(axes):
            axes[ax_idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color=colors[i % len(colors)])
            axes[ax_idx].set_xlabel(col.replace('_', ' ').title())
            axes[ax_idx].set_ylabel('Frequency')
            axes[ax_idx].set_title(f'Distribution of {col.replace("_", " ").title()}')
            axes[ax_idx].grid(True, alpha=0.3)
            
    for i in range(len(available_cols) + 1, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('cleaned_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(available_cols) >= 2:
        correlation_cols = ['pIC50'] + available_cols
        corr_matrix = df[correlation_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
        plt.title('Molecular Descriptor Correlations')
        plt.tight_layout()
        plt.savefig('descriptor_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    if 'sa_score' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['sa_score'], df['pIC50'], alpha=0.6, color='green')
        plt.xlabel('Synthetic Accessibility Score (lower = easier)')
        plt.ylabel('pIC50')
        plt.title('Potency vs Synthetic Accessibility')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('potency_vs_sa.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    input_path = "AChE_Bioactivity_data_1.csv"
    output_path = "cleaned_ACHE_data_optimization_ready.csv"
    
    if os.path.exists(input_path):
        df = pd.read_csv(input_path,sep=';')
        cleaned_df = clean_chembl_dataset(df)
        cleaned_df.to_csv(output_path, index=False)
        visualize_cleaned_data(cleaned_df)
        print(f"\nSaved cleaned dataset to: {output_path}")
    else:
        print(f"Dataset non-existent at path: {input_path}. Please place your CSV in the directory.")