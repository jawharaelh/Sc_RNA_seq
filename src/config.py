"""
Shared configuration for ember and optimized polygon pipeline.
"""
import os

# Paths
DATA_FILE  = '/home/igvf-analysis/projects/8cube/processed_data/uci_submitted/annotated/Kidney_annotated.h5ad'
OUTPUT_DIR = 'output'

# obs column names
CATEGORY_COL  = 'Genotype'      # 8 strains, matches the reference octagon plot
SAMPLE_ID_COL = 'lab_sample_id' # 64 unique mouse samples, 8 per strain (4F+4M)
CONDITION_COL = 'Sex'           # 'Female' / 'Male', balanced within each strain

# CATEGORY_COL is partition label passed to ember.
PARTITION_LABEL = CATEGORY_COL

# Strain categories
# Order only affects color assignment / initial display
CATEGORIES = ['129S1J', 'AJ', 'B6J', 'CASTJ', 'NODJ', 'NZOJ', 'PWKJ', 'WSBJ']

CATEGORY_COLORS = {
    '129S1J': '#E41A1C',
    'AJ':     '#377EB8',
    'B6J':    '#4DAF4A',
    'CASTJ':  '#984EA3',
    'NODJ':   '#FF7F00',
    'NZOJ':   '#A65628',
    'PWKJ':   '#F781BF',
    'WSBJ':   '#999999',
}

# ember run parameters
NUM_DRAWS         = 100   # balanced bootstrap draws for Psi / Psi_block
N_PVAL_ITERATIONS = 1000  # permutations for empirical p-values
N_CPUS            = 2

# Gene significance filter
P_THRESH   = 0.05
Q_THRESH   = 0.05
PSI_THRESH = 0.6   # minimum overall Psi to call a gene "specific" at all

# Highlight-gene selection for the polygon plot
N_SINGLE   = 10    # top genes per strain (single-strain specific)
N_PER_PAIR = 2     # top genes per strain pair (multi-strain specific)
MIN_PSI    = 0.65  # min normalized Psi_block score to qualify as single-strain specific
