from pathlib import Path
import pandas as pd

METADATA_DIR = Path('data/metadata')
DATASET_DIR = Path('data/hands')
OUTPUT_DIR = Path('output')
N_MANI = 30
FOTO_PER_SUPERFICIE = 2
SEED = 2026


def main():
    csv_files = sorted(METADATA_DIR.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError('Nessun CSV in data/metadata')
    df = pd.read_csv(csv_files[0]).copy()
    required = {'id', 'aspectOfHand', 'imageName'}
    if not required.issubset(df.columns):
        raise ValueError(f'Colonne mancanti: {required - set(df.columns)}')

    files_present = {p.name for p in DATASET_DIR.rglob('*.jpg')}
    df = df[df['imageName'].isin(files_present)].copy()
    df['surface'] = df['aspectOfHand'].astype(str).str.split().str[0]
    df['side'] = df['aspectOfHand'].astype(str).str.split().str[-1]
    df = df[df['surface'].isin(['palmar', 'dorsal']) & df['side'].isin(['left', 'right'])]
    df = df.drop_duplicates('imageName')

    counts = (df.groupby(['id', 'side', 'surface'])['imageName']
              .nunique().unstack('surface', fill_value=0))
    for surface in ('palmar', 'dorsal'):
        if surface not in counts.columns:
            counts[surface] = 0
    valid = counts[(counts['palmar'] >= FOTO_PER_SUPERFICIE) &
                   (counts['dorsal'] >= FOTO_PER_SUPERFICIE)]
    if len(valid) < 2:
        raise ValueError(f'Solo {len(valid)} mani idonee: campione insufficiente')

    chosen = valid.sample(n=min(N_MANI, len(valid)), random_state=SEED).index
    pieces = []
    for subject_id, side in chosen:
        for surface in ('palmar', 'dorsal'):
            subset = df[(df['id'] == subject_id) &
                        (df['side'] == side) &
                        (df['surface'] == surface)]
            pieces.append(subset.sample(n=FOTO_PER_SUPERFICIE,
                                        random_state=SEED))
    selected = pd.concat(pieces, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'campione_geometria.csv'
    selected[['imageName', 'id', 'aspectOfHand']].to_csv(path, index=False)
    print(f'Mani idonee disponibili: {len(valid)}')
    print(f'Mani selezionate: {len(chosen)}')
    print(f'Fotografie selezionate: {len(selected)}')
    print(f'File salvato: {path}')


if __name__ == '__main__':
    main()