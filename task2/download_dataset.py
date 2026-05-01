import pandas as pd
import os

# NHANES 2013-2014 prefix is _H
# URLs for datasets:
urls = {
    'DEMO': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT',
    'BMX': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BMX_H.XPT',
    'BPQ': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPQ_H.XPT',
    'GLU': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/GLU_H.XPT',
    'COT': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/COT_H.XPT',
    'BIOPRO': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BIOPRO_H.XPT',
    'DPQ': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DPQ_H.XPT',
    'DIQ': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DIQ_H.XPT'
}

def download_data(output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    df_merged = None
    
    for name, url in urls.items():
        print(f"Downloading {name} from {url}...")
        try:
            df = pd.read_sas(url)
            # SEQN is the respondent sequence number
            if 'SEQN' in df.columns:
                df.set_index('SEQN', inplace=True)
            print(f"Loaded {name} with shape {df.shape}")
            
            if df_merged is None:
                df_merged = df
            else:
                df_merged = df_merged.join(df, how='outer')
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            
    print(f"Merged dataset shape: {df_merged.shape}")
    csv_path = os.path.join(output_dir, 'nhanes_2013_2014_merged.csv')
    df_merged.to_csv(csv_path)
    print(f"Saved merged dataset to {csv_path}")

if __name__ == "__main__":
    download_data()
