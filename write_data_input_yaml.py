from utils import write_yaml
import pandas as pd

if __name__ == "__main__":
    client = "c2"
    train_df = pd.read_excel(fr"/Users/bchang/Library/CloudStorage/Box-Box/Research/Conference_Stuff/AGU_2023/results/train_msnet.xlsx")#, index_col=0)
    val_df = pd.read_excel(fr"/Users/bchang/Library/CloudStorage/Box-Box/Research/Conference_Stuff/AGU_2023/results/val_msnet.xlsx")#, index_col=0)
    test_df = pd.read_excel(r"/Users/bchang/Library/CloudStorage/Box-Box/Research/Conference_Stuff/AGU_2023/results/test_msnet.xlsx")#, index_col=0)

    write_yaml([train_df['MSNet_Name'], val_df['MSNet_Name']], ["train", "val"], f"train_val_msnet.yml")
    write_yaml([test_df['MSNet_Name']], ["test"], "test_msnet.yml")