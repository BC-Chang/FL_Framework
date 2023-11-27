from utils import write_yaml
import pandas as pd

if __name__ == "__main__":
    train_df = pd.read_excel(r"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\train.xlsx", index_col=0)
    val_df = pd.read_excel(r"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\val.xlsx", index_col=0)
    test_df = pd.read_excel(r"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\test.xlsx", index_col=0)
    print(train_df['MSNet_Name'])
    write_yaml([train_df['MSNet_Name'], val_df['MSNet_Name']], ["train", "val"], "train_data.yml")
    write_yaml([test_df['MSNet_Name']], ["test"], "test_data.yml")