from utils import write_yaml
import pandas as pd

if __name__ == "__main__":
    client = "c2"
    train_df = pd.read_excel(fr"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\train_{client}.xlsx", index_col=0)
    val_df = pd.read_excel(fr"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\val_{client}.xlsx", index_col=0)
    test_df = pd.read_excel(r"C:\Users\bchan\Box\Research\Conference_Stuff\AGU_2023\results\test.xlsx", index_col=0)

    write_yaml([train_df['MSNet_Name'], val_df['MSNet_Name']], ["train", "val"], f"{client}_train_data.yml")
    # write_yaml([test_df['MSNet_Name']], ["test"], "test_data.yml")