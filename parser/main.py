from parser import Parser
import pandas as pd


def main():
    pars = Parser()
    table_path = "pars_data.csv"

    pars.value_count(table_path)
    # pars_data = pd.DataFrame(tuple(a for a in pars.parse_from_table(table_path, show=False, return_data=True)), columns=['text', 'target'])
    # pars_data.to_csv("pars_data.csv", index=False, sep=';')


if __name__ == '__main__':
    main()
