import pandas as pd


def main():
    data = pd.read_csv('expo_test.csv')
    print(data.head())
    print(data.columns)

    l1 = [1, 2, 3]
    l2 = [4, 5, 6]
    print(l1 + l2)
    raise ValueError('test')


if __name__ == '__main__':
    main()