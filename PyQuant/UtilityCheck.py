import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt


def ts_check():
    result = ts.get_hist_data('600000')
    # print(result)
    # print(type(result))
    print(result.columns)
    result['close'].plot( )

    plt.show()
    print(result.index)


def main():
    ts_check()


if __name__ == '__main__':
    main()
