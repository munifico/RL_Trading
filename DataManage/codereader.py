import urllib.parse
import pandas as pd
from pandas_datareader import data

MARKET_CODE_DICT = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt',
    'konex': 'konexMkt'
}

DOWNLOAD_URL = 'kind.krx.co.kr/corpgeneral/corpList.do'

def download_stock_codes(market=None, delisted=False):
    params = {'method': 'download'}

    if market.lower() in MARKET_CODE_DICT:
        params['marketType'] = MARKET_CODE_DICT[market]

    if not delisted:
        params['searchType'] = 13

    params_string = urllib.parse.urlencode(params)
    request_url = urllib.parse.urlunsplit(['http', DOWNLOAD_URL, '', params_string, ''])

    df = pd.read_html(request_url, header=0)[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format)

    return df

def read_kosdaq_code():
    kosdaq_stocks = download_stock_codes('kosdaq')
    kosdaq_codes=[]
    for code in kosdaq_stocks.종목코드:
        kosdaq_codes.append(code)
    return kosdaq_codes

def read_kospi_code():
    kospi_stocks = download_stock_codes('kospi')
    kospi_codes=[]
    for code in kospi_stocks.종목코드:
        kospi_codes.append(code)
    return kospi_codes

'''
if __name__=="__main__":
    kosdaq_codes = read_kosdaq_code()
    print(kosdaq_codes)
    kospi_codes = read_kospi_code()
    print(kospi_codes)
'''

