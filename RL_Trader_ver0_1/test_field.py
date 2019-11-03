# import os
# import sqlite3
# import pandas as pd
# import numpy as np

# os.chdir("/Users/unanimous0")

# def load_chart_data(stock_code):

#     # Connecting database 
#     conn = sqlite3.connect("kospi.db")
#     cur = conn.cursor()

#     # Executing SQL Query
#     stock_code_KS = stock_code + ".KS"
#     cur.execute("SELECT * FROM '{}' LIMIT 10;".format(stock_code_KS))

#     # Getting data
#     rows = cur.fetchall()
#     # Checking the data 
#     # for row in rows:
#     #     print(row)

#     # Processing the data table
#     header = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#     chart_data = pd.DataFrame(rows, columns=header)
#     date = chart_data['Date']
#     date_list = list(date)
#     del chart_data['Date']
#     chart_data.index = date_list

#     # Sorting in reverse order by date
#     chart_data = chart_data[::-1]

#     print(type(chart_data))

#     conn.close()

#     return chart_data


import os
import settings
import logging

stock_code = '000020'
log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
timestr = settings.get_time_str()
if not os.path.exists('logs/%s' % stock_code):
    os.makedirs('logs/%s' % stock_code)
file_handler = logging.FileHandler(filename=os.path.join(log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)