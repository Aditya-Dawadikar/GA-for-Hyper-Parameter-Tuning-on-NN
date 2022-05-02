import pandas as pd
import calendar
import time

class Export:
    def __init__(self):
        self.base_path = 'results/'
        pass

    def createDataFrame(self, data):
        df = pd.DataFrame(data, columns=[
                          'Generation', 'Model Name', 'Layer Description', 'max iterations', 'activation', 'accuracy'])
        
        filename = self.base_path + str(calendar.timegm(time.gmtime())) + '.csv'
        
        df.to_csv(path_or_buf=filename, sep=',', columns=None, header=True, index=False)
        return filename
