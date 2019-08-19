import csv
import random


class SpecialTsvWriter:
    def __init__(self, filename, column_names):
        self.__file = open(filename, 'wt')
        self.__writer = csv.writer(self.__file, delimiter="\t")
        self.column_names = column_names
        self.column_N = len(column_names)
        self.__writer.writerow(column_names)

    def __del__(self):
        self.__file.close()

    def write_values(self, values):
        self.__writer.writerow(values)
        
writer = SpecialTsvWriter('testgen.tsv', ['id_job', 'features'])

for i in range(10**10):
    writer.write_values([
        random.randint(10**19, 10**20),
        ','.join(('3', *(str(random.randint(9000, 10001)) for x in range(256))))])
