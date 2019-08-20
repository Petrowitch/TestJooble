# coding: utf-8
import csv
from os import path
import time


# -----TSV----- #
class SpecialTsvReader:
    def __init__(self, filename, data_type):
        self.__file = open(filename, 'rt')
        self.__reader = csv.reader(self.__file, delimiter="\t")
        self.column_names = next(self.__reader)
        self.first_row = next(self.__reader)
        self.param_name, *values = (x for x in self.first_row[1].split(','))
        self.values_n = len(values)
        self.data_type = data_type

    def __split_values(self, row):
        return (row[0], *(self.data_type(x) for x in row[1].split(',')[1:]))

    def __del__(self):
        self.__file.close()

    def __iter__(self):
        yield self.__split_values(self.first_row)
        for line in self.__reader:
            yield self.__split_values(line)


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


# _____MEANS_____ #
class GeneralMeanAccum:
    step_n: int
    values: list

    def __init__(self, length: int):
        self._values = [0]*length
        self.step_n = 0

    def add_values(self, new_values: list):
        raise NotImplementedError()

    @property
    def values(self):
        return self._values


class AriphmeticMeanAccumulator(GeneralMeanAccum):
    def add_values(self, new_values: list):
        self.step_n += 1
        self._values = list(map(
            lambda old_mean, new_value: (old_mean + (new_value-old_mean)/self.step_n),
            self._values, new_values
        ))


class StandartDeviationAccumulator(GeneralMeanAccum):
    def add_values(self, new_values: list, old_means: list, new_means: list):
        self.step_n += 1
        self._values = list(map(
            lambda deviation, value, old_mean, new_mean: deviation + (value-new_mean)*(value-old_mean),
            self._values, new_values, old_means, new_means
        ))

    @property
    def values(self):
        return [(x/(self.step_n-1))**0.5 for x in self._values]


# -----SCORES----- #
class GeneralScore:
    data_type: type

    def __init__(self):
        self.train_file_name = None

    def train(self, train_file_name):
        self.train_file_name = train_file_name
        self._train(SpecialTsvReader(self.train_file_name, self.data_type))

    def _train(self, reader):
        raise NotImplementedError()

    def calculate(self, calculation_file_name):
        if self.train_file_name is None:
            raise Exception('{} was not trained! plz run .train("train_file_name.tsv") before!'.format(type(self).__name__))
        temp_name = path.splitext(calculation_file_name)
        result_file_name = ''.join((*temp_name[0:-1], '_proc', temp_name[-1]))
        reader = SpecialTsvReader(calculation_file_name, self.data_type)
        writer = SpecialTsvWriter(result_file_name, self._get_writer_column_names(reader))
        start_time = time.time()
        i = 0
        last_time = start_time
        for job_id, *values in reader:
            writer.write_values([job_id, *self._calculate_row(values)])
            i += 1
            if i % 10000 == 0:
                curr_time = time.time()
                print("{}){}, {}".format(i, curr_time - start_time, curr_time - last_time))
                last_time = curr_time
        print("process finished for {} entries. time spent {}".format(i, time.time() - start_time))

    def _calculate_row(self, values):
        raise NotImplementedError()

    @staticmethod
    def _get_writer_column_names(reader):
        raise NotImplementedError()


class ZScore(GeneralScore):
    data_type = int

    @staticmethod
    def _get_writer_column_names(reader):
        return [
            reader.column_names[0],
            *("feature_{}_stand_{}".format(reader.param_name, i) for i in range(reader.values_n)),
            "max_feature_{}_index".format(reader.param_name),
            "max_feature_{}_abs_mean_diff".format(reader.param_name)
        ]

    def _train(self, reader):
        mean = AriphmeticMeanAccumulator(reader.values_n)
        std = StandartDeviationAccumulator(reader.values_n)
        for index, *values in reader:
            old_means = mean.values
            mean.add_values(values)
            std.add_values(values, old_means, mean.values)
        self.mean = mean.values
        self.std = std.values

    def _calculate_row(self, values):
        stand = list(map(lambda value, mean, std: (value-mean)/std, values, self.mean, self.std))
        max_index, max_value = max(enumerate(values), key=lambda x: x[1])
        return [*stand, max_index, abs(max_value-self.mean[max_index])]


if __name__ == "__main__":
    scorer = ZScore()
    scorer.train('train.tsv')
    scorer.calculate('test.tsv')
