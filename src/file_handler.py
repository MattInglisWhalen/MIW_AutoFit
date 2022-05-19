
# built-in libraries
import csv as csv
import math
import re as regex

# external libraries
import numpy as np

# user-defined classes
from autofit.src.datum1D import Datum1D


class FileHandler:

    def __init__(self, filepath):

        self._filepath = filepath

        # csv variables
        self._num_lines = 0
        self._line_width = 0
        self._x_label = None
        self._y_label = None
        self._header_flag = 0

        # excel variables
        # print(self._num_lines)

        # data
        self._data = []

        if filepath[-4:] == ".csv" or filepath[-4:] == ".txt" :
            self.read_csv()
        elif filepath[-4:] == ".xls" or filepath[-5:] == ".xlsx" or filepath[-4:] == ".ods" :
            print("Please provide start and endpoints for x-values")
            print("Please provide start and endpoints for y-values")
            x_bounds, y_bounds = ("A1","A7") , ("B1","B7")
            self.read_excel(x_bounds,y_bounds)

    @property
    def data(self):
        return self._data
    @property
    def x_label(self):
        return self._x_label
    @property
    def y_label(self):
        return self._y_label

    def calc_num_lines(self):
        self._num_lines = sum(1 for _ in open(self._filepath))
        return self._num_lines

    def calc_entries_per_line(self, delim):
        with open(self._filepath) as file:
            for line in file :
                # read the first line only
                data_str = FileHandler.cleaned_line_as_str_list(line, delim)
                self._line_width = len(data_str)

                # also check for headers
                if regex.search(f"[a-zA-Z]", data_str[0]):
                    self._header_flag = 1

                # error if more than 4 columns
                if self._num_lines>1 and self._line_width > 4 :
                    raise AttributeError

                return self._line_width

    @ staticmethod
    def cleaned_line_as_str_list(line, delim):
        data = regex.split(f"\s*{delim}\s*", line[:-1])
        while data[-1] == "":
            data = data[:-1]
        return data

    def read_csv(self, delim = ','):

        length = self.calc_num_lines()
        width = self.calc_entries_per_line(delim)

        print(f"File is {width}x{length}")

        if (length == 1 and width > 1) or (length > 1 and width == 1) :
            self.read_as_histogram(delim)
        else:
            self.read_as_scatter(delim)

    def read_as_histogram(self,delim):

        vals = []
        with open(self._filepath) as file:

            # single line data set
            if self._num_lines == 1:
                for line in file:
                    data_str = FileHandler.cleaned_line_as_str_list(line, delim)
                    if self._header_flag:
                        self._x_label = data_str[0]
                        vals = [float(item) for item in data_str[1:]]
                    else:  # no label for data
                        self._x_label = "x"
                        self._y_label = "N"
                        vals = [float(item) for item in data_str]

            # single column dataset
            for line_num, line in enumerate(file) :
                data_str = FileHandler.cleaned_line_as_str_list(line,delim)

                if line_num == 0 :
                    if self._header_flag :
                        self._x_label = data_str[0]
                    else :  # no label for data
                        self._x_label = "x"
                        self._y_label = "N"
                        vals.append( float(data_str[0]) )
                else:
                    vals.append( float(data_str[0]) )

        # bin the values, with a minimum count per bin of 1, and number of bins = sqrt(count)
        minval, maxval, count = min(vals), max(vals), len(vals)

        if minval - math.floor(minval) < 2/count or math.ceil(maxval) - maxval < 2/count :
            # if it looks like the min and max vals are bolted to an integer, use the integers as a bin boundary
            minval = math.floor(minval)
            maxval = math.floor(maxval)

        num_bins = math.floor( math.sqrt(count) )
        bin_width = (maxval-minval)/num_bins

        hist_counts, hist_bounds = np.histogram(vals,bins=np.linspace(minval, maxval, num=num_bins+1) )
        print(hist_counts)
        for idx, count in enumerate(hist_counts) :
            self._data.append( Datum1D( pos = ( hist_bounds[idx+1]+hist_bounds[idx] )/2,
                                        val = count,
                                        sigma_pos = bin_width/2,
                                        sigma_val = math.sqrt(count)
                                      )
                             )

    def read_as_scatter(self,delim):

        with open(self._filepath) as file:

            # get headers if they exist
            if self._header_flag :
                first_line = file.readline()
                data_str = FileHandler.cleaned_line_as_str_list(first_line, delim)
                if self._line_width == 2 or self._line_width == 3  :
                    self._x_label = data_str[0]
                    self._y_label = data_str[1]
                if self._line_width == 4 :
                    self._x_label = data_str[0]
                    self._y_label = data_str[2]
                # file seeker/pointer will now be at start of second line when header is read

            # it's messy to repeat the logic and loop, but it's inefficient to have an if in a for loop
            if self._line_width == 2:
                # x and y values
                for line in file :
                    data_str = FileHandler.cleaned_line_as_str_list(line,delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1])
                                              )
                                      )

            if self._line_width == 3:
                # x, y, and sigma_y values
                for line in file :
                    data_str = FileHandler.cleaned_line_as_str_list(line,delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1]),
                                              sigma_val=float(data_str[2])
                                              )
                                      )
            if self._line_width == 4:
                # x, sigma_x, y, and sigma_y values
                for line in file :
                    data_str = FileHandler.cleaned_line_as_str_list(line,delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[2]),
                                              sigma_pos=float(data_str[1]),
                                              sigma_val=float(data_str[3])
                                              )
                                      )



    def read_excel(self, x_range_tuple, y_range_tuple, x_error_tuple = None, y_error_tuple = None):
        pass


