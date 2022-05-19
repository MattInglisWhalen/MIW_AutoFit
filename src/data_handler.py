
# built-in libraries
import csv as csv
import math
import re as regex

# external libraries
import numpy as np
import xlrd as excel

# user-defined classes
from autofit.src.datum1D import Datum1D


class DataHandler:

    def __init__(self, filepath):

        self._filepath = filepath

        # csv variables
        self._num_lines = 0
        self._line_width = 0
        self._x_label = None
        self._y_label = None
        self._header_flag = 0

        # excel variables
        self._sheet_index = 0
        self._x_column_endpoints = None
        self._sigmax_column_endpoints = None
        self._y_column_endpoints = None
        self._sigmay_column_endpoints = None

        # data
        self._data = []
        self._histogram_flag = False

        if filepath[-4:] == ".csv" or filepath[-4:] == ".txt" :
            self.read_csv()
        elif filepath[-4:] == ".xls" or filepath[-5:] == ".xlsx" or filepath[-4:] == ".ods" :
            print("Please provide start and endpoints for x-values")
            print("Please provide start and endpoints for y-values")
            x_bounds, y_bounds = ("A1","A7") , ("B1","B7")
            self.read_excel()

    @property
    def data(self):
        return self._data
    @property
    def x_label(self):
        return self._x_label
    @property
    def y_label(self):
        return self._y_label
    @property
    def histogram_flag(self):
        return self._histogram_flag

    def calc_num_lines(self):
        self._num_lines = sum(1 for _ in open(self._filepath))
        return self._num_lines

    def calc_entries_per_line(self, delim):
        with open(self._filepath) as file:
            for line in file :
                # read the first line only
                data_str = DataHandler.cleaned_line_as_str_list(line, delim)
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
            self._histogram_flag = True
        else:
            self.read_as_scatter(delim)

    def read_as_histogram(self,delim):

        vals = []
        with open(self._filepath) as file:

            # single line data set
            if self._num_lines == 1:
                for line in file:
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    if self._header_flag:
                        self._x_label = data_str[0]
                        vals = [float(item) for item in data_str[1:]]
                    else:  # no label for data
                        self._x_label = "x"
                        vals = [float(item) for item in data_str]

            # single column dataset
            for line_num, line in enumerate(file) :
                data_str = DataHandler.cleaned_line_as_str_list(line, delim)

                if line_num == 0 :
                    if self._header_flag :
                        self._x_label = data_str[0]
                    else :  # no label for data
                        self._x_label = "x"
                        vals.append( float(data_str[0]) )
                else:
                    vals.append( float(data_str[0]) )
        self._y_label = "N"
        self.make_histogram_data_from_vals(vals)

    def make_histogram_data_from_vals(self, vals):
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
                data_str = DataHandler.cleaned_line_as_str_list(first_line, delim)
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
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1])
                                              )
                                      )

            if self._line_width == 3:
                # x, y, and sigma_y values
                for line in file :
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1]),
                                              sigma_val=float(data_str[2])
                                              )
                                      )
            if self._line_width == 4:
                # x, sigma_x, y, and sigma_y values
                for line in file :
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[2]),
                                              sigma_pos=float(data_str[1]),
                                              sigma_val=float(data_str[3])
                                              )
                                      )

    def set_excel_args(self, x_range_tuple, y_range_tuple=None, x_error_tuple = None, y_error_tuple = None,
                       all_sheets_flag=True):
        pass

    @staticmethod
    def excel_range_as_list_of_idx_tuples(excel_vec):
        left, right = regex.split(f":", excel_vec)
        left_chars = regex.split( f"[0-9]*", left)[0]
        left_ints = regex.split( f"[A-Z]*", left)[-1]

        right_chars = regex.split( f"[0-9]*", left)[0]
        right_ints = regex.split( f"[A-Z]*", left)[-1]

        if left_chars == right_chars :
            return [ (DataHandler.excel_chars_as_idx(left_chars), idx)
                       for idx in range( DataHandler.excel_ints_as_idx(left_ints),
                                         DataHandler.excel_ints_as_idx(left_ints)  )
                   ]
        if left_ints == right_ints :
            return [ (idx, DataHandler.excel_chars_as_idx(left_ints), )
                       for idx in range( DataHandler.excel_chars_as_idx(left_chars),
                                         DataHandler.excel_chars_as_idx(right_chars)  )
                   ]

    @staticmethod
    def excel_cell_as_idx_tuple(excel_cell_str):

        chars = regex.split( f"[0-9]*", excel_cell_str)[0]
        ints = regex.split( f"[A-Z]*", excel_cell_str)[-1]

        return DataHandler.excel_chars_as_idx(chars), int(ints)

    @staticmethod
    def excel_chars_as_idx(chars):

        length = len(chars)
        power = length-1
        integer = 0
        for char in chars:
            integer += (ord(char)-65) * 26**power
            power -= 1
        return integer

    @staticmethod
    def excel_ints_as_idx(ints):
        return int(ints-1)

    def read_excel(self):

        if self._y_column_endpoints is None:
            self.read_excel_as_histogram()
            self._histogram_flag = True
        else:
            self.read_excel_as_scatter()

    def read_excel_as_histogram(self):

        wb = excel.open_workbook(self._filepath)
        sheet = wb.sheet_by_index(self._sheet_index)

        vals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            val = sheet.cell_value(*loc)
            if idx == 0 and regex.search("[a-zA-Z]", val) :
                self._x_label = val
            else:
                self._x_label = "x"
                vals.append( val )
        self._y_label = "N"
        self.make_histogram_data_from_vals(vals)

    def read_excel_as_scatter(self):

        wb = excel.open_workbook(self._filepath)
        sheet = wb.sheet_by_index(self._sheet_index)

        xvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            val = sheet.cell_value(*loc)
            if idx == 0 and regex.search("[a-zA-Z]", val) :
                self._x_label = val
            else:
                xvals.append( val )

        yvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._y_column_endpoints )) :
            val = sheet.cell_value(*loc)
            if idx == 0 and regex.search("[a-zA-Z]", val) :
                self._y_label = val
            else:
                yvals.append( val )

        # create the data
        for x, y in zip(xvals, yvals) :
            self._data.append( Datum1D(pos=x, val=y) )

        sigmaxvals = []
        if self._sigmax_column_endpoints is not None:
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmax_column_endpoints )):
                val = sheet.cell_value(*loc)
                if idx == 0 and regex.search("[a-zA-Z]", val):
                    pass
                else:
                    sigmaxvals.append(val)

        sigmayvals = []
        if self._sigmay_column_endpoints is not None:
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmay_column_endpoints )):
                val = sheet.cell_value(*loc)
                if idx == 0 and regex.search("[a-zA-Z]", val):
                    pass
                else:
                    sigmayvals.append(val)

        # add the errors to the data
        for idx, err in enumerate(sigmaxvals) :
            self._data[idx].sigma_pos = err
        for idx, err in enumerate(sigmayvals) :
            self._data[idx].sigma_val = err




    def normalize_histogram_data(self):

        area = 0
        count = 0
        for datum in self._data :
            count += datum.val

            bin_height = datum.val
            bin_width = 2*datum.sigma_pos  # relies on histogram x errors corresponding to half the bin width
            area += bin_height*bin_width
        if abs(area - 1) < 1e-5 :
            print("Histogram already normalized")
            return

        for datum in self._data :

            bin_height = datum.val
            bin_width = 2*datum.sigma_pos
            bin_mass_density = bin_height/bin_width
            bin_probability_density = bin_mass_density/count

            datum.val = bin_probability_density
            datum.sigma_val = datum.sigma_val / (bin_width * count )
        self._y_label = "probability density"


