
# built-in libraries
import csv as csv
import math
import re as regex

# external libraries
import numpy as np
# import xlrd as excel
import pandas as pd

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
        self._delim = ','

        # excel variables
        self._excel_sheet_name = 0
        self._x_column_endpoints = None
        self._sigmax_column_endpoints = None
        self._y_column_endpoints = None
        self._sigmay_column_endpoints = None

        # data
        self._data = []
        self._histogram_flag = False
        self._normalized_histogram_flag = False

        # logarithmic handling and view
        self._logx_flag = False
        self._X0 = 1.
        self._logy_flag = False
        self._Y0 = 1.

        if filepath[-4:] in [".csv",".txt"] :
            self.read_csv()
        elif filepath[-4:] in [".xls","xlsx",".ods"] :
            print("Please first provide start- and end-points for data ranges")

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
    @property
    def X0(self):
        return self._X0
    @property
    def logx_flag(self):
        return self._logx_flag
    @logx_flag.setter
    def logx_flag(self, new_flag):
        if new_flag and not self._logx_flag :
            # we are not currently logging the data, but want to log it now
            if self._histogram_flag :
                # calculates the regular-space data, but with geometrically-spaced widths
                self._logx_flag = True
                self.recalculate_bins()
                if not self._logx_flag :
                    # there was a reason we couldn't take the x-log, so break out of function
                    return
                new_flag =  True
                # re-normalize if it was normalized before
                if self._normalized_histogram_flag :
                    self.normalize_histogram_data()

            min_X, max_X = min( [datum.pos for datum in self._data] ), max( [datum.pos for datum in self._data] )
            if min_X < 0 :
                print("You can't log the x-data if there are negative numbers!")
                return
            self._X0 = math.sqrt(min_X*max_X)
            for datum in self._data :
                if self._histogram_flag :
                    sigma_lower, sigma_upper = datum.assym_sigma_pos[0]/datum.pos, datum.assym_sigma_pos[1]/datum.pos
                    datum.assym_sigma_pos = (sigma_lower, sigma_upper)
                else:
                    datum.sigma_pos = datum.sigma_pos/datum.pos
                datum.pos = math.log(datum.pos / self._X0)
                # print(datum.pos)
        if not new_flag and self._logx_flag :
            # we are currently logging the data, but now we want to switch it back
            if self._histogram_flag :
                self._logx_flag = False
                self.recalculate_bins()
                # re-normalize if it was normalized before
                if self._normalized_histogram_flag :
                    self.normalize_histogram_data()
            else:
                for datum in self._data :
                    datum.pos = self._X0 * math.exp(datum.pos)
                    if self._histogram_flag :
                        sigma_lower, sigma_upper = datum.pos*datum.assym_sigma_pos[0], datum.pos*datum.assym_sigma_pos[1]
                        datum.assym_sigma_pos = (sigma_lower, sigma_upper)
                    else:
                        datum.sigma_pos = datum.sigma_pos * datum.pos
        self._logx_flag = new_flag
        print("Finished logging x")
    @property
    def Y0(self):
        return self._Y0
    @property
    def logy_flag(self):
        return self._logy_flag
    @logy_flag.setter
    def logy_flag(self, new_flag):
        if new_flag and not self._logy_flag :
            # we are not currently logging the data, but want to log it now
            min_Y, max_Y = min( [datum.val for datum in self._data] ), max( [datum.val for datum in self._data] )
            if min_Y <= 0 :
                print("You can't log the y-data if there are zeroes or negative numbers!")
                return
            self._Y0 = math.sqrt(min_Y*max_Y)
            for datum in self._data :
                datum.sigma_val = datum.sigma_val/datum.val
                datum.val = math.log( datum.val / self._Y0 )
        if not new_flag and self._logy_flag :
            # we are currently logging the data, but now we want to switch it back
            for datum in self._data :
                datum.val = self._Y0 * math.exp( datum.val )
                datum.sigma_val = datum.sigma_val*datum.val
        self._logy_flag = new_flag

    @property
    def unlogged_x_data(self):
        if self._logx_flag :
            # we're logging the data, so return it unlogged
            return [self._X0 * math.exp( datum.pos ) for datum in self._data ]
        # else
        return [datum.pos for datum in self._data]

    @property
    def unlogged_y_data(self):
        if self._logy_flag :
            # we're logging the data, so return it unlogged
            return [self._Y0 * math.exp( datum.val ) for datum in self._data ]
        # else
        return [datum.val for datum in self._data]

    @property
    def unlogged_sigmax_data(self):
        if self._logx_flag :
            # we're logging the data, so return it unlogged
            if self._histogram_flag :
                return np.array( [ ( datum.assym_sigma_pos[0] * self._X0 * math.exp(datum.pos),
                                     datum.assym_sigma_pos[1] * self._X0 * math.exp(datum.pos) )
                                     for datum in self._data] ).T
            else:
                return [datum.sigma_pos * self._X0 * math.exp(datum.pos) for datum in self._data]
        # else
        return [datum.sigma_pos for datum in self._data]
    @property
    def unlogged_sigmay_data(self):
        if self._logy_flag :
            # we're logging the data, so return it unlogged
            return [datum.sigma_val * self._Y0 * math.exp(datum.val) for datum in self._data]
        # else
        return [datum.sigma_val for datum in self._data]



    def calc_num_lines(self):
        self._num_lines = sum(1 for _ in open(self._filepath))
        return self._num_lines

    # TODO: add support for non-comma delimeters. Need to get user input
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
                if self._num_lines > 1 and self._line_width > 4 :
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

    def recalculate_bins(self):
        self._data = []
        if self._filepath[-4:] in [".xsl", "xlsx", ".ods" ]:
            self.read_excel_as_histogram()
        if self._filepath[-4:] in [".csv", ".txt"] :
            self.read_as_histogram(self._delim)

    def make_histogram_data_from_vals(self, vals):
        # bin the values, with a minimum count per bin of 1, and number of bins = sqrt(count)
        minval, maxval, count = min(vals), max(vals), len(vals)

        if minval < 0 and self._logx_flag :
            print(f"Can't x-log histogram when {minval=}<0")
            self._logx_flag = False

        if minval - math.floor(minval) < 2/count or math.ceil(maxval) - maxval < 2/count :
            # if it looks like the min and max vals are bolted to an integer, use the integers as a bin boundary
            minval = math.floor(minval)
            maxval = math.floor(maxval)

        num_bins = math.floor( math.sqrt(count) )

        if self._logx_flag :
            hist_counts, hist_bounds = np.histogram(vals, bins=np.geomspace(minval, maxval, num=num_bins+1) )
            if min( hist_counts ) == 0 and self._logy_flag :
                print("In make_histogram_data, your can't x-log for this data because you're already y-logging, "
                      "and you can't take the log of 0.")
                self._logx_flag = False
        if not self.logx_flag :
            hist_counts, hist_bounds = np.histogram(vals,  bins=np.linspace(minval, maxval, num=num_bins+1) )
        print(f"Made histogram with bin counts {hist_counts}")
        if self._logx_flag :
            for idx, count in enumerate(hist_counts) :
                geom_mean = math.sqrt(hist_bounds[idx+1]*hist_bounds[idx])
                self._data.append( Datum1D( pos = geom_mean,
                                            val = count,
                                            assym_sigma_pos = (geom_mean-hist_bounds[idx],hist_bounds[idx+1]-geom_mean),
                                            sigma_val = math.sqrt(count)
                                          )
                                 )
        else:
            for idx, count in enumerate(hist_counts) :
                self._data.append( Datum1D( pos = ( hist_bounds[idx+1]+hist_bounds[idx] )/2,
                                            val = count,
                                            sigma_pos = ( hist_bounds[idx+1]-hist_bounds[idx] )/2,
                                            sigma_val = math.sqrt(count)
                                          )
                                 )
        if self._logy_flag :
            # Funny thing with the setter of logy_flag
            self._logy_flag = False
            self.logy_flag = True

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
            else:
                self._x_label = "x"
                self._y_label = "y"

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

    def set_excel_args(self, x_range_str, y_range_str=None, x_error_str = None, y_error_str = None,
                       all_sheets_flag=True):
        print("Thank you for providing data ranges")
        self._x_column_endpoints = x_range_str
        self._y_column_endpoints = y_range_str
        self._sigmax_column_endpoints = x_error_str
        self._sigmay_column_endpoints = y_error_str
        print(self._x_column_endpoints, self._y_column_endpoints)
        self.read_excel()

    def set_excel_sheet_name(self, name):
        print("Thank you for providing data ranges")
        self._excel_sheet_name = name

    @staticmethod
    def excel_range_as_list_of_idx_tuples(excel_vec):
        # print(f"{excel_vec} as range should be")
        if excel_vec == "" :
            # for empty range creation, e.g. empty sigma_x range
            return []
        left, right = regex.split(f"\:", excel_vec)

        print(f"{left=} {right=}")
        left_chars = regex.split( f"[0-9]", left)[0]
        left_ints = regex.split( f"[A-Z]", left)[-1]

        right_chars = regex.split( f"[0-9]", right)[0]
        right_ints = regex.split( f"[A-Z]", right)[-1]

        print(f"{left_chars=} {left_ints=} {right_chars=} {right_ints=}")

        print([idx for idx in
               range(DataHandler.excel_chars_as_idx(left_chars), DataHandler.excel_chars_as_idx(right_chars) + 1)])
        print( [ idx for idx in range(DataHandler.excel_ints_as_idx(left_ints),DataHandler.excel_ints_as_idx(right_ints)+1) ] )
        if left_chars == right_chars :
            # A1 denotes ColRow so transpose the two
            return [ (idx,DataHandler.excel_chars_as_idx(left_chars))
                       for idx in range( DataHandler.excel_ints_as_idx(left_ints),
                                         DataHandler.excel_ints_as_idx(right_ints)+1  )
                   ]

        if left_ints == right_ints :
            return [ (DataHandler.excel_ints_as_idx(left_ints),idx )
                       for idx in range( DataHandler.excel_chars_as_idx(left_chars),
                                         DataHandler.excel_chars_as_idx(right_chars)+1  )
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
            integer += (ord(char)-64) * 26**power
            power -= 1
        return integer-1

    @staticmethod
    def excel_ints_as_idx(ints):
        return int(ints)-1

    def read_excel(self):



        if self._y_column_endpoints == "" :
            self.read_excel_as_histogram()
            self._histogram_flag = True
        else:
            self.read_excel_as_scatter()

    def read_excel_as_histogram(self):

        data_frame = pd.read_excel(self._filepath, self._excel_sheet_name)

        print("Excel histogram chosen")

        vals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            val = data_frame.iloc[loc[0],loc[1]]
            if idx == 0 and regex.search("[a-zA-Z]", str(val)) :
                self._x_label = val
            elif str(val) == "" :
                print(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                self._x_label = "x"
                vals.append( val )
        self._y_label = "N"
        print(f"Excel histogram raw data is {vals}")
        self.make_histogram_data_from_vals(vals)

    def read_excel_as_scatter(self):

        data_frame = pd.read_excel(self._filepath, self._excel_sheet_name, header=None)

        print("Excel scatter plot chosen")

        xvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            val = data_frame.iloc[loc[0],loc[1]]
            if idx == 0 and regex.search("[a-zA-Z]", str(val) ) :
                self._x_label = val
            elif pd.isna(val) :
                print(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                xvals.append( val )

        print("Done x collection")

        yvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._y_column_endpoints )) :
            val = data_frame.iloc[loc[0],loc[1]]
            if idx == 0 and regex.search("[a-zA-Z]", str(val) ) :
                self._y_label = val
            elif pd.isna(val) :
                print(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                yvals.append( val )

        print("Done y collection")

        # create the data
        for x, y in zip(xvals, yvals) :
            self._data.append( Datum1D(pos=x, val=y) )

        sigmaxvals = []
        if self._sigmax_column_endpoints is not None:
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmax_column_endpoints )):
                val = data_frame.iloc[loc[0],loc[1]]
                if idx == 0 and regex.search("[a-zA-Z]", str(val) ):
                    pass
                elif pd.isna(val):
                    print(f"Invalid value >{val}< encountered in excel workbook. ")
                    continue
                else:
                    sigmaxvals.append(val)

        sigmayvals = []
        if self._sigmay_column_endpoints is not None:
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmay_column_endpoints )):
                val = data_frame.iloc[loc[0],loc[1]]
                if idx == 0 and regex.search("[a-zA-Z]", str(val) ):
                    pass
                elif pd.isna(val):
                    print(f"Invalid value >{val}< encountered in excel workbook. ")
                    continue
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

        # if we are y-logging the data, unlog the data then restore the y-logging at the end
        if self.logy_flag :
            for datum in self._data :
                datum.val = self._Y0 * math.exp( datum.val )
                datum.sigma_val = datum.sigma_val*datum.val

                # # we are currently logging the data, but now we want to switch it back
                # for datum in self._data :
                #     datum.pos = self._Y0 * math.exp( datum.pos )
                #     datum.sigma_pos = datum.sigma_pos * datum.pos

        for datum in self._data :


            count += datum.val
            bin_height = datum.val

            if self._logx_flag :
                # relies on histogram asymmetric x-errors corresponding to the upper and lower bin widths
                bin_width = datum.assym_sigma_pos[0]+datum.assym_sigma_pos[1]
            else:
                # relies on histogram x-errors corresponding to half the bin width
                bin_width = 2*datum.sigma_pos

            area += bin_height*bin_width
        if abs(area - 1) < 1e-5 :
            print("Histogram already normalized")
            return

        for datum in self._data :

            bin_height = datum.val

            if self._logx_flag :
                # relies on histogram asymmetric x-errors corresponding to the upper and lower bin widths
                bin_width = datum.assym_sigma_pos[0]+datum.assym_sigma_pos[1]
            else:
                # relies on histogram x-errors corresponding to half the bin width
                bin_width = 2*datum.sigma_pos

            bin_mass_density = bin_height/bin_width
            bin_probability_density = bin_mass_density/count

            datum.val = bin_probability_density
            datum.sigma_val = datum.sigma_val / (bin_width * count)

        self._y_label = "probability density"
        if self._logy_flag :
            min_Y, max_Y = min([datum.val for datum in self._data]), max([datum.val for datum in self._data])
            if min_Y < 0:
                print("You can't log the y-data if there are negative numbers!")
                return
            self._Y0 = math.sqrt(min_Y * max_Y)
            for datum in self._data:
                datum.sigma_val = datum.sigma_val / datum.val
                datum.val = math.log(datum.val / self._Y0)

        self._normalized_histogram_flag = True

