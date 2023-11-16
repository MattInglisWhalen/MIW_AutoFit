
# built-in libraries
import re as regex
from typing import Union

# external libraries
import numpy as np
from pandas import read_excel, isna

# user-defined classes
from autofit.src.datum1D import Datum1D
from autofit.src.package import logger


class DataHandler:
    """
    Each dataset is provided a DataHandler instance to do I/O reading and writing
    """
    def __init__(self, filepath):

        self._filepath = filepath

        # csv variables
        self._num_lines: int = 0
        self._line_width: int = 0
        self._x_label: str = ""
        self._y_label: str = ""
        self._header_flag: int = 0
        self._delim: str = ','

        # excel variables
        self._excel_sheet_name: int = 0
        self._x_column_endpoints: str = ""
        self._sigmax_column_endpoints: str = ""
        self._y_column_endpoints: str = ""
        self._sigmay_column_endpoints: str = ""

        # data
        self._data: list[Datum1D] = []
        self._histogram_flag: bool = False
        self._normalized_histogram_flag: bool = False

        # logarithmic handling and view
        self._logx_flag: bool = False
        self._X0: float = 1.
        self._logy_flag: float = False
        self._Y0: float = 1.

        # TODO:
        # consider adding an optimizer instance to the DataHandler class, so that each file's optimization can only
        # be accessed through the file handler

        if filepath[-4:] in [".csv",".txt"] :
            self.read_csv()
        elif filepath[-4:] in [".xls","xlsx",".ods"] :
            logger("Please first provide start- and end-points for data ranges")

    @property
    def filepath(self) -> str:
        return self._filepath
    @property
    def shortpath(self) -> str:
        return regex.split("/", self._filepath)[-1]
    @property
    def data(self) -> list[Datum1D]:
        return self._data
    @property
    def x_label(self) -> str:
        return self._x_label
    @property
    def y_label(self) -> str:
        return self._y_label
    @property
    def histogram_flag(self) -> bool:
        return self._histogram_flag
    @property
    def normalized(self) -> bool:
        return self._normalized_histogram_flag
    @property
    def X0(self) -> float:
        return self._X0
    @X0.setter
    def X0(self, val):
        self._X0 = val
    @property
    def logx_flag(self) -> bool:
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
            if min_X <= 0 :
                logger("You can't log the x-data if there are negative numbers!")
                return
            self._X0 = np.sqrt(min_X*max_X) if self.X0 > 0 else -self._X0
            for datum in self._data :
                if self._histogram_flag :
                    sigma_lower, sigma_upper = datum.assym_sigma_pos[0]/datum.pos, datum.assym_sigma_pos[1]/datum.pos
                    datum.assym_sigma_pos = (sigma_lower, sigma_upper)
                else:
                    datum.sigma_pos = datum.sigma_pos/datum.pos
                datum.pos = np.log(datum.pos / self._X0)
                # logger(datum.pos)
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
                    datum.pos = self._X0 * np.exp(datum.pos)
                    if self._histogram_flag :
                        sigma_lower, sigma_upper = datum.pos*datum.assym_sigma_pos[0],datum.pos*datum.assym_sigma_pos[1]
                        datum.assym_sigma_pos = (sigma_lower, sigma_upper)
                    else:
                        datum.sigma_pos = datum.sigma_pos * datum.pos
        self._logx_flag = new_flag
        logger(f"Finished logging x with {self._X0=}")
    @property
    def Y0(self) -> float:
        return self._Y0
    @Y0.setter
    def Y0(self, val):
        self._Y0 = val
    @property
    def logy_flag(self) -> bool:
        return self._logy_flag
    @logy_flag.setter
    def logy_flag(self, new_flag):
        if new_flag and not self._logy_flag :
            # we are not currently logging the data, but want to log it now
            min_Y, max_Y = min( [datum.val for datum in self._data] ), max( [datum.val for datum in self._data] )
            if min_Y <= 0 :
                logger("You can't log the y-data if there are zeroes or negative numbers!")
                return
            self._Y0 = np.sqrt(min_Y*max_Y) if self._Y0 > 0 else -self._Y0
            for datum in self._data :
                datum.sigma_val = datum.sigma_val/datum.val
                datum.val = np.log( datum.val / self._Y0 )
        if not new_flag and self._logy_flag :
            # we are currently logging the data, but now we want to switch it back
            for datum in self._data :
                datum.val = self._Y0 * np.exp( datum.val )
                datum.sigma_val = datum.sigma_val*datum.val
        self._logy_flag = new_flag
        logger(f"Finished logging y with {self._Y0=}")

    @property
    def unlogged_x_data(self) -> list[float]:
        if self._logx_flag :
            # we're logging the data, so return it unlogged
            return [self._X0 * np.exp( datum.pos ) for datum in self._data ]
        # else
        return [datum.pos for datum in self._data]
    @property
    def unlogged_y_data(self) -> list[float]:
        if self._logy_flag :
            # we're logging the data, so return it unlogged
            return [self._Y0 * np.exp( datum.val ) for datum in self._data ]
        # else
        return [datum.val for datum in self._data]

    @property
    def unlogged_sigmax_data(self) -> list[float]:
        if self._logx_flag :
            # we're logging the data, so return it unlogged
            if self._histogram_flag :
                return np.array( [ ( datum.assym_sigma_pos[0] * self._X0 * np.exp(datum.pos),
                                     datum.assym_sigma_pos[1] * self._X0 * np.exp(datum.pos) )
                                     for datum in self._data] ).T
            else:
                return [datum.sigma_pos * self._X0 * np.exp(datum.pos) for datum in self._data]
        # else
        return [datum.sigma_pos for datum in self._data]
    @property
    def unlogged_sigmay_data(self) -> list[float]:
        if self._logy_flag :
            # we're logging the data, so return it unlogged
            return [datum.sigma_val * self._Y0 * np.exp(datum.val) for datum in self._data]
        # else
        return [datum.sigma_val for datum in self._data]


    def calc_num_lines(self) -> int:
        self._num_lines = sum(1 for _ in open(self._filepath))
        return self._num_lines

    # TODO: add support for non-comma delimeters. Need to get user input
    def calc_entries_per_line(self, delim) -> int:
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
        return 0

    @ staticmethod
    def cleaned_line_as_str_list(line, delim) -> list[str]:
        data = regex.split(f"\s*{delim}\s*", line[:-1])
        while data[-1] == "":
            data = data[:-1]
        return data

    def read_csv(self, delim = ','):

        length = self.calc_num_lines()
        width = self.calc_entries_per_line(delim)

        logger(f"File is {width}x{length}")

        if (length == 1 and width > 1) or (length > 1 and width == 1) :
            self.read_as_histogram(delim)
            self._histogram_flag = True
        else:
            self.read_as_scatter(delim)
    def read_as_scatter(self,delim):
        # should probably be using pandas pd.read_csv()
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
                    if line == "\n" :
                        continue
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1])
                                              )
                                      )

            if self._line_width == 3:
                # x, y, and sigma_y values
                for line in file :
                    if line == "\n" :
                        continue
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[1]),
                                              sigma_val=float(data_str[2])
                                              )
                                      )
            if self._line_width == 4:
                # x, sigma_x, y, and sigma_y values
                for line in file :
                    if line == "\n" :
                        continue
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    self._data.append(Datum1D(pos=float(data_str[0]),
                                              val=float(data_str[2]),
                                              sigma_pos=float(data_str[1]),
                                              sigma_val=float(data_str[3])
                                              )
                                      )
    def read_as_histogram(self,delim):
        # should probably be using pandas pd.read_csv()
        vals = []
        with open(self._filepath) as file:

            # single line data set
            if self._num_lines == 1:
                for line in file:
                    if line == "\n" :
                        continue
                    data_str = DataHandler.cleaned_line_as_str_list(line, delim)
                    if self._header_flag:
                        self._x_label = data_str[0]
                        vals = [float(item) for item in data_str[1:]]
                    else:  # no label for data
                        self._x_label = "x"
                        vals = [float(item) for item in data_str]

            # single column dataset
            for line_num, line in enumerate(file) :
                if line == "\n":
                    continue
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

    def bin_width(self) -> float:
        if not self.histogram_flag :
            return -1.
        if self.logx_flag :
            return -1.
        if len(self._data) < 2 :
            return 1.
        return self._data[1].pos - self._data[0].pos

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
            logger(f"Can't x-log histogram when {minval=}<0")
            self._logx_flag = False

        if minval - np.floor(minval) < 2/count :
            # logger("Integer bolting min")
            # if it looks like the min and max vals are bolted to an integer, use the integers as a bin boundary
            minval = minval // 1
        if np.ceil(maxval) - maxval < 2/count :
            # logger("Integer bolting max")
            maxval = maxval // 1

        num_bins = int( np.sqrt(count) // 1 )

        if self._logx_flag :
            hist_counts, hist_bounds = np.histogram(vals, bins=np.geomspace(minval, maxval, num=num_bins+1) )
            if min( hist_counts ) == 0 and self._logy_flag :
                logger("In make_histogram_data, your can't x-log for this data because you're already y-logging, "
                       "and you can't take the log of 0.")
                self._logx_flag = False
        # this used to be         if not self.logx_flag :
        else :
            hist_counts, hist_bounds = np.histogram(vals,  bins=np.linspace(minval, maxval, num=num_bins+1) )
        logger(f"Made histogram with bin counts {hist_counts}")
        if 0 in hist_counts :
            logger(f"Histogram creation error with {hist_bounds=}")
        if self._logx_flag :
            for idx, count in enumerate(hist_counts) :
                geom_mean = np.sqrt(hist_bounds[idx+1]*hist_bounds[idx])
                self._data.append( Datum1D( pos = geom_mean,
                                            val = count,
                                            assym_sigma_pos = (geom_mean-hist_bounds[idx],hist_bounds[idx+1]-geom_mean),
                                            sigma_val = max(np.sqrt(count),1)
                                          )  # check that this fixing the zero-bin error not allowing fit
                                 )
        else:
            for idx, count in enumerate(hist_counts) :
                self._data.append( Datum1D( pos = ( hist_bounds[idx+1]+hist_bounds[idx] )/2,
                                            val = count,
                                            sigma_pos = ( hist_bounds[idx+1]-hist_bounds[idx] )/2,
                                            sigma_val = max(np.sqrt(count),1)
                                          )
                                 )
        if self._logy_flag :
            # Funny thing with the setter of logy_flag
            self._logy_flag = False
            self.logy_flag = True

    def set_excel_args(self, x_range_str, y_range_str="", x_error_str = "", y_error_str = ""):
        logger(f"Thank you for providing data ranges {x_range_str} {y_range_str} {x_error_str} {y_error_str}")
        self._x_column_endpoints = x_range_str
        self._y_column_endpoints = y_range_str
        self._sigmax_column_endpoints = x_error_str
        self._sigmay_column_endpoints = y_error_str
        self.read_excel()
    def set_excel_sheet_name(self, name):
        logger("Thank you for providing data ranges")
        self._excel_sheet_name = name

    @staticmethod
    def valid_excel_endpoints(excel_range_str: str) -> bool:
        if excel_range_str == "":
            return True
        if regex.match("[A-Z][A-Z]*[0-9][0-9]*:[A-Z][A-Z]*[0-9][0-9]*", excel_range_str) :
            return True
        return False
    @staticmethod
    def excel_range_as_list_of_idx_tuples(excel_vec):
        # logger(f"{excel_vec} as range should be")
        if excel_vec == "" :
            # for empty range creation, e.g. empty sigma_x range
            return []
        try:
            left, right = regex.split(f":", excel_vec)
        except ValueError :
            logger(f"{excel_vec=}")
            raise ValueError
        # logger(f"{left=} {right=}")
        left_chars = regex.split( f"[0-9]", left)[0]
        left_ints = regex.split( f"[A-Z]", left)[-1]

        right_chars = regex.split( f"[0-9]", right)[0]
        right_ints = regex.split( f"[A-Z]", right)[-1]

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
    def excel_chars_as_idx(chars) -> int:

        length = len(chars)
        power = length-1
        integer = 0
        for char in chars:
            integer += (ord(char)-64) * 26**power
            power -= 1
        return integer-1
    @staticmethod
    def excel_ints_as_idx(ints) -> int:
        return int(ints)-1

    def read_excel(self):

        if self._y_column_endpoints == "" :
            self.read_excel_as_histogram()
            self._histogram_flag = True
        else:
            self.read_excel_as_scatter()

    def read_excel_as_scatter(self):

        data_frame = read_excel(self._filepath, self._excel_sheet_name, header=None)

        logger("Excel scatter plot chosen")

        xvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            try:
                val = data_frame.iloc[loc[0],loc[1]]
            except IndexError:
                print(data_frame)
                logger(f"Provided spreadsheet range is invalid for spreadsheet at {self._filepath}.")
            if idx == 0 and regex.search("[a-zA-Z]", str(val) ) :
                self._x_label = val
            elif isna(val) :
                logger(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                xvals.append( val )

        logger("Done x collection")

        yvals = []
        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._y_column_endpoints )) :
            try:
                val = data_frame.iloc[loc[0],loc[1]]
            except IndexError:
                print(data_frame)
                logger(f"Provided spreadsheet range is invalid for spreadsheet at {self._filepath}.")
            if idx == 0 and regex.search("[a-zA-Z]", str(val) ) :
                self._y_label = val
            elif isna(val) :
                logger(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                yvals.append( val )

        logger("Done y collection")

        # create the data
        for x, y in zip(xvals, yvals) :
            self._data.append( Datum1D(pos=x, val=y) )

        sigmaxvals = []
        if DataHandler.valid_excel_endpoints(self._sigmax_column_endpoints):
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmax_column_endpoints )):
                try:
                    val = data_frame.iloc[loc[0], loc[1]]
                except IndexError:
                    print(data_frame)
                    logger(f"Provided spreadsheet range is invalid for spreadsheet at {self._filepath}.")
                if idx == 0 and regex.search("[a-zA-Z]", str(val) ):
                    pass
                elif isna(val):
                    logger(f"Invalid value >{val}< encountered in excel workbook. ")
                    continue
                else:
                    sigmaxvals.append(val)

        sigmayvals = []
        if DataHandler.valid_excel_endpoints(self._sigmay_column_endpoints):
            for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._sigmay_column_endpoints )):
                try:
                    val = data_frame.iloc[loc[0], loc[1]]
                except IndexError:
                    print(data_frame)
                    logger(f"Provided spreadsheet range is invalid for spreadsheet at {self._filepath}.")
                if idx == 0 and regex.search("[a-zA-Z]", str(val) ):
                    pass
                elif isna(val):
                    logger(f"Invalid value >{val}< encountered in excel workbook. ")
                    continue
                else:
                    sigmayvals.append(val)

        # add the errors to the data
        for idx, err in enumerate(sigmaxvals) :
            self._data[idx].sigma_pos = err
        for idx, err in enumerate(sigmayvals) :
            self._data[idx].sigma_val = err
    def read_excel_as_histogram(self):

        data_frame = read_excel(self._filepath, self._excel_sheet_name, header=None)

        logger("Excel histogram chosen")

        vals = []

        for idx, loc in enumerate(DataHandler.excel_range_as_list_of_idx_tuples( self._x_column_endpoints )) :
            try:
                val = data_frame.iloc[loc[0],loc[1]]
            except IndexError:
                print(data_frame)
                logger(f"Provided spreadsheet range is invalid for spreadsheet at {self._filepath}.")
            if idx == 0 and regex.search("[a-zA-Z]", str(val)) :
                self._x_label = val
            elif str(val) == "" :
                logger(f"Invalid value >{val}< encountered in excel workbook. ")
                continue
            else:
                self._x_label = "x"
                vals.append( val )
        self._y_label = "N"
        logger(f"Excel histogram raw data is {vals}")
        self.make_histogram_data_from_vals(vals)

    def normalize_histogram_data(self, error_handler = print) -> bool:

        area = 0
        count = 0

        # if we are y-logging the data, unlog the data then restore the y-logging at the end
        if self.logy_flag :
            for datum in self._data :
                datum.val = self._Y0 * np.exp( datum.val )
                datum.sigma_val = datum.sigma_val*datum.val

                # # we are currently logging the data, but now we want to switch it back
                # for datum in self._data :
                #     datum.pos = self._Y0 * np.exp( datum.pos )
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
            error_handler("Histogram already normalized")
            return True

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
            if min_Y <= 0:
                error_handler("\n \n> Normalize: You can't log the y-data if there are non-positive numbers!")
                self._normalized_histogram_flag = False
                return False
            self._Y0 = np.sqrt(min_Y * max_Y)
            for datum in self._data:
                datum.sigma_val = datum.sigma_val / datum.val
                datum.val = np.log(datum.val / self._Y0)

        self._normalized_histogram_flag = True
        return True
