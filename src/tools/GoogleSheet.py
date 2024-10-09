import pygsheets


class GoogleSheet:
    """
    A class for interacting with Google Sheets.
    """
    def __init__(self, sheet_name, service_file=None):
        """
        Initialize a GoogleSheet object with the given options.
        
        Args:
            config: A configuration object containing the service file path.
        """
        self.gc = pygsheets.authorize(service_file=service_file)
        self.worksheets = {}
        self.sheet = self.gc.open(sheet_name)

    def load_data(self, wks_name, read_start, read_end):
        """
        Load data from the specified sheet and range.
        
        Args:
            read_start (str): The start cell of the data range (e.g., 'A1').
            read_end (str): The end cell of the data range (e.g., 'B10').
            
        Returns:
            np.array: The data from the specified range as a NumPy array.
        """
        wks = self.worksheets[wks_name] if wks_name in self.worksheets else self.sheet.worksheet_by_title(wks_name)
        data_range = f"{read_start}:{read_end}"
        cell_list=wks.get_values_batch([data_range])
        data = [cell for row in cell_list for cell in row]
        return data

    def write_cell(self, wks_name, cell_location, value):
        wks = self.worksheets[wks_name] if wks_name in self.worksheets else self.sheet.worksheet_by_title(wks_name)
        cell = wks.cell(cell_location)
        cell.value = str(value)

    def write_data(self, wks_name, data, ws):
        """
        Write data to the specified sheet.
        
        Args:
            data (pd.DataFrame): The data to write as a pandas DataFrame.
            ws (int): The current worksheet number.
        """
        write_start = self.column_shift('F', (ws - 1) * 8) + '4'
        wks = self.worksheets[wks_name] if wks_name in self.worksheets else self.sheet.worksheet_by_title(wks_name)
        wks.set_dataframe(data, write_start, extend=True)

    def column_shift(self, column, shift):
        """
        Shift the given column by the specified number.
        
        Args:
            column (str): The initial column as a letter (e.g., 'A').
            shift (int): The number of columns to shift by.
            
        Returns:
            str: The new column as a letter.
        """
        asc_src = 0
        s = 1
        while len(column) > 0:
            asc_src += (ord(column[-1]) - 64) * s
            column = column[:-1]
            s *= 26
        
        asc_column = asc_src + shift
        new_column = ''
        
        while asc_column > 0:
            remainder = asc_column % 26
            if remainder == 0:
                remainder = 26
                asc_column -= 1
            asc_column = asc_column // 26
            new_column = chr(remainder + 64) + new_column
        
        return new_column
