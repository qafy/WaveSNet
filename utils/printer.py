"""
This script keeps a printer that saves certain log messages 
to a file while outputting them to the screen, and saves the time 
when the log message was generated
"""

import os
from datetime import datetime


class Printer():
    """
    Describes a type that simultaneously outputs a string to the terminal and saves it to a file
    """
    def __init__(self, file):
        self.file = file
        self.open_or_close = False
        self._check()
        self._open()

    def _check(self):
        """"""
        path, _ = os.path.split(self.file)
        assert os.path.isdir(path)

    def _open(self):
        self.info = open(self.file, 'w')
        self.open_or_close = True

    def _close(self):
        self.info.close()
        self.open_or_close = False

    def pprint(self, text):
        """
        Output the string to the screen or terminal while writing it as a line to a file
        :param text: The string to be output
        :return:
        """
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        print(time + text)
        self.info.write(time + text + '\n')

def pr():
    """
    there is nothing
    :return:
    """
    pass

if __name__ == '__main__':
    print(pr.__doc__)