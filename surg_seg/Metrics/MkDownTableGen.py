from collections import defaultdict
from typing import List
from tabulate import tabulate
from abc import ABC, abstractmethod

tabulate.PRESERVE_WHITESPACE = True

"""
Class to generate markdown tables. 

Errors:
- Any empty values will result in incorrect number formatting. 
"""


class MarkdownTable:
    def __init__(self, headers: List[str], table_format: str = "github"):
        self.headers = headers
        self.table_format = table_format
        self.table = []

    def add_data(self, convert_to_float=True, **kwargs):
        """
        Kwargs must be a dictionary with keys that match the headers of the
        table.  If `convert_to_float` is True, then all numerical values are
        converted to float.
        """

        new_data = []
        for header in self.headers:
            if header in kwargs.keys():
                value = self.convert_value(kwargs[header], convert_to_float)
                new_data.append(value)
            else:
                new_data.append(" ")

        self.table.append(new_data)

        ignored_keys = [key for key in kwargs.keys() if key not in self.headers]

        return ignored_keys

    def convert_value(self, value, convert_to_float):
        if isinstance(value, int) and convert_to_float:
            return float(value)
        return value

    def get_full_table(self, floatfmt=".4f"):
        return tabulate(
            self.table, headers=self.headers, tablefmt=self.table_format, floatfmt=floatfmt
        )

    def print(self, floatfmt=".4f"):
        print(f"\n{self.get_full_table(floatfmt=floatfmt)}\n")


if __name__ == "__main__":
    table = MarkdownTable(headers=["type", "q1", "q2", "q3", "q4", "q5", "q6", "cartesian"])

    # fmt: off
    data1 = dict( type="robot", q1=-5.53456, q2=5.0, q3=4.0, q4=5, q5=6, q6=7, cartesian=8)
    data2 = dict( type="robot", q1=-5.5345, q2=5, q3=4, q4=5.0, q5=6, q6=7, cartesian=8)
    data3 = dict( type="network", q4=5, q5=6, cartesian=8)
    data4 = dict( random="network", q4=5, q5=6, cartesian=8)
    # fmt: on

    table.add_data(**data1)
    table.add_data(**data2)
    # table.add_data(**data3)

    print(f"{table.get_full_table(floatfmt='.3f')}\n")

    table2 = MarkdownTable(headers=["type", "q1"])
    table2.add_data(**dict(type="robot", q1=-5.53456))
    table2.add_data(**dict(type="robot", q1=-8.345))

    table2.print(floatfmt=".6f")
