import time
from dataclasses import dataclass
from typing import Any

import clickhouse_connect

from utils.simulationObjects import Reports, IndividualReports, CompanyReports
from utils.data import to_db_types

report_types = [IndividualReports, CompanyReports]

def create_table(table_name: str, columns: dict[str: Any]):
    """Generates the SQL to create a table with the given column schema"""
    column_names = [f"{name} {to_db_types(typing)}" for name, typing in columns.items()]
    return f"CREATE TABLE {table_name} ({', '.join(column_names)}) ENGINE = MergeTree() ORDER BY tuple()"


@dataclass
class DatabaseInfo:
    host: str
    port: int
    username: str
    password: str


class SimDatabase:
    def __init__(self, db_name_prefix: str, db_info: DatabaseInfo):
        self.client = clickhouse_connect.get_client(**db_info.__dict__)
        # Generate a unique database name based on the current date
        db_name = f"{db_name_prefix}_{time.strftime("%Y%m%d")}"
        existing_databases = [db[0] for db in self.client.query('SHOW DATABASES').result_rows]
        idx = 1
        while f"{db_name}_{idx}" in existing_databases:
            idx += 1
        # Create the new database
        db_name = f"{db_name}_{idx}"
        self.client.command(f'CREATE DATABASE {db_name}')
        self.client.database = db_name

        # Create tables
        self.tables = {}
        for report_type in report_types:
            table_name = report_type.table_name()
            columns = list(report_type.__dataclass_fields__.keys())
            table_scheme = {col: report_type.__annotations__[col] for col in columns if col in columns}
            columns = [col for col in columns if col in table_scheme.keys()]
            self.client.command(create_table(table_name, table_scheme))
            self.tables[report_type] = columns


    def insert_reports(self, reports: list[Reports]):
        # Check that all reports are of the same type
        for rp_type, col in self.tables.items():
            if all(isinstance(r, rp_type) for r in reports):
                columns = col
                report_type = rp_type
                break
        else:
            raise ValueError("All reports must be of the same type")

        # Generate a nested list and insert the data at once
        data = [[getattr(rp, col) for col in columns] for rp in reports]
        self.client.insert(report_type.table_name(), data)


if __name__ == '__main__':
    from json import load
    with open('../db_info.json', 'r') as f:
        db_info = DatabaseInfo(**load(f))
    sim_db = SimDatabase('testing', db_info)