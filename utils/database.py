import time
from dataclasses import dataclass
from typing import Any

import clickhouse_connect

from utils.simulationObjects import Reports

def create_table(table_name: str, columns: dict[str, Any]):
    """Generates the SQL to create a table with the given column schema"""
    str_columns = [f"{name} {typ}" for name, typ in columns.items()]
    return f"CREATE TABLE {table_name} ({', '.join(str_columns)}) ENGINE = MergeTree() ORDER BY tuple()"


@dataclass
class DatabaseInfo:
    host: str
    port: int
    username: str
    password: str


class SimDatabase:
    def __init__(self, db_info: DatabaseInfo, report_types: list[type[Reports]]):
        self.client = clickhouse_connect.get_client(**db_info.__dict__)
        self.report_types = report_types
        self.tables = {}
        self._db_loaded = False

    def _check_loaded_db(self):
        if not self._db_loaded:
            raise RuntimeError("No database loaded. Use create_database or load_database first")

    def create_database(self, db_name: str, exact_name: bool=False) -> str:
        """Create a new database with the given name + idx. Then creates tables inside for each of the report types.

        Args:
            db_name (str): The name of the database to create
            exact_name (bool): If True, create a database with the exact name, WARNING will overwrite existing databases!
        """
        if exact_name:
            self.client.command(f'DROP DATABASE IF EXISTS {db_name}')
            self.client.command(f'CREATE DATABASE {db_name}')
        else:
            # Generate a unique database name based on the current date
            existing_databases = [db[0] for db in self.client.query('SHOW DATABASES').result_rows]
            idx = 1
            while f"{db_name}_{idx}" in existing_databases:
                idx += 1
            # Create the new database
            db_name = f"{db_name}_{idx}"
            self.client.command(f'CREATE DATABASE {db_name}')
        self.client.database = db_name

        # Create tables
        for report_type in self.report_types:
            table_name = report_type.table_name()
            table_scheme = report_type.get_db_types()
            self.client.command(create_table(table_name, table_scheme))
            self.tables[report_type] = list(table_scheme.keys())
        self._db_loaded = True
        return db_name

    def load_database(self, db_name: str, exact_name: bool = False):
        """Load an existing database with the given name. The database must already exist and contain the expected tables,
        which is determined by the report_types provided in the constructor.

        Args:
            db_name (str): The name of the database to load
            exact_name (bool): If True, load a database with the exact name, otherwise search for a database with the given name + highest idx

        Raises:
            ValueError: If the database or any of the tables are not found or the tables does not match the expected schema
        """
        existing_db = self.client.query('SHOW DATABASES').result_rows
        if not exact_name:
            # Find the database with the highest idx
            db_name = max((db[0] for db in existing_db if db[0].startswith(db_name)), key=lambda x: int(x.split('_')[-1]))
        if db_name not in (db[0] for db in existing_db):
            raise ValueError(f"Database {db_name} not found in the server")

        self.client.database = db_name
        for report_type in self.report_types:
            table_name = report_type.table_name()
            expected_schema = report_type.get_db_types()  # { column_name: type_str, ... }

            # Query the database for the table columns
            query = f"""
                SELECT columns.column_name, columns.data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = '{db_name}'
            """
            result = self.client.query(query)
            if not result.row_count:
                raise ValueError(f"Table {table_name} not found in the database {db_name}")

            # Check that the columns match the expected schema
            for col_name, col_type in result.result_rows:
                if col_name not in expected_schema or expected_schema[col_name] != col_type:
                    raise ValueError(
                        f"Schema mismatch for table {table_name}. Name and type must both match. "
                        f"Unexpected column: {col_name} ({col_type})"
                    )
            self.tables[report_type] = list(expected_schema.keys())
        self._db_loaded = True

    def insert_reports(self, reports: list[Reports]):
        self._check_loaded_db()
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

    def get_reports(self, report_type: type[Reports], step: int) -> list[Reports]:
        self._check_loaded_db()
        # Retrieve column names for the report type
        columns = self.tables.get(report_type)
        if not columns:
            raise ValueError("Report type not found in the database tables.")
        # Query the database
        query = f"SELECT {', '.join(columns)} FROM {report_type.table_name()} WHERE step = {step}"
        result = self.client.query(query)
        rows = result.result_rows
        # Convert each row into a Reports instance
        return [report_type(**dict(zip(columns, row))) for row in rows] # type: ignore


if __name__ == '__main__':
    from json import load
    from simulationObjects import IndividualReports, CompanyReports
    from uuid import uuid4

    ind_name, comp_name = str(uuid4()), str(uuid4())

    ind_reports = IndividualReports(
        step=1,
        name=ind_name,
        funds=2000,
        income=1000,
        expenses=500,
        salary=200,
        talent=100,
        risk_tolerance=1.2,
        owning_company=[comp_name],
    )

    comp_reports = CompanyReports(
        step=1,
        name=comp_name,
        owner=ind_name,
        funds=10000,
        employees=[],
        product='Some Product',
        costs=2000,
        revenue=3000,
        profit=1000,
        dividend=100,
        bankruptcy=False
    )

    print(ind_reports)
    print(comp_reports)

    # quit()
    with open('../db_info.json', 'r') as f:
        db_info = DatabaseInfo(**load(f))
    sim_db = SimDatabase(db_info, [IndividualReports, CompanyReports])
    sim_db.load_database('testing_1')

    sim_db.insert_reports([ind_reports])
    sim_db.insert_reports([comp_reports])

    print(sim_db.get_reports(IndividualReports, 1))
    print(sim_db.get_reports(CompanyReports, 1))