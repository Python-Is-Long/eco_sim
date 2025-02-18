from dataclasses import dataclass
from typing import Any, Sequence
import re

import clickhouse_connect

from utils.simulationUtils import Reports

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
        """Create a new SimDatabase instance

        Args:
            db_info (DatabaseInfo): The database connection information
            report_types (list[type[Reports]]): A list of the report types that the database will store
        """
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

            str_columns = [f"{name} {typ}" for name, typ in table_scheme.items()]
            # Create the main table
            self.client.command(f"""
                -- Create the main table
                CREATE TABLE {table_name} ({', '.join(str_columns)})
                ENGINE = ReplacingMergeTree(step) ORDER BY name
            """)
            # Create the history table
            self.client.command(f"""
                CREATE TABLE {table_name}_history ({', '.join(str_columns)})
                ENGINE = MergeTree() ORDER BY step
            """)
            # Make the history table a materialized view of the main table
            self.client.command(f"""
                CREATE MATERIALIZED VIEW {table_name}_mv TO {table_name}_history
                AS SELECT * FROM {table_name}
            """)
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
        # Get the list of existing databases
        existing_db = self.client.query('SHOW DATABASES').result_rows
        existing_db = [db[0] for db in existing_db]

        if not exact_name:
            # Find the database with the highest idx
            filtered_dbs = [db for db in existing_db if re.match(rf"^{db_name}_\d+$", db)]
            if not filtered_dbs:
                raise ValueError(f"Database starting with {db_name} not found in the server")
            db_name = max(filtered_dbs, key=lambda x: int(x.split('_')[-1]))

        if db_name not in existing_db:
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

    def insert_reports(self, reports: Reports | Sequence[Reports]):
        """Insert reports into the database.
        If inserting multiple reports, they must all be of the same type.
        """
        self._check_loaded_db()

        if isinstance(reports, Reports):
            reports = [reports]

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
    ind_name, comp_name = '4327f758-8aa1-410f-aa4a-596c03f702b9', '37017488-cdb1-4a0e-bd1b-8605f90d1b6c'

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
    sim_db.load_database('testing')

    sim_db.insert_reports([ind_reports])
    sim_db.insert_reports([comp_reports])

    print(sim_db.get_reports(IndividualReports, 1))
    print(sim_db.get_reports(CompanyReports, 1))