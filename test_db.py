import clickhouse_connect

# Connect to the ClickHouse server
client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='password')

# Test the connection
print('Database version:')
print(client.query('SELECT version()').result_rows, '\n')
# Print existing databases
print('Existing databases:')
print(client.query('SHOW DATABASES').result_rows, '\n')

database_name = 'test_database'
table_name = 'test_table'

print(f'Creating database({database_name}) and table({table_name})...\n')
# Create new database
client.command(f'CREATE DATABASE IF NOT EXISTS {database_name}')
client.database = database_name

# Create new table
client.command(f'CREATE TABLE IF NOT EXISTS {table_name} (id Int32, name String) ENGINE = Memory')

# Query table schema
print('Table Schema:')
query_result = client.query(f"SELECT columns.column_name, columns.data_type FROM information_schema.columns WHERE table_name = 'test_table'")
print(query_result.column_names, '\n')
print(query_result.column_types, '\n')

# Insert test data
print('Inserting data...\n')
data = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
client.insert(table=table_name, data=data)

# Query the table
query = f'SELECT * FROM {table_name}'
result = client.query(query)
print('Query Results:')
print(result.result_rows)