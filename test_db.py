import clickhouse_connect

# Connect to the ClickHouse server
client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='password')

# Test the connection
print(client.query('SELECT version()').result_rows)
# Print existing databases
print(client.query('SHOW DATABASES').result_rows)


database_name = 'test_database'
table_name = 'test_table'

# Create new database
client.command(f'CREATE DATABASE IF NOT EXISTS {database_name}')
client.database = database_name

# Create new table
client.command(f'CREATE TABLE IF NOT EXISTS {table_name} (id Int32, name String) ENGINE = Memory')

# Insert test data
data = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
client.insert(table=table_name, data=data)