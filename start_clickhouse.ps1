$container_name = "clickhouse-server"
$default_user = "default"
$default_password = "password"

# Check if the container exists
$container_exists = docker ps -a --filter "name=$container_name" --format "{{.Names}}" | Select-String -Pattern $container_name

if ($container_exists) {
    # Stop and remove the container if it exists
    docker stop $container_name
    docker rm $container_name
}

# Pull the latest image
docker pull clickhouse/clickhouse-server:latest

# Run the container
docker run -d --name $container_name `
    -p 8123:8123 -p 9000:9000 -p 9009:9009 `
    --ulimit nofile=262144:262144 `
    --rm -e CLICKHOUSE_USER=$default_user -e CLICKHOUSE_PASSWORD=$default_password `
    clickhouse/clickhouse-server