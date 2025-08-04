# Load configuration
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

print(config["data"]["path"])

test_size=config["training"]["test_size"]
print(test_size)
random_state=config["training"]["random_state"]
print(random_state)

docker_image_Path=config['docker']['docker_images_path']
docker_image_name=config['docker']['docker_image_name']
print(docker_image_Path)
print(docker_image_name)