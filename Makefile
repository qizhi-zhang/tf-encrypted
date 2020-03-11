.PHONY: up compose morsetfe

TFE_VERSION = v2

# run all docker
up:
	docker-compose up

# build & run all docker
compose:
	python precompile.py && docker-compose up --build

# Build smodel Docker Image
morsetfe:
	python precompile.py
	docker build -t morsetfe:$(TFE_VERSION) -f conf/DockerfileTFE . | tail -n 1 | cut -d' ' -f3 | xargs -t -I{} docker tag {} registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:$(TFE_VERSION)
	@echo "build morsetfe docker image ok"


# Push docker images
push:
	docker push registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:$(TFE_VERSION)