.PHONY: up compose morsetfe

TFE_VERSION = v3.05
BASE_VERSION = vpython3base

# run all docker
up:
	docker-compose up

# build & run all docker
compose:
	python precompile.py && docker-compose up --build



# Build Base Docker Image
# mbase:
#   docker build -t morsebase:$(BASE_VERSION) -f conf/DockerfileBase . | tail -n 1 | cut -d' ' -f3 | xargs -t -I{} docker tag {} registry.cn-hangzhou.aliyuncs.com/dtunion/morsebase:$(BASE_VERSION)
#   @echo "build base docker image ok"
#     docker push registry.cn-hangzhou.aliyuncs.com/dtunion/morsebase:$(BASE_VERSION)



# Build morsetfe Docker Image
morsetfe:
	python precompile.py
	docker build -t morsetfe:$(TFE_VERSION) -f conf/DockerfileTFE . | tail -n 1 | cut -d' ' -f3 | xargs -t -I{} docker tag {} registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:$(TFE_VERSION)
	echo "build morsetfe docker image ok"


# Push docker images
push:
	docker push registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:$(TFE_VERSION)