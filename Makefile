


NAME   := francescsco/cnn-singularity
TAG    := v0.0.1
IMG    := ${NAME}:${TAG}
LATEST := ${NAME}:latest

build:
	docker build -t ${IMG} .
	docker tag ${IMG} ${LATEST}

push: login
	docker push ${NAME}

login:
	docker login -u ${DOCKER_USER} -p ${DOCKER_PASS}
