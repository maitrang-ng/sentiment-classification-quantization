# set base image (host OS)
FROM python:3.9

MAINTAINER MaiTrang<nguyen.maitrang.2406@gmail.com>

# set the working directory in the container
WORKDIR app

# copy the requirements file to the working directory
COPY requirements.txt .

# install requirements
RUN pip install -r requirements.txt

# create src directory & copy the content of the local src directory to it
RUN mkdir -p app/src
COPY src/  app/src

# create model directory & copy the content of the local src directory to it
RUN mkdir -p app/models
COPY models/  app/models

# create data directory & copy the content of the local src directory to it
RUN mkdir -p app/data
COPY dataset/  app/data

# copy the content of the local src directory to the working directory
#COPY src/ .

# copy datasets & model to the working directory
#COPY model/ .
#COPY dataset/ .

# command to run on container start
WORKDIR app/src
CMD ["python", "server.py"]