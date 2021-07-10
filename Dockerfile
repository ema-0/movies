FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY movies.yml .
RUN conda env create -f movies.yml
COPY movie_classifier ./movie_classifier/

RUN echo 'alias movie_classifier="/opt/conda/envs/movies/bin/python /usr/src/app/movie_classifier/main.py"' >> ~/.bashrc
