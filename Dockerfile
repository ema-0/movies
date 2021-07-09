FROM continuumio/miniconda3
#FROM python:3.9.6

WORKDIR /usr/src/app

COPY movies.yml .
RUN conda env create -f movies.yml
#RUN pip install conda && conda env create -f movies.yml
#SHELL ["conda", "run", "-n", "movies", "/bin/bash", "-c"]

#RUN which python

COPY movie_classifier ./movie_classifier/
RUN echo 'alias movie_classifier="/opt/conda/envs/movies/bin/python /usr/src/app/movie_classifier/main.py"' >> ~/.bashrc
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "movies", "python", "movie_classifier/main.py --title aaa -- description bbb"]