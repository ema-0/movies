# Movie classifier

## 1. How to run the application
- Make sure you have **Docker** and **Docker Compose** installed. 
You can find all the documentation [here](https://www.docker.com/).
- Clone the repo.
- Move to the repo root directory `movies`.
- Build the *Docker Image* and run the *Container*: 
    ```console 
    docker-compose up -d
  ```
- Once the container is ready, open its bash:
    ```console
    docker-compose exec movies_classifier bash
    ```
- Now you can use movie classifier application just typing:
    ```console
    movie_classier --title <title> --description <description> 
    ```
### Note
The command `movie_classifer` is an alias to use the correct Python environment.
Instead, you can use the following commands:
```console
conda activate movies 
cd /usr/src/app/movie_classifier/ 
python main.py --title <title> --description <description>
```

# 2. About the repo
The repo files are organized in 5 folders:
- `input_data`: *csv* file used to train and validate the model. 
  For sake of simplicity and because its small size, I included in the repo.
  Ideally it should be directly downloaded by the user from 
  [here](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv). 
  
    It is available to everyone but requires to be logged on Kaggle.
- `movie_classifier`: the running application. It loads the trained model and all
the other required data from `trained_model`.

- `notebooks`: the prototype of the algorithm, from data manipulation to training and testing of the 
neural network.
  
- `tests`: unit tests for the application.
- `trained_model`: written by the notebook and read by the application. It contains
the trained model, the list of accepted genres and the words tokenizer.