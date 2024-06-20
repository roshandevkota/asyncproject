
# Machine Learning Web Application

  

This project is a web application built with Django for the backend and React for the frontend. It allows users to upload data, train machine learning models, and make predictions.

  

## Overview

  

-  **Backend:** Django

-  **Frontend:** React

-  **Machine Learning:** CatBoost, Optuna

  

## Setup

  

### Backend

  

#### Prerequisites

  

- Python 3.6 or later

- pip

  

#### Installation

  

1.  **Clone the repository:**

  

```sh

git clone https://github.com/roshandevkota/asyncproject

cd <REPOSITORY_NAME>

```

  

2.  **Create a virtual environment:**

  

```sh

python -m venv venv

source venv/bin/activate # On Windows use `venv\Scripts\activate`

```

  

3.  **Install dependencies:**

  

```sh

pip install -r requirements.txt

```

  

4.  **Run migrations:**

  

```sh

python manage.py migrate

```

  

5.  **Start the development server:**

  

```sh

daphne -p 8000 asyncproject.asgi:application

```
or
  ```sh

python manage.py runserver

```

#### Packages

  

Here are the main packages used in the backend:

  

- Django

- djangorestframework

- pandas

- numpy

- catboost

- optuna

- simplejson

- joblib

- daphne
- celery (not used)

  

### Frontend

  

#### Prerequisites

  

- Node.js

- npm

  

#### Installation

  

1.  **Navigate to the `frontend` directory:**

  

```sh

cd frontend

```

  

2.  **Install dependencies:**

  

```sh

npm install

```

  

3.  **Start the development server:**

  

```sh

npm start

```

  

#### Packages

  

Here are the main packages used in the frontend:

  

- React

- axios

- react-bootstrap

- js-cookie

  

## Usage

  

-  **Backend API:**  `http://localhost:8000`

-  **Frontend App:**  `http://localhost:3000`