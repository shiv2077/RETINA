# RETINA framework documentation


## Getting started
The RETINA Framework is built in Python (tested on Python version 3.12).
It also depends on the followıng supporting packages/programs (see installation instructions below): Redis, PostgreSQL

### Backend (directory: back_end)
To ensure that you have the required libraries installed, install the packages mentioned in the requirements.txt file.
```
pip install -r back_end/requirements.txt
```
Also ensure that your PYTHONPATH environment variable is correctly initialized relative to the repository:
```
~/<your retina repository directory>/back_end $ export PYTHONPATH=$PWD
```
Tools that is needed to boot the application successfully are as follows, since the app is run locally the tools are connected by the *localhost* URL's:
* **Redis**
Prerequisite: Docker Installation
```
docker network create mv4qc-net
```
Network is useful when there is multiple containers from the *localhost*
```
docker run -d --name redis ^
--network mv4qc-net ^
-p 6379:6379 ^
redis:7-alpine
```
See the running images in Docker and ensure if the redis is on:
```
docker ps
docker exec -it redis redis-cli PING
```
* **PostgreSQL**:
Ensure the port 5432 is not in use, not conflicting service.
```
netstat -ano | findstr :5432
```
Prerequisite: Docker Installation
```
docker run -d --name pg \
  -e POSTGRES_USER=mv4qc_user \
  -e POSTGRES_PASSWORD=mv4qc123 \
  -e POSTGRES_DB=mv4qc_db \
  -p 5432:5432 \
  -v pg_data:/var/lib/postgresql/data \
  postgres:16-alpine
```
See the running images:
```
docker ps
```

There are two entry points:
* The main back-end application gets run by calling
```
python -m uvicorn app.main:app
```
* The supervised pipeline in the application gets run by calling
```
python -m app.worker
```

### Frontend (directory: front_end)
Install the packages mentioned in the requirements.txt file.
```
pip install -r back_end/requirements.txt
```
There is one entry point:
```
streamlit run 0_Home.py
```
### Which components to swap out and how

* Unsupervised learning classifier
All classifier computer vision models' pt / pth files take place in the following directory:
`back_end/app/models/`
In this directory, relative files of supervised and unsupervised models are contained.
    * **Unsupervised**
    `~/unsupervised/output/`
    patchcore and padim .pt files are kept inside and can be replaced with anew one.

* Supervised learning classifier
* Active learning component
Active Learning Module has its client file kept in the `~/app/` directory. Likewise, the server side of the module is another application and takes place in the `back_end/` directory. It is designed as a dummy server and can be replaced with a proper server side design.
```

```
