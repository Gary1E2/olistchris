# specify your base image to be jupyter
FROM jupyter/base-notebook

# set working directory to be /usr/src/app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . /usr/src/app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose flask app on port 8881
EXPOSE 8888

# run the command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]