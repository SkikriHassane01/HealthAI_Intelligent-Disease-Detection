FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME healthai

# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]


# to build the docker image then run it folow this steps:

    # docker build -t healthai .
    # to see all the images : docker images
    # docker run -p 4000:5000 healthai
    # to see all the running containers : docker ps
    # to stop the container : docker stop container_id
    # to remove the container : docker rm container_id
    # to remove the image : docker rmi image_id

# then to deploy it in heroku folow this steps :
    
        # heroku login
        # heroku container:login
        # heroku create app_name
        # heroku stack:set container --app app_name
        # heroku container:push web --app app_name
        # heroku container:release web --app app_name
        # heroku open --app app_name
 