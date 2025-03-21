# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /contextGame

## Copy the current directory contents into the container at /app
COPY . /contextGame

RUN #mkdir -p $APP_HOME

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK resources
RUN python -m nltk.downloader punkt stopwords

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=ContextConnect

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]