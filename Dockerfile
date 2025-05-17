FROM python:3.12-slim

WORKDIR /app

# install make
RUN apt-get update && apt-get install -y make

# Copy requirements first for better caching
COPY requirements.txt .

COPY .env .

# Copy makefile
COPY Makefile . 

# Install dependencies
RUN make install

# Copy the rest of the application
COPY . .


# Create an .env file with defaults if it doesn't exist
RUN touch .env

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Expose the port specified in the application
EXPOSE 8000

# Command to run the application
CMD ["make", "run"]