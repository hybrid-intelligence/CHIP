FROM python:3.10-slim

# Install python deps
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

# Copy over source
COPY app /app

# Run the server
CMD [ "flask", "--app", "app", "--debug", "run", "--host", "0.0.0.0"]
