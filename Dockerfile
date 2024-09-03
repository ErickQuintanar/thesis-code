FROM pennylaneai/pennylane:v0.37.0-lightning-gpu
#FROM pennylaneai/pennylane:v0.37.0-lightning-qubit

COPY pipeline/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# docker build . -t erick/pennylane-gpu
# docker run -it --name test -v $(pwd):/Code erick/pennylane-gpu:latest bash