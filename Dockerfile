# Use the official RunPod PyTorch image with CUDA 12 support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code into the container
COPY handler.py .

# Run the handler when the container starts
# The '-u' flag ensures logs are printed to the console in real-time
CMD [ "python", "-u", "handler.py" ]
