import subprocess
import time
import requests
import socket
import pytest
import os

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def test_app_startup():
    port = 8501
    timeout = 30 # seconds
    
    # Run streamlit in headless mode
    process = None
    try:
        process = subprocess.Popen(
            ["python", "-m", "streamlit", "run", "src/app.py", "--server.port", str(port), "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        start_time = time.time()
        startup_successful = False
        
        while time.time() - start_time < timeout:
            if is_port_in_use(port):
                # Attempt to ping the app
                try:
                    response = requests.get(f"http://localhost:{port}")
                    if response.status_code == 200:
                        startup_successful = True
                        break
                except requests.exceptions.ConnectionError:
                    pass
            
            # Check if the process exited prematurely
            if process.poll() is not None:
                # Process exited, something went wrong
                stdout, stderr = process.communicate() # Get remaining output
                pytest.fail(f"Streamlit app exited prematurely with code {process.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

            time.sleep(0.5) # Wait a bit before checking again

        if not startup_successful:
            stdout, stderr = process.communicate() # Get remaining output
            pytest.fail(f"Streamlit app did not start successfully within {timeout} seconds.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    except Exception as e:
        pytest.fail(f"An unexpected error occurred during Streamlit app startup test: {e}")
    finally:
        if process and process.poll() is None:
            # Terminate the process if it's still running
            process.terminate()
            process.wait(timeout=5) # Wait for it to terminate
            if process.poll() is None:
                process.kill() # Force kill if still not terminated
        # Give some time for the port to be released
        time.sleep(1)
