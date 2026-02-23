import pytest
import subprocess
import time
import os
import sys

def test_app_startup_no_runtime_errors():
    """
    Tests that the Streamlit application starts without immediate runtime errors.
    This test runs Streamlit in headless mode and checks for a successful startup message.
    """
    command = [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    process = None
    try:
        # Start the Streamlit app as a subprocess
        # Use Popen and communicate to control the process
        # Using universal_newlines=True for text output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        startup_successful = False
        start_time = time.time()
        timeout = 30  # seconds to wait for successful startup message

        while time.time() - start_time < timeout:
            # Read stdout and stderr without blocking
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if "You can now view your Streamlit app in your browser." in stdout_line:
                startup_successful = True
                break
            
            # Check if the process exited prematurely
            if process.poll() is not None:
                # Process exited, something went wrong
                stdout, stderr = process.communicate() # Get remaining output
                pytest.fail(f"Streamlit app exited prematurely with code {process.returncode}.
STDOUT:
{stdout}
STDERR:
{stderr}")

            time.sleep(0.5) # Wait a bit before checking again

        if not startup_successful:
            stdout, stderr = process.communicate() # Get remaining output
            pytest.fail(f"Streamlit app did not start successfully within {timeout} seconds.
STDOUT:
{stdout}
STDERR:
{stderr}")

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

