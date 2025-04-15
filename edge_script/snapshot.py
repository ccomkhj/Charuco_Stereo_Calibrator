import paramiko
from scp import SCPClient, SCPException
import os
import time


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def execute_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    errors = stderr.read().decode()
    if errors:
        print(f"Error executing command: {errors}")
        return None
    return stdout.read().decode()


def download_image(ssh_client, remote_file, local_dir):
    if remote_file:
        print(f"Downloading {remote_file}...")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.get(remote_file, local_path=local_dir)


def main():

    # SSH connection details - Update these as needed
    hostname = "192.168.0.105"
    port = 22  # Default SSH port
    username = "pi"  # SSH username
    password = "pw"  # SSH password
    unique_id = "test_unique_id"

    # Local directory where downloaded images will be saved
    base_local_dir = os.path.join("./downloaded_images/v1", unique_id)
    os.makedirs(base_local_dir, exist_ok=True)

    # Get the current epoch time to uniquely identify the files
    epoch = int(time.time())

    try:
        # Establish SSH connection
        ssh_client = create_ssh_client(hostname, port, username, password)

        # Trigger the snapshot_input.bash script with epoch and unique_id as arguments
        print("Executing snapshot script...")
        execute_command(
            ssh_client, f"bash /home/pi/take_snap_shot.bash {epoch} {unique_id}"
        )
        time.sleep(5)  # Increase delay for processing time

        # Define the paths to check images
        for side in ["left", "right"]:
            remote_dir = f"/home/pi/{unique_id}/{side}"
            remote_file = f"{remote_dir}/{unique_id}_{epoch}_{side}.jpg"

            local_subdir = os.path.join(base_local_dir, side)
            os.makedirs(local_subdir, exist_ok=True)

            download_image(ssh_client, remote_file, local_subdir)

        ssh_client.close()
        print("Images downloaded successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
