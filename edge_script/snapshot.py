import paramiko
from scp import SCPClient, SCPException
import os
import time
from typing import Optional
import logging
from stat import S_ISREG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ssh_download.log")],
)
logger = logging.getLogger(__name__)


class SSHDownloader:
    def __init__(
        self,
        hostname: str,
        port: int,
        username: str,
        password: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ssh_client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Establish SSH connection with retries"""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Attempting SSH connection (attempt {attempt}/{self.max_retries})"
                )
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.load_system_host_keys()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    self.hostname,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    timeout=30,
                    banner_timeout=30,
                )
                logger.info("SSH connection established successfully")
                return
            except Exception as e:
                logger.error(f"SSH connection failed (attempt {attempt}): {str(e)}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)
                self.close()

    def close(self):
        """Close SSH connection if it exists"""
        if self.ssh_client:
            try:
                self.ssh_client.close()
                logger.info("SSH connection closed")
            except Exception as e:
                logger.error(f"Error closing SSH connection: {str(e)}")
            finally:
                self.ssh_client = None

    def execute_command(self, command: str) -> Optional[str]:
        """Execute remote command with retries"""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Executing command (attempt {attempt}): {command}")
                stdin, stdout, stderr = self.ssh_client.exec_command(
                    command, timeout=60
                )
                exit_status = stdout.channel.recv_exit_status()

                errors = stderr.read().decode().strip()
                if errors:
                    logger.warning(f"Command produced stderr output: {errors}")

                if exit_status != 0:
                    raise Exception(f"Command failed with exit status {exit_status}")

                output = stdout.read().decode().strip()
                return output if output else None
            except Exception as e:
                logger.error(f"Command execution failed (attempt {attempt}): {str(e)}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)

    def remote_file_exists(self, remote_path: str) -> bool:
        """Check if remote file exists and is a regular file"""
        try:
            sftp = self.ssh_client.open_sftp()
            try:
                file_stat = sftp.stat(remote_path)
                return S_ISREG(file_stat.st_mode)
            except IOError:
                return False
            finally:
                sftp.close()
        except Exception as e:
            logger.error(f"Error checking remote file existence: {str(e)}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file with verification and retries"""
        if not self.remote_file_exists(remote_path):
            logger.error(f"Remote file does not exist: {remote_path}")
            return False

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Attempting download (attempt {attempt}): {remote_path} -> {local_path}"
                )

                # Get remote file size for verification
                sftp = self.ssh_client.open_sftp()
                remote_size = sftp.stat(remote_path).st_size
                sftp.close()

                # Download the file
                with SCPClient(
                    self.ssh_client.get_transport(), socket_timeout=60
                ) as scp:
                    scp.get(remote_path, local_path=local_path)

                # Verify local file
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    if local_size == remote_size:
                        logger.info("Download completed and verified successfully")
                        return True
                    else:
                        logger.error(
                            f"File size mismatch: remote={remote_size}, local={local_size}"
                        )
                        os.remove(local_path)  # Remove corrupted file
                else:
                    logger.error("Downloaded file not found at local path")

                if attempt == self.max_retries:
                    return False

                time.sleep(self.retry_delay)
            except SCPException as e:
                logger.error(f"SCP download failed (attempt {attempt}): {str(e)}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Download failed (attempt {attempt}): {str(e)}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)

        return False


def main():
    # Configuration - Update these as needed
    config = {
        "hostname": "192.168.0.105",
        "port": 22,
        "username": "pi",
        "password": "...",
        "unique_id": "C250416_v0.1",
        "max_retries": 5,
        "retry_delay": 10,
        "command_delay": 15,  # Increased delay for processing time
    }

    # Prepare local directories
    base_local_dir = os.path.join("./downloaded_images/v0.1", config["unique_id"])
    os.makedirs(os.path.join(base_local_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(base_local_dir, "right"), exist_ok=True)

    # Get the current epoch time to uniquely identify the files
    epoch = int(time.time())

    try:
        with SSHDownloader(
            hostname=config["hostname"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"],
        ) as downloader:
            # Trigger the snapshot script
            logger.info("Executing snapshot script...")
            downloader.execute_command(
                f"bash /home/pi/take_snap_shot.bash {epoch} {config['unique_id']}"
            )
            time.sleep(config["command_delay"])

            # Download images for both sides
            success = True
            for side in ["left", "right"]:
                remote_dir = f"/home/pi/{config['unique_id']}/{side}"
                remote_file = f"{remote_dir}/{config['unique_id']}_{epoch}_{side}.jpg"
                local_subdir = os.path.join(base_local_dir, side)
                local_file = os.path.join(local_subdir, os.path.basename(remote_file))

                os.makedirs(local_subdir, exist_ok=True)

                if not downloader.download_file(remote_file, local_file):
                    logger.error(f"Failed to download {remote_file}")
                    success = False
                else:
                    logger.info(f"Successfully downloaded {remote_file}")

            if success:
                logger.info("All images downloaded successfully")
            else:
                logger.error("Some images failed to download")

    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
