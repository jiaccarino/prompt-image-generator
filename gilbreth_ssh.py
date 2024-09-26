import paramiko
import sys
import os
from dotenv import load_dotenv
import time
import io

def load_env_variables():
    load_dotenv()
    username = os.getenv('GILBRETH_USERNAME')
    ssh_key = os.getenv('SSH_KEY')
    if not username or not ssh_key:
        print("Please ensure GILBRETH_USERNAME and SSH_KEY are set in your .env file")
        sys.exit(1)
    return username, ssh_key

def ssh_connect(hostname, username, ssh_key):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Create a file-like object from the SSH key string
        key_file = io.StringIO(ssh_key)
        pkey = paramiko.RSAKey.from_private_key(key_file)
        
        client.connect(hostname, username=username, pkey=pkey)
        return client
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your credentials and Duo Mobile.")
        sys.exit(1)
    except paramiko.SSHException as ssh_exception:
        print(f"Unable to establish SSH connection: {ssh_exception}")
        sys.exit(1)

def handle_duo_push(client):
    print("Duo Mobile push notification sent. Please approve it on your device.")
    time.sleep(10)  # Adjust this delay as needed
    
    try:
        client.exec_command('echo "Authentication successful"', timeout=5)
        print("Duo authentication successful.")
        return True
    except paramiko.SSHException:
        print("Duo authentication failed or timed out. Please try again.")
        return False

def execute_command(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())

def main():
    hostname = "gilbreth.rcac.purdue.edu"
    username, ssh_key = load_env_variables()
    
    client = ssh_connect(hostname, username, ssh_key)
    
    if not handle_duo_push(client):
        client.close()
        sys.exit(1)

    print(f"Connected to {hostname}")
    
    while True:
        command = input("Enter command to execute (or 'exit' to quit): ")
        if command.lower() == 'exit':
            break
        execute_command(client, command)

    client.close()

if __name__ == "__main__":
    main()
