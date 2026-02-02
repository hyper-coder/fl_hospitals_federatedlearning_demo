import sys
import subprocess

def ping_site():
    # 1. Get input directly from the user (Untrusted Source)
    # CodeQL flags this because it enters the system here.
    target = input("Enter website to ping: ")
    
    # 2. Pass it directly to the shell (The Vulnerability)
    # checks for "subprocess.call" with "shell=True" using a variable
    print(f"Pinging {target}...")
    subprocess.call("ping -c 1 " + target, shell=True)

if __name__ == "__main__":
    ping_site()