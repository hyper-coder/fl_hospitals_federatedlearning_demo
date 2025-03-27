import os
import time
import json
import requests
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Also suppress from environment (in case subprocesses are involved)
os.environ["PYTHONWARNINGS"] = "ignore"

# ANSI Colors for Formatting
GREEN = "\033[92m"
BLUE = "\033[97m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def draw_line(len):
    print(f"{GREEN}" + "*" * len + RESET)


# Clear screen function
def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")

# Fancy CLI intro
def print_intro():
    clear_screen()
    print(f"\n{YELLOW}_______ THE SMALL WALL PODCAST  _______{RESET}")
    draw_line(50)
    print(f"{GREEN}************* FEDERATED LEARNING DEMO *************{RESET}")
    print(f"{BLUE}Description:{RESET} This demo simulates how multiple hospitals train AI models locally while keeping patient data private.")
    print(f"{BLUE}Run Date:{RESET} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    draw_line(50)
    input(f"\n{BLUE}Press Enter to begin the demo...{RESET}")
    #clear_screen()

# Print step with explanation
def print_step(message):
    print(f"{YELLOW}\n[Step] {message}...{RESET}")
    input(f"{BLUE}Press Enter to continue...{RESET}")
    #clear_screen()

# Run a shell command with a description
def run_command(command, description):
    print(f"{GREEN}\n▶ {description}{RESET}")
    time.sleep(1)
    os.system(command)

# Run the demo
def main():
    print_intro()

    # Step 1: Ensure Aggregation Server is Running
    print(f"{RED}🚨 IMPORTANT: Ensure the Aggregation Server is Running 🚨{RESET}")
    print(f"{BLUE}Run this command in another terminal before proceeding:{RESET}")
    print(f"{GREEN}python aggregator/server.py{RESET}\n")
    input("Press Enter once the server is running...")
    #clear_screen()

    # Step 2: Train Models for Each Hospital
    print_step("Training AI Model for Hospital 1")
    run_command("python hospitals/hospital_1.py 1", "Training Model for Hospital 1")

    print_step("Training AI Model for Hospital 2")
    run_command("python hospitals/hospital_2.py 2", "Training Model for Hospital 2")

    print_step("Training AI Model for Hospital 3")
    run_command("python hospitals/hospital_3.py 3", "Training Model for Hospital 3")

    # Step 3: Show Available Weights
    print_step("Checking Saved Model Weights")
    run_command("ls models/hospital_*_weights.pth", "Listing trained weights")

    # Step 4: Aggregate Weights
    print_step("Aggregating Weights to Create a Global Model")
    run_aggregation()

    # Step 5: Show Global Model
    print_step("Checking Aggregated Global Model")
    run_command("ls models/global_model.pth", "Checking if global model exists")

    # Step 6: Hospitals Download Updated Model
    print_step("Global Model for Hospitals is now ready")
    #run_command("python download_model.py", "Downloading and Applying the Global Model")

    # End of Demo
    #clear_screen()
    draw_line(50)
    print(f"{GREEN}\n✅ Federated Learning Demo Completed Successfully!{RESET}\n")
   



def run_aggregation():
    print(f"{GREEN}\n▶ Performing Federated Model Aggregation...{RESET}")
    try:
        response = requests.get("http://127.0.0.1:5000/aggregate")
        if response.status_code == 200:
            data = response.json()
            print(f"{BLUE}\n✅ Aggregation Result:{RESET}")
            print(json.dumps(data, indent=2))
        else:
            print(f"{RED}❌ Aggregation failed with status {response.status_code}{RESET}")
            print(response.text)
    except Exception as e:
        print(f"{RED}❌ Error connecting to aggregator: {e}{RESET}")


# Run the script
if __name__ == "__main__":
    main()