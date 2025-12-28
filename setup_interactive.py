import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def print_step(step):
    print(f"\n{'='*50}")
    print(f"STEP: {step}")
    print(f"{'='*50}\n")

def check_command(command):
    return shutil.which(command) is not None

def install_uv():
    print_step("Checking 'uv' installation")
    if check_command("uv"):
        print("✅ 'uv' is already installed.")
        return

    print("❌ 'uv' is not found.")
    choice = input("Do you want to install 'uv' now? (y/n): ").strip().lower()
    if choice != "y":
        print("Skipping 'uv' installation. Note: This is required for dependency management.")
        return

    try:
        if platform.system() == "Windows":
             subprocess.run(["pip", "install", "uv"], check=True)
        else:
            # Install via curl for Mac/Linux
            install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            subprocess.run(install_cmd, shell=True, check=True)
            
            # Add to PATH for current session if likely installed to ~/.cargo/bin or similar
            # This is best effort; user might need to restart shell
            print("⚠️  You may need to restart your terminal or source your profile to use 'uv'.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install 'uv': {e}")
        print("Please install manually: https://docs.astral.sh/uv/getting-started/installation/")

def check_ollama():
    print_step("Checking Ollama installation")
    if check_command("ollama"):
        print("✅ Ollama is installed.")
        return True

    print("❌ Ollama is not found.")
    if platform.system() == "Windows":
        print("Please download and install Ollama manually from: https://ollama.com/download")
        input("Press Enter after you have installed Ollama...")
    else:
        choice = input("Do you want to install Ollama via script? (y/n): ").strip().lower()
        if choice == "y":
            try:
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            except subprocess.CalledProcessError:
                print("❌ Failed to install Ollama automatically.")
                print("Please install manually: https://ollama.com/download")
    
    return check_command("ollama")

def install_dependencies():
    print_step("Installing Python Dependencies")
    if not check_command("uv"):
        print("❌ 'uv' command not found. Cannot run 'uv sync'.")
        return

    print("Running 'uv sync'...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed successfully.")
        
        print("Installing project in editable mode...")
        subprocess.run(["uv", "pip", "install", "-e", "."], check=True)
        print("✅ Project installed in editable mode.")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")

def setup_models():
    print_step("Setting up Ollama Models")
    print("This project requires specific models to be running locally.")
    print("Please open a NEW terminal window and run the following commands:")
    print("-" * 40)
    print("ollama run gpt-oss:20b")
    print("-" * 40)
    print("Note: 'ollama run' downloads and chats with the model. You can exit the chat with Ctrl+D")
    print("but ensure the models have been pulled successfully.")
    
    input("\nPress Enter once you have confirmed the models are ready...")

def setup_api_key():
    print_step("Configuring Ollama Web Search API Key")
    
    if os.environ.get("OLLAMA_API_KEY"):
        print("✅ OLLAMA_API_KEY is already set in your environment.")
        return

    print("The Ollama Python library requires an API key for web search.")
    print("You can create it by logging in to ollama and creatingthe API key here: https://ollama.com/settings/keys")
    print("(Reference: check your search provider or Ollama settings instructions if applicable)")
    
    key = input("Enter your OLLAMA_API_KEY (leave empty to skip): ").strip()
    if not key:
        print("Skipping API key configuration.")
        return

    # 1. Update .env file
    env_path = Path(".env")
    env_content = ""
    if env_path.exists():
        env_content = env_path.read_text()
    
    if "OLLAMA_API_KEY=" in env_content:
        # Simple replacement (imperfect but works for simple cases)
        lines = env_content.splitlines()
        new_lines = [l if not l.startswith("OLLAMA_API_KEY=") else f"OLLAMA_API_KEY={key}" for l in lines]
        env_content = "\n".join(new_lines)
    else:
        env_content += f"\nOLLAMA_API_KEY={key}\n"
    
    env_path.write_text(env_content)
    print("✅ Saved to .env file.")

    # 2. Persist to System Env
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["setx", "OLLAMA_API_KEY", key], check=True)
            print("✅ Saved to user environment variables (Windows).")
        else:
            shell = os.environ.get("SHELL", "")
            rc_file = None
            if "zsh" in shell:
                rc_file = Path.home() / ".zshrc"
            elif "bash" in shell:
                rc_file = Path.home() / ".bashrc"
            
            if rc_file:
                with open(rc_file, "a") as f:
                    f.write(f"\nexport OLLAMA_API_KEY='{key}'\n")
                print(f"✅ Appended export to {rc_file}")
                print("⚠️  Run 'source " + str(rc_file) + "' or restart terminal to apply.")
            else:
                print(f"⚠️  Could not detect shell profile. Please manually add: export OLLAMA_API_KEY='{key}'")
                
    except Exception as e:
        print(f"❌ Failed to set system environment variable: {e}")

def run_app():
    print_step("Launch Application")
    choice = input("Do you want to run the Streamlit app now? (y/n): ").strip().lower()
    if choice == "y":
        try:
            cmd = ["uv", "run", "streamlit", "run", "src/web_app/streamlit_deepresearch_chat_app.py"]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Application crashed: {e}")
        except KeyboardInterrupt:
            print("\nApplication stopped.")

def main():
    print("\n🚀 OpenDeepResearch Interactive Setup 🚀\n")
    
    check_ollama()
    install_uv()
    install_dependencies()
    setup_models()
    setup_api_key()
    run_app()

    print("\n✅ Setup Complete! Happy Researching! 🕵️‍♂️")

if __name__ == "__main__":
    main()
