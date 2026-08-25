#!/usr/bin/env python3
"""
Fan Fault Detection - Main Entry Point
"""
import os
import sys
import argparse
import subprocess

def run_train(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "src.train"], check=True)

def run_inference(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "src.inference", "--model", args.model, "--audio", args.audio]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.device:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, check=True)

def run_api(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["MODEL_PATH"] = args.model
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "src.api:app",
        "--host", args.host,
        "--port", str(args.port),
        "--reload"
    ], check=True)

def run_frontend(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    subprocess.run([
        sys.executable, "-m", "http.server", 
        str(args.port),
        "--directory", frontend_dir
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Fan Fault Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    train_parser = subparsers.add_parser("train", help="Train the model")
    
    infer_parser = subparsers.add_parser("infer", help="Run inference on audio file")
    infer_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--audio", required=True, help="Path to audio file or directory")
    infer_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    infer_parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument("--model", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    frontend_parser = subparsers.add_parser("frontend", help="Start frontend server")
    frontend_parser.add_argument("--port", type=int, default=3000, help="Port number")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "frontend":
        run_frontend(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()