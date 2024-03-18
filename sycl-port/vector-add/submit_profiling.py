import os
import argparse

def main(args):
    gpu_script = "run_geforce.sh" if args.gpu == "GeForce" else "run_a100.sh"
    if args.jobname:
        os.system(f"sbatch -J {args.jobname} --export=DIR={args.dir},ITERS={args.iters},FILE={args.file} {gpu_script}")
    else:
        os.system(f"sbatch --export=DIR={args.dir},ITERS={args.iters},FILE={args.file} {gpu_script}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an sbatch script to time the execution of a specified vector addition code.")
    parser.add_argument(
        "--iters", type=int, help="Number of times to run the code."
        ,required=True
    )
    parser.add_argument(
        "--dir", type=str, help="Location of code to compile and run."
        ,required=True
    )
    parser.add_argument(
        "--file", type=str, help="C++ file to compile and run."
        ,required=True
    )
    parser.add_argument(
        "--gpu", choices={"A100", "GeForce"}, help="GPU to run on."
        ,required=True
    )
    parser.add_argument(
        "--jobname", type=str, help="(Optional) specify Slurm job name."
    )
    main(args=parser.parse_args())
