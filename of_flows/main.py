import argparse
import json
from pathlib import Path
from train.loop import training
from config._config import Config

def setup_directories(args):
    """Create and return directory paths for results, checkpoints, and figures."""
    results_dir = f"Results/{args.mol_name}_{args.cc}_{args.x}_{args.c}_{args.solver}_{args.prior}"
    if args.sched.lower() not in ['c', 'const']:
        results_dir += f"_sched_{args.sched.upper()}"
    
    ckpt_dir = f"{results_dir}/Checkpoints"
    
    # Create directories
    for directory in [results_dir, ckpt_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return results_dir, ckpt_dir


def save_job_params(results_dir, args):
    """Save training parameters to JSON file."""
    job_params = {
        'mol_name': args.mol_name,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'hidden_layer': args.hl, 
        'lr': args.lr,
        'kinetic': args.kin,
        'external': args.nuc,
        'hartree': args.hart,
        'exchange': args.x,
        'correlation': args.c,
        'core_correction': args.cc,
        'scheduler': args.sched,
        'solver': args.solver,
        'prior' : args.prior,
    }
    
    with open(f"{results_dir}/job_params.json", "w") as outfile:
        json.dump(job_params, outfile, indent=4)
    
    return job_params


def main():
    parser = argparse.ArgumentParser()
    ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
    # Model parameters
    parser.add_argument("--mol_name", type=str, default='H2',
                        help="Molecule name")
    parser.add_argument("--epochs", type=int, default=5000, 
                        help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--hl", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--prior", type=str, default='db_sir',
                    choices=['promolecular', 'db_sir'],
                    help="Prior distribution type")
    
    # Functionals
    parser.add_argument("--kin", type=str, default='tf_w',
                        choices=['tf', 'w', 'tf_w'],
                        help="Kinetic energy functional")
    parser.add_argument("--nuc", type=str, default='np',
                        help="Nuclear potential functional")
    parser.add_argument("--hart", type=str, default='coulomb',
                        help="Hartree energy functional")
    parser.add_argument("--x", type=str, default='lda',
                        choices=['lda', 'b88_x'],
                        help="Exchange energy functional")
    parser.add_argument("--c", type=str, default='none',
                        choices=['vwn_c', 'pw92_c', 'none'],
                        help="Correlation energy functional")
    parser.add_argument("--cc", type=str, default='kato',
                        choices=['kato', 'none'],
                        help="Core correction functional")
    
    # Training settings
    parser.add_argument("--sched", type=str, default='mix',
                        help="Learning rate scheduler type")
    parser.add_argument("--solver", type=str, default='dopri8',
                        choices=['dopri5', 'tsit5'],
                        help="ODE solver")
    parser.add_argument("--ckpt_freq", type=int, default=50,
                        help="Checkpoint saving frequency (epochs)")
    
    args = parser.parse_args()
    
    Config.from_args(args)
    
    # Setup directories
    results_dir, ckpt_dir = setup_directories(args)
    Config.set_directories(results_dir, ckpt_dir)
    
    # Save parameters
    job_params = save_job_params(results_dir, args)
    print(f"Starting training with parameters:")
    print(json.dumps(job_params, indent=2))
    print(f"\nResults will be saved to: {results_dir}")
    
    # Run training
    model, df, df_ema = training(
        mol_name=args.mol_name,
        tw_kin=args.kin,
        n_pot=args.nuc,
        h_pot=args.hart,
        x_pot=args.x,
        c_pot=args.c,
        cc_pot=args.cc,
        batch_size=args.bs,
        hidden_layer=args.hl, 
        epochs=args.epochs,
        lr=args.lr,
        scheduler_type=args.sched,
        solver_type=args.solver,
        prior_type = args.prior,
        checkpoint_dir=ckpt_dir,
        checkpoint_freq=args.ckpt_freq,
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Final energy (EMA): {df_ema['E'].iloc[-1]:.6f}")

if __name__ == "__main__":
    main()