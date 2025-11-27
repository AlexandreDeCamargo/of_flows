"""
Global configuration module for storing runtime parameters and directories.
"""

class Config:
    """Global configuration class to store all runtime parameters."""
    
    # Model parameters
    mol_name = None
    epochs = None
    bs = None
    hl = None
    lr = None
    
    # Functionals
    kin = None
    nuc = None
    hart = None
    x = None
    c = None
    cc = None
    
    # Training settings
    sched = None
    solver = None
    ckpt_freq = None
    
    # Directories (set during runtime)
    results_dir = None
    ckpt_dir = None
    
    @classmethod
    def from_args(cls, args):
        """
        Initialize configuration from argparse arguments.
        
        Args:
            args: argparse.Namespace object containing parsed arguments
        """
        # Model parameters
        cls.mol_name = args.mol_name
        cls.epochs = args.epochs
        cls.bs = args.bs
        cls.hl = args.hl
        cls.lr = args.lr
        
        # Functionals
        cls.kin = args.kin
        cls.nuc = args.nuc
        cls.hart = args.hart
        cls.x = args.x
        cls.c = args.c
        cls.cc = args.cc
        
        # Training settings
        cls.sched = args.sched
        cls.solver = args.solver
        cls.ckpt_freq = args.ckpt_freq
    
    @classmethod
    def set_directories(cls, results_dir, ckpt_dir):
        """
        Set the results and checkpoint directories.
        
        Args:
            results_dir: Path to results directory
            ckpt_dir: Path to checkpoint directory
        """
        cls.results_dir = results_dir
        cls.ckpt_dir = ckpt_dir
    
    @classmethod
    def get_model_params(cls):
        """Return dictionary of model parameters."""
        return {
            'mol_name': cls.mol_name,
            'epochs': cls.epochs,
            'bs': cls.bs,
            'hl': cls.hl,
            'lr': cls.lr
        }
    
    @classmethod
    def get_functionals(cls):
        """Return dictionary of functional parameters."""
        return {
            'kin': cls.kin,
            'nuc': cls.nuc,
            'hart': cls.hart,
            'x': cls.x,
            'c': cls.c,
            'cc': cls.cc
        }
    
    @classmethod
    def __repr__(cls):
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  Model: mol_name={cls.mol_name}, epochs={cls.epochs}, "
            f"bs={cls.bs}, hl={cls.hl}, lr={cls.lr}\n"
            f"  Functionals: kin={cls.kin}, x={cls.x}, c={cls.c}, cc={cls.cc}\n"
            f"  Directories: results={cls.results_dir}, ckpt={cls.ckpt_dir}\n"
            f")"
        )