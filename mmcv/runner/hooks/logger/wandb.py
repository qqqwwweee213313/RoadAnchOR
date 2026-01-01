import os.path as osp

from mmcv.utils import TORCH_VERSION, digit_version, master_only
from ..hook import HOOKS
from .base import LoggerHook

@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    def __init__(self, 
                 init_kwargs=None,
                 interval=50,
                 log_model=False,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 **kwargs):
        super(WandbLoggerHook, self).__init__(
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            by_epoch=by_epoch
        )
        self.interval = interval
        self.log_model = log_model
        self.init_kwargs = init_kwargs or {}
        self.import_wandb()

    def import_wandb(self):
        """Import wandb here, not at module import time: the logger package is
        pulled in by every ``import mmcv.runner``, and wandb is optional."""
        try:
            import wandb
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb
        
    @master_only
    def before_run(self, runner):
        if self.wandb.run is None:
            init_kwargs = self.init_kwargs.copy()
            
            # disable console capture
            if 'settings' not in init_kwargs:
                init_kwargs['settings'] = self.wandb.Settings(
                    console='off',  # do not mirror terminal output
                    _disable_stats=True,  # do not collect system stats
                )
                
            self.wandb.init(**self.init_kwargs)
    
    @master_only
    def log(self, runner):
        """
        Override LoggerHook.log(); it is called automatically every `interval`
        iterations.
        """
        log_dict = {}
        
        # Training metrics
        for key, val in runner.log_buffer.output.items():
            if isinstance(val, (int, float)):
                log_dict[f'train/{key}'] = val
            elif hasattr(val, 'item'):
                log_dict[f'train/{key}'] = val.item()
        
        # Learning rate
        if hasattr(runner, 'current_lr'):
            log_dict['train/lr'] = runner.current_lr()[0]
        
        # Epoch/Iter info
        log_dict['train/epoch'] = runner.epoch
        self.wandb.log(log_dict, step=runner.epoch)

        log_dict['train/iter'] = runner.iter
        self.wandb.log(log_dict, step=runner.iter)
    
    @master_only
    def after_train_epoch(self, runner):
        log_dict = {}
        for key, val in runner.log_buffer.output.items():
            if isinstance(val, (int, float)):
                log_dict[f'train_epoch/{key}'] = val
            elif hasattr(val, 'item'):
                log_dict[f'train_epoch/{key}'] = val.item()
        
        self.wandb.log(log_dict, step=runner.epoch)
    
    @master_only
    def after_val_epoch(self, runner):
        log_dict = {}
        for key, val in runner.log_buffer.output.items():
            if isinstance(val, (int, float)):
                log_dict[f'val/{key}'] = val
            elif hasattr(val, 'item'):
                log_dict[f'val/{key}'] = val.item()
        
        self.wandb.log(log_dict, step=runner.epoch)
    
    @master_only
    def after_run(self, runner):
        if self.log_model and self.wandb.run is not None:
            self.wandb.save(runner.work_dir + '/*.pth')
        if self.wandb.run is not None:  # guard against an uninitialised run
            self.wandb.finish()