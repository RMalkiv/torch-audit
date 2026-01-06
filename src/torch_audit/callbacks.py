from .core.auditor import Auditor

try:
    from lightning.pytorch.callbacks import Callback as PLCallback

    class LightningAuditCallback(PLCallback):
        def __init__(self, auditor: Auditor):
            self.auditor = auditor

        def on_fit_start(self, trainer, pl_module):
            """Run static analysis (architecture, weights) before training loop."""
            self.auditor.audit_static()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if (self.auditor._step_count % self.auditor.config.interval == 0):
                self.auditor.audit_data(batch)

            self.auditor.start_dynamic_audit()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.auditor.stop_dynamic_audit()

except ImportError:
    LightningAuditCallback = None

try:
    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


    class HFAuditCallback(TrainerCallback):
        def __init__(self, auditor: Auditor):
            self.auditor = auditor

        def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            inputs = kwargs.get('inputs')
            if inputs is not None and (self.auditor._step_count % self.auditor.config.interval == 0):
                self.auditor.audit_data(inputs)

            self.auditor.start_dynamic_audit()

        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self.auditor.stop_dynamic_audit()

except ImportError:
    HFAuditCallback = None
