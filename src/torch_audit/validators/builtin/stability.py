from collections.abc import Generator

import torch

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...utils.torch_safe import isfinite_all, safe_norm, tensor_summary
from ...validator import BaseValidator

# --- Rules ---

TA100_NAN_INF = Rule(
    id="TA100",
    title="NaN or Inf Detected",
    description="Parameters or gradients contain invalid values (NaN/Inf).",
    remediation="Check learning rate, loss scaling, or initialization.",
    category="Stability",
    default_severity=Severity.ERROR,
)

TA102_GRAD_EXPLOSION = Rule(
    id="TA102",
    title="Gradient Explosion",
    description="Global gradient norm is excessively high.",
    remediation="Check learning rate or apply `torch.nn.utils.clip_grad_norm_`.",
    category="Stability",
    default_severity=Severity.WARN,
)

TA103_DEAD_UNITS = Rule(
    id="TA103",
    title="Dead Units (Zero Grads)",
    description="Parameter gradients are entirely zero for the batch (Dead Neurons).",
    remediation="Check initialization (e.g. ReLU dead units) or data flow.",
    category="Stability",
    default_severity=Severity.INFO,
)

TA104_NO_GRADS = Rule(
    id="TA104",
    title="No Gradients Found",
    description="No gradients were found in the model during backward pass.",
    remediation="Ensure `loss.backward()` was called and parameters are not frozen.",
    category="Stability",
    default_severity=Severity.ERROR,
)

RuleRegistry.register(TA100_NAN_INF)
RuleRegistry.register(TA102_GRAD_EXPLOSION)
RuleRegistry.register(TA103_DEAD_UNITS)
RuleRegistry.register(TA104_NO_GRADS)


class StabilityValidator(BaseValidator):
    def __init__(self, max_grad_norm: float = 100.0):
        self.max_grad_norm = max_grad_norm

    @property
    def rule(self):
        return TA100_NAN_INF

    @property
    def rules(self):
        return [TA100_NAN_INF, TA102_GRAD_EXPLOSION, TA103_DEAD_UNITS, TA104_NO_GRADS]

    @property
    def supported_phases(self):
        # Stability checks are meaningful for:
        #  - STATIC: one-shot parameter sanity (NaN/Inf in weights)
        #  - BACKWARD: gradient health
        #  - OPTIMIZER: post-step parameter/gradient sanity
        # INIT is intentionally skipped to avoid duplicate findings when users
        # run both static + init at startup.
        return {Phase.STATIC, Phase.BACKWARD, Phase.OPTIMIZER}

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        check_weights = context.phase in [Phase.STATIC, Phase.OPTIMIZER]
        check_grads = context.phase in [Phase.BACKWARD, Phase.OPTIMIZER]

        if not (check_weights or check_grads):
            return

        total_grad_norm_sq = 0.0
        has_grads = False

        with torch.no_grad():
            for name, param in context.model.named_parameters():
                if "." in name:
                    mod_path, param_name = name.rsplit(".", 1)
                else:
                    mod_path, param_name = "global", name

                # 1. Check Weights (TA100)
                if check_weights:
                    if not isfinite_all(param):
                        yield Finding(
                            rule_id=TA100_NAN_INF.id,
                            message=f"Parameter '{name}' contains NaNs or Infs.",
                            severity=TA100_NAN_INF.default_severity,
                            module_path=mod_path,
                            entity=f"param:{param_name}",
                            step=context.step,
                            phase=context.phase,
                            metadata=tensor_summary(param),
                        )

                # 2. Check Gradients (TA100, TA103)
                if check_grads and param.grad is not None:
                    has_grads = True

                    if not isfinite_all(param.grad):
                        yield Finding(
                            rule_id=TA100_NAN_INF.id,
                            message=f"Gradient for '{name}' contains NaNs or Infs.",
                            severity=TA100_NAN_INF.default_severity,
                            module_path=mod_path,
                            entity=f"grad:{param_name}",
                            step=context.step,
                            phase=context.phase,
                            metadata={"shape": tuple(param.grad.shape)},
                        )
                        continue

                    p_norm = safe_norm(param.grad)
                    if p_norm is not None:
                        total_grad_norm_sq += p_norm**2

                    if p_norm == 0.0:
                        yield Finding(
                            rule_id=TA103_DEAD_UNITS.id,
                            message=f"Gradient for '{name}' is exactly zero (Dead Unit/Layer).",
                            severity=TA103_DEAD_UNITS.default_severity,
                            module_path=mod_path,
                            entity=f"grad:{param_name}",
                            step=context.step,
                            phase=context.phase,
                        )

            # 3. Finalize Global Checks
            if check_grads:
                if has_grads:
                    global_norm = total_grad_norm_sq**0.5
                    if global_norm > self.max_grad_norm:
                        yield Finding(
                            rule_id=TA102_GRAD_EXPLOSION.id,
                            message=f"Global gradient norm ({global_norm:.2f}) exceeds threshold ({self.max_grad_norm}).",
                            severity=TA102_GRAD_EXPLOSION.default_severity,
                            step=context.step,
                            phase=context.phase,
                            metadata={"global_norm": global_norm},
                        )
                else:
                    # TA104: No gradients found in the entire model
                    yield Finding(
                        rule_id=TA104_NO_GRADS.id,
                        message="No gradients found in the model. Did you forget `loss.backward()`?",
                        severity=TA104_NO_GRADS.default_severity,
                        step=context.step,
                        phase=context.phase,
                    )
