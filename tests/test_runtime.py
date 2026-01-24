import torch
import torch.nn as nn


def test_auditor_runs_stateful_validators_without_manual_attach():
    from torch_audit.runtime import Auditor
    from torch_audit.validators.builtin.activation import (
        TA105_ACTIVATION_COLLAPSE,
        ActivationValidator,
    )

    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.fill_(-1.0)

    auditor = Auditor(model, validators=[ActivationValidator(threshold=0.9)])

    with auditor:
        _ = auditor.forward(torch.ones(2, 5))
        result = auditor.finish()

    assert any(f.rule_id == TA105_ACTIVATION_COLLAPSE.id for f in result.findings)


def test_auditor_graph_validator_zombie_detection():
    from torch_audit.runtime import Auditor
    from torch_audit.validators.builtin.graph import TA500_UNUSED_LAYER, GraphValidator

    class ZombieModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(1, 1)
            self.unused = nn.Linear(1, 1)

        def forward(self, x):
            return self.used(x)

    model = ZombieModel()
    auditor = Auditor(model, validators=[GraphValidator()])
    with auditor:
        _ = auditor.forward(torch.randn(1, 1))
        result = auditor.finish()

    assert any(f.rule_id == TA500_UNUSED_LAYER.id for f in result.findings)


def test_autopatch_mode_runs_forward_audit_and_restores_methods():
    from torch_audit.runtime import autopatch
    from torch_audit.validators.builtin.activation import (
        TA105_ACTIVATION_COLLAPSE,
        ActivationValidator,
    )

    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.fill_(-1.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Capture the underlying implementation so we can verify restoration.
    orig_forward_func = model.forward.__func__
    orig_step_func = optimizer.step.__func__

    auditor = autopatch(
        model,
        optimizer=optimizer,
        validators=[ActivationValidator(threshold=0.9)],
        every_n_steps=1,
        run_static=False,
        run_init=False,
    )

    # A normal loop (no wrappers)
    out = model(torch.ones(2, 5))
    loss = out.sum()
    loss.backward()
    optimizer.step()

    result = auditor.finish()
    assert any(f.rule_id == TA105_ACTIVATION_COLLAPSE.id for f in result.findings)

    auditor.unpatch()

    # Model/optimizer methods should be restored.
    assert hasattr(model.forward, "__func__")
    assert model.forward.__func__ == orig_forward_func

    assert hasattr(optimizer.step, "__func__")
    assert optimizer.step.__func__ == orig_step_func


def test_audit_step_decorator_advances_step_and_gates_by_every_n_steps():
    from torch_audit.core import Finding, Phase, Rule, Severity
    from torch_audit.runtime import Auditor, audit_step
    from torch_audit.validator import BaseValidator

    class CountingValidator(BaseValidator):
        @property
        def rule(self):
            return Rule(
                "CNT001",
                "Counting",
                "desc",
                "fix",
                "Test",
                Severity.WARN,
            )

        @property
        def supported_phases(self):
            return {Phase.OPTIMIZER}

        def check(self, context):
            yield Finding(
                rule_id=self.rule.id,
                message="count",
                severity=Severity.WARN,
            )

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    auditor = Auditor(
        model,
        optimizer=optimizer,
        validators=[CountingValidator()],
        every_n_steps=2,
    )

    @audit_step(
        auditor,
        every_n_steps=2,
        batch_extractor=lambda batch: batch,
    )
    def train_step(batch):
        out = model(batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    with auditor:
        for _ in range(5):
            train_step(torch.randn(4, 2))

        result = auditor.finish()

    # Step should advance once per wrapped call.
    assert auditor.step == 5

    # Audits should run at optimizer steps 0, 2, 4 (every 2 steps).
    cnt_findings = [f for f in result.findings if f.rule_id == "CNT001"]
    assert len(cnt_findings) == 3
    assert [f.step for f in cnt_findings] == [0, 2, 4]
    assert all(f.phase == Phase.OPTIMIZER for f in cnt_findings)


def test_auditor_attach_exception_does_not_crash_and_records_internal_error():
    from torch_audit.core import Phase, Rule, Severity
    from torch_audit.runtime import Auditor
    from torch_audit.validator import BaseValidator

    class BadAttachValidator(BaseValidator):
        @property
        def rule(self):
            return Rule(
                "BAD001",
                "Bad Attach",
                "desc",
                "fix",
                "Test",
                Severity.WARN,
            )

        @property
        def supported_phases(self):
            return {Phase.FORWARD}

        def attach(self, model):
            raise RuntimeError("boom in attach")

        def check(self, context):
            if False:
                yield  # pragma: no cover

    model = nn.Linear(2, 2)
    auditor = Auditor(model, validators=[BadAttachValidator()])

    with auditor:
        _ = auditor.forward(torch.randn(1, 2))
        result = auditor.finish()

    assert any(f.rule_id == "TA000" for f in result.findings)


def test_auditor_phase_hook_exception_does_not_crash_and_records_internal_error():
    from torch_audit.core import Phase, Rule, Severity
    from torch_audit.runtime import Auditor
    from torch_audit.validator import BaseValidator

    class BadHookValidator(BaseValidator):
        @property
        def rule(self):
            return Rule(
                "BAD002",
                "Bad Hook",
                "desc",
                "fix",
                "Test",
                Severity.WARN,
            )

        @property
        def supported_phases(self):
            return {Phase.FORWARD}

        def on_phase_start(self, context):
            raise RuntimeError("boom in on_phase_start")

        def check(self, context):
            if False:
                yield  # pragma: no cover

    model = nn.Linear(2, 2)
    auditor = Auditor(model, validators=[BadHookValidator()])

    with auditor:
        _ = auditor.forward(torch.randn(1, 2))
        result = auditor.finish()

    assert any(f.rule_id == "TA000" for f in result.findings)
