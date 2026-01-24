import json
import os
import traceback
import warnings

from .config import AuditConfig
from .context import AuditContext
from .core import AuditResult, Finding, Rule, Severity
from .registry import RuleRegistry
from .utils.sanitize import sanitize_metadata
from .validator import BaseValidator

INTERNAL_ERROR_RULE = Rule(
    id="TA000",
    title="Internal Validator Error",
    description="A validator crashed during execution.",
    remediation="Check stack trace.",
    category="Internal",
    default_severity=Severity.WARN,
)
RuleRegistry.register(INTERNAL_ERROR_RULE)


class AuditRunner:
    def __init__(self, config: AuditConfig, validators: list[BaseValidator]):
        self.config = config
        self.validators = validators
        self.findings: list[Finding] = []
        self.suppressed_count: int = 0

        # 1. Build Rule Map & Detect Duplicates
        self.rules_map: dict[str, Rule] = {}
        for v in validators:
            for r in v.rules:
                if r.id in self.rules_map:
                    raise ValueError(
                        f"Duplicate Rule ID detected: '{r.id}'. "
                        f"Claimed by {self.rules_map[r.id]} and {r}."
                    )
                self.rules_map[r.id] = r

        self.rules_map[INTERNAL_ERROR_RULE.id] = INTERNAL_ERROR_RULE

        self.baseline_fingerprints: set[str] = set()
        if self.config.baseline_file and not self.config.update_baseline:
            self._load_baseline()

    def _load_baseline(self):
        if not os.path.exists(self.config.baseline_file):
            return
        try:
            with open(self.config.baseline_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.baseline_fingerprints = set(data)
        except Exception as e:
            warnings.warn(
                f"Failed to load baseline file '{self.config.baseline_file}': {e}. Proceeding without baseline.",
                RuntimeWarning,
                stacklevel=2,
            )

    def run_step(
        self, context: AuditContext, validators: list[BaseValidator] | None = None
    ):
        """Run validators for a given context.

        Args:
            context: The audit context for the current phase/step.
            validators: Optional subset of validators to execute. If None,
                runs the runner's full validator list.
        """
        selected = validators if validators is not None else self.validators

        active_validators = [
            v
            for v in selected
            if v.supported_phases is None or context.phase in v.supported_phases
        ]

        for validator in active_validators:
            try:
                for finding in validator.check(context):
                    # 2. Runtime Contract: Validator must emit one of its declared Rule IDs
                    if finding.rule_id not in validator.emits_rule_ids:
                        raise ValueError(
                            f"Validator '{type(validator).__name__}' declared rules {sorted(validator.emits_rule_ids)} "
                            f"but illegally emitted finding with Rule ID '{finding.rule_id}'."
                        )

                    # Normalize finding context. Many validators omit phase/step.
                    # The runner is the source of truth for the current execution phase.
                    finding.phase = context.phase
                    if finding.step is None:
                        finding.step = context.step

                    self._process_finding(finding)
            except Exception as e:
                self._handle_crash(validator, e, context)

    def _process_finding(self, finding: Finding):
        # 1. Check Rule Filtering (--select / --ignore)
        if not self.config.is_rule_allowed(finding.rule_id):
            return

        # 2. Check Suppressions
        is_suppressed = self.config.is_suppressed(finding)

        if is_suppressed:
            self.suppressed_count += 1
            if self.config.show_suppressed:
                finding.suppressed = True
                # Sanitize and store even if suppressed
                finding.metadata = sanitize_metadata(finding.metadata)
                self.findings.append(finding)
        else:
            finding.metadata = sanitize_metadata(finding.metadata)
            self.findings.append(finding)

    def _handle_crash(
        self, validator: BaseValidator, e: Exception, context: AuditContext
    ):
        if self.config.suppress_internal_errors:
            return

        crash_finding = Finding(
            rule_id=INTERNAL_ERROR_RULE.id,
            message=f"Validator {validator.rule.id} crashed: {str(e)}",
            severity=INTERNAL_ERROR_RULE.default_severity,
            metadata={"traceback": traceback.format_exc()},
        )
        crash_finding.phase = context.phase
        crash_finding.step = context.step
        self._process_finding(crash_finding)

    def finish(self) -> AuditResult:
        # 3. Calculate Severity Stats (exclude suppressed unless active)
        # Note: self.findings contains active findings AND (suppressed findings IF show_suppressed=True)

        active_findings = [f for f in self.findings if not f.suppressed]

        max_sev_all = Severity.INFO
        if active_findings:
            max_sev_all = max(f.severity for f in active_findings)

        new_findings = []
        for f in active_findings:
            if f.get_fingerprint() not in self.baseline_fingerprints:
                new_findings.append(f)

        max_sev_new = Severity.INFO
        if new_findings:
            max_sev_new = max(f.severity for f in new_findings)

        if self.config.update_baseline and self.config.baseline_file:
            # We baseline ALL active findings (current state)
            all_fps = sorted([f.get_fingerprint() for f in active_findings])
            with open(self.config.baseline_file, "w") as f:
                json.dump(all_fps, f, indent=2)

        return AuditResult(
            findings=self.findings,  # Contains suppressed if requested
            exit_code=1 if self.config.is_failure(max_sev_new) else 0,
            max_severity=max_sev_new,
            max_severity_all=max_sev_all,
            new_findings_count=len(new_findings),
            baseline_applied=bool(self.baseline_fingerprints),
            suppressed_count=self.suppressed_count,
            rules=self.rules_map,
        )
