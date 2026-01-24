from .core import Rule


class RuleRegistry:
    _rules: dict[str, Rule] = {}

    @classmethod
    def register(cls, rule: Rule):
        if rule.id in cls._rules:
            existing = cls._rules[rule.id]
            if existing == rule:
                return
            raise ValueError(f"Rule ID conflict: {rule.id}")
        cls._rules[rule.id] = rule

    @classmethod
    def all_rules(cls) -> list[Rule]:
        return sorted(cls._rules.values(), key=lambda r: r.id)

    @classmethod
    def get(cls, rule_id: str) -> Rule | None:
        return cls._rules.get(rule_id)

    @classmethod
    def clear(cls):
        """Clears the registry. Useful for unit testing isolation."""
        cls._rules.clear()
