.PHONY: audit

# Scan requirements.txt for known-vulnerable dependency versions.
# Findings are reported only — upgrades are a deliberate, separate decision
# since some pins (celery, redis, torch, etc.) are intentional.
audit:
	pip install --quiet pip-audit
	pip-audit -r requirements.txt
