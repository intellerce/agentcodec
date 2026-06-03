"""Hardcoded production endpoints.

Single source of truth for the public AgentCodec host. SemKNN routing and
anonymous telemetry both default here. Overrides are intentionally limited:
- `AGENTCODEC_TELEMETRY=0` to disable telemetry entirely
- `AGENTCODEC_TELEMETRY_ENDPOINT` to redirect (e.g. licensed self-host)
- YAML `telemetry.endpoint:` for the same purpose, persisted in config
"""

AGENTCODEC_SERVER_URL = "https://agentcodec.intellerce.com"
DEFAULT_TELEMETRY_ENDPOINT = AGENTCODEC_SERVER_URL + "/telemetry"
