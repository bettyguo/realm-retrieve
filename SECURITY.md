# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |
| < 1.0   | ❌        |

We backport security fixes to the latest minor release of the current
major version only.

## Reporting a Vulnerability

**Please do not open a public issue for security-sensitive reports.**

Email **bettyguo@connect.hku.hk** with:

- a description of the vulnerability and the affected component,
- the steps required to reproduce it,
- the impact you believe it has,
- (optional) a proposed fix or mitigation.

You can expect:

- an acknowledgement within **3 business days**,
- a status update within **10 business days**,
- a coordinated public disclosure once a fix is available — usually
  within 30 days for high-severity issues, 90 days otherwise.

We are happy to credit reporters in the release notes. If you'd prefer
to remain anonymous, just say so.

## Out of scope

ReaLM-Retrieve is a research codebase. The following are *not* treated as
vulnerabilities:

- Issues that require an attacker to already control the local Python
  environment or the ColBERT index files.
- Resource-exhaustion issues caused by feeding extremely large
  reasoning chains or unbounded retrieval `k` (documented limits apply).
- Findings in third-party dependencies — please report those upstream;
  we will track them via Dependabot.
