# Security and Privacy Reporting

This application handles human-participant research and compensation-administration data. Do not include participant data, anonymous SONA codes, payment information, consent records, database files, exports, credentials, or PINs in a public issue, pull request, email thread, or chat message.

## Reporting a problem

Report security and privacy incidents privately to the repository owner and the institutionally designated information-security, data-protection, or research-ethics contact. Include only the minimum information necessary to identify the affected repository, commit, or file. Do not attach the sensitive file itself.

If the repository has GitHub private vulnerability reporting enabled, it may be used for application vulnerabilities that do not require uploading study data. Do not open a public GitHub issue for a data or credential exposure.

## Accidentally committed participant data

1. Stop further pushes, forks, downloads, and deployments.
2. Keep the repository private and restrict access immediately.
3. Notify the repository owner and the approved institutional privacy or research-ethics contact.
4. Record the affected commit IDs and file paths without copying participant content into the report.
5. Follow institutional incident-response instructions for containment, history removal, notification, and documentation.

Deleting a file in a new commit does not remove it from Git history. Treat history rewriting and force-pushing as coordinated incident-response actions, not routine cleanup.

## Exposed credentials or experimenter PINs

Revoke or rotate the affected credential immediately. Replace an exposed PIN hash with a newly generated salted hash, invalidate relevant sessions where applicable, inspect access logs, and report the exposure privately. Never reuse an exposed PIN or token.

## Database, compensation data, or export committed by mistake

Treat any SQLite database, compensation/SONA record, CSV/JSON export, robot log, backup, or related sidecar as potentially containing participant data. Follow the participant-data incident process above even if the file was committed only to a private repository. Do not assume that deleting the branch or repository removes existing clones or caches.

## Other vulnerabilities

Privately report reproduction steps, impact, affected versions or commits, and a proposed mitigation when available. Avoid testing against real participant data; use temporary synthetic databases from the test suite.

## Preventive controls

- Run `python scripts/install_git_hooks.py` after cloning.
- Run `python scripts/check_before_commit.py` before every commit.
- Verify `git status` and the staged diff before every push.
- Store runtime data only in ignored, institutionally approved local storage.
- Keep the GitHub repository private throughout the study.
