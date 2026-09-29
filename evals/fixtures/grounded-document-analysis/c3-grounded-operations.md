# Alder release operations note

## Release decision

The internal release codename is **Alder**. The production rollout is planned
for **October 14, 2026**, beginning at 09:30 Mountain Time. Release Engineering
owns the rollout decision.

## Staging evidence

The final staging exercise measured API p95 latency at **840 ms** with 250
concurrent sessions. The observed request error rate was 0.7 percent. These
figures describe the staging exercise only; they are not production forecasts.

## Recovery evidence

The rollback rehearsal completed in **7 minutes 40 seconds**. The rehearsal
restored the prior application image and database view without losing the
synthetic orders created during the exercise.

## Open risk

The remaining operational risk is a possible schema lock on the billing ledger
during the migration. The team will pause the rollout if the lock lasts longer
than 45 seconds.

## Scope limit

This note does not measure support demand, staffing coverage, customer
sentiment, or customer satisfaction.
